import os
import re
import logging
import sys
from typing import TypedDict, List, Optional
import operator

# Annotated was added in Python 3.9, use typing_extensions for older versions
if sys.version_info >= (3, 9):
    from typing import Annotated
else:
    from typing_extensions import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from src.sandbox import SandboxWrapper

logger = logging.getLogger(__name__)


# --- Structured Output for Visual Critic ---
class VisualCritique(BaseModel):
    """Structured output for visual verification."""

    is_valid: bool = Field(description="Whether the chart correctly answers the query")
    has_title: bool = Field(description="Whether the chart has a readable title")
    has_labels: bool = Field(description="Whether axes are properly labeled")
    has_data: bool = Field(description="Whether the chart contains visible data (not empty)")
    feedback: str = Field(description="Specific feedback for improvement if needed")


# --- State Definition ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    context_data: str
    generated_code: str
    execution_result: dict
    retry_count: int
    is_solved: bool
    original_query: str  # Preserve original query


# --- The Agent Architecture ---
class AnalystAgent:
    """
    A Graph-based Autonomous Agent for Data Analysis.
    Architecture: Plan -> Code -> Execute -> VisualVerify -> Refine -> END

    Features:
    - Stateful execution in Firecracker microVM sandbox (E2B)
    - Multimodal grounding via Visual Critic loop
    - Semantic compression of dataset context
    """

    def __init__(self, config: dict):
        self.config = config
        self.model = ChatAnthropic(
            model=config["agent"]["model_id"],
            temperature=config["agent"]["temperature"],
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
        self.visual_critic = self.model.with_structured_output(VisualCritique)
        self.sandbox_wrapper: Optional[SandboxWrapper] = None
        self.workflow = self._build_graph()

    def _build_graph(self):
        """Constructs the LangGraph State Machine."""
        workflow = StateGraph(AgentState)

        # Nodes: Plan -> Code -> Execute -> VisualVerify -> Refine
        workflow.add_node("planner", self.plan_node)
        workflow.add_node("coder", self.code_node)
        workflow.add_node("executor", self.execute_node)
        workflow.add_node("visual_critic", self.visual_critic_node)
        workflow.add_node("refiner", self.refine_node)

        workflow.set_entry_point("planner")

        workflow.add_edge("planner", "coder")
        workflow.add_edge("coder", "executor")

        # Conditional edge based on execution result
        workflow.add_conditional_edges(
            "executor",
            self.should_continue,
            {"verify": "visual_critic", "retry": "refiner", "end": END},
        )

        # Visual critic routes to refiner or end
        workflow.add_conditional_edges(
            "visual_critic", self.critic_router, {"refine": "refiner", "end": END}
        )

        # Refiner always goes back to coder for another attempt
        workflow.add_edge("refiner", "coder")

        return workflow.compile()

    def run(self, query: str, context: str, sandbox: SandboxWrapper) -> AgentState:
        """
        Main entry point to run the agent.

        Args:
            query: The user's data analysis query
            context: Semantic context extracted from the dataset
            sandbox: The E2B sandbox wrapper for code execution

        Returns:
            The final agent state after execution
        """
        self.sandbox_wrapper = sandbox

        # Initialize the DataFrame in the sandbox
        self._setup_sandbox_environment()

        initial_state: AgentState = {
            "messages": [HumanMessage(content=query)],
            "context_data": context,
            "generated_code": "",
            "execution_result": {},
            "retry_count": 0,
            "is_solved": False,
            "original_query": query,
        }

        try:
            final_state = self.workflow.invoke(initial_state)
            return final_state
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            return {
                **initial_state,
                "execution_result": {"error": str(e), "stderr": str(e)},
                "is_solved": False,
            }

    def _setup_sandbox_environment(self):
        """Initialize the sandbox with necessary imports and DataFrame loading."""
        setup_code = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure matplotlib for non-interactive backend
plt.switch_backend('Agg')

# Load the uploaded data
df = pd.read_csv('data.csv')
print(f"DataFrame loaded: {df.shape[0]} rows, {df.shape[1]} columns")
"""
        result = self.sandbox_wrapper.run_code(setup_code)
        if result.get("error"):
            logger.warning(f"Sandbox setup warning: {result.get('stderr', '')}")

    # --- Nodes ---

    def plan_node(self, state: AgentState) -> dict:
        """
        Planning node: Analyzes the request and creates an execution plan.
        Preserves the original query in state.
        """
        query = (
            state["original_query"] if state.get("original_query") else state["messages"][0].content
        )
        context = state.get("context_data", "")

        plan_prompt = f"""
Analyze this data analysis request and create a brief execution plan.

Dataset Context:
{context}

User Query: {query}

Provide a 2-3 step plan for how to answer this query with Python code.
"""
        try:
            response = self.model.invoke([HumanMessage(content=plan_prompt)])
            plan = response.content
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            plan = "Proceeding with direct code generation due to planning error."

        return {"messages": [SystemMessage(content=f"Plan: {plan}")], "original_query": query}

    def code_node(self, state: AgentState) -> dict:
        """Generates Python code based on query, context, and any previous errors."""
        query = state.get("original_query", state["messages"][0].content)
        context = state.get("context_data", "")
        prev_res = state.get("execution_result", {})
        retry_count = state.get("retry_count", 0)

        # Build error context for self-healing
        error_context = ""
        if prev_res.get("stderr"):
            error_context = f"\nPrevious Error (fix this): {prev_res['stderr']}"
        if prev_res.get("visual_feedback"):
            error_context += f"\nVisual Feedback (address this): {prev_res['visual_feedback']}"

        prompt = f"""You are an expert Python Data Analyst.

Dataset Context:
{context}

User Query: {query}
{error_context}

IMPORTANT RULES:
1. Write ONLY valid Python code - no markdown, no explanations
2. The DataFrame 'df' is already loaded from 'data.csv' - do NOT reload it
3. For plots: use plt.savefig() is NOT needed - just create the plot and call plt.show()
4. For text answers: use print() to output results
5. Handle potential errors gracefully (e.g., check column existence)
6. If creating plots, always include: title, axis labels, and legend if applicable
7. Keep code concise and focused - maximum 50 lines
8. Do NOT use external libraries beyond pandas, numpy, matplotlib, seaborn
9. Do NOT attempt to read/write files or make network calls
10. Limit data processing to reasonable size (no full dataset dumps)

Write the Python code now:
"""

        try:
            response = self.model.invoke([HumanMessage(content=prompt)])
            code = self._extract_code(response.content)
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            code = f"print('Code generation error: {e}')"

        return {
            "generated_code": code,
            "retry_count": retry_count + 1,
            "messages": [AIMessage(content=f"Generated code:\n```python\n{code}\n```")],
        }

    def execute_node(self, state: AgentState) -> dict:
        """Runs code in the Firecracker Sandbox."""
        code = state.get("generated_code", "")

        if not code:
            return {
                "execution_result": {
                    "error": "No code to execute",
                    "stdout": "",
                    "stderr": "No code was generated",
                    "image_base64": None,
                }
            }

        try:
            result = self.sandbox_wrapper.run_code(code)
        except Exception as e:
            logger.error(f"Sandbox execution failed: {e}")
            result = {"error": str(e), "stdout": "", "stderr": str(e), "image_base64": None}

        return {"execution_result": result}

    def visual_critic_node(self, state: AgentState) -> dict:
        """
        The Research Twist: Multimodal Visual Verification.
        Uses VLM with structured output to verify that generated charts actually answer the query.
        """
        if not self.config["agent"].get("enable_visual_critic", True):
            return {"is_solved": True}

        image_data = state["execution_result"].get("image_base64")
        if not image_data:
            return {
                "execution_result": {
                    **state["execution_result"],
                    "visual_feedback": "Warning: No image was generated for a visual query.",
                },
                "is_solved": False,
            }

        query = state.get("original_query", "")

        critique_prompt = f"""You are a data visualization critic. Analyze this chart and provide structured feedback.

Original Query: "{query}"

Evaluate and return JSON with these exact fields:
- is_valid: boolean - Does the chart correctly answer the query?
- has_title: boolean - Does the chart have a clear, readable title?
- has_labels: boolean - Are the axes properly labeled?
- has_data: boolean - Is there visible data (not empty/blank)?
- feedback: string - Specific, actionable feedback if improvements are needed"""

        try:
            msg = HumanMessage(
                content=[
                    {"type": "text", "text": critique_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}"},
                    },
                ]
            )

            critique = self.visual_critic.invoke([msg])

            if not critique.is_valid:
                return {
                    "execution_result": {
                        **state["execution_result"],
                        "visual_feedback": (
                            f"Title: {critique.has_title}, Labels: {critique.has_labels}, "
                            f"Data: {critique.has_data}. Feedback: {critique.feedback}"
                        ),
                    },
                    "is_solved": False,
                }

            return {
                "execution_result": {
                    **state["execution_result"],
                    "visual_feedback": f"Validated: {critique.feedback}",
                },
                "is_solved": True,
            }

        except Exception as e:
            logger.error(f"Visual critic failed: {e}")
            return {"is_solved": True}

    def refine_node(self, state: AgentState) -> dict:
        """
        Refine node: Analyzes errors and prepares context for code regeneration.
        This is the feedback integration step before retrying.
        """
        exec_result = state.get("execution_result", {})
        retry_count = state.get("retry_count", 0)

        # Compile all feedback
        error_summary = []

        if exec_result.get("error"):
            error_summary.append(f"Runtime Error: {exec_result['error']}")
        if exec_result.get("stderr"):
            error_summary.append(f"Stderr: {exec_result['stderr']}")
        if exec_result.get("visual_feedback"):
            error_summary.append(f"Visual Feedback: {exec_result['visual_feedback']}")

        feedback_msg = " | ".join(error_summary) if error_summary else "Unknown issue"

        logger.info(f"Refine node (attempt {retry_count}): {feedback_msg}")

        return {"messages": [SystemMessage(content=f"Refinement needed: {feedback_msg}")]}

    # --- Edge Routing Functions ---

    def should_continue(self, state: AgentState) -> str:
        """Determines next step after execution."""
        res = state.get("execution_result", {})
        retry_count = state.get("retry_count", 0)
        max_retries = self.config["agent"].get("max_retries", 3)

        # Check for errors
        if res.get("error") or res.get("stderr"):
            if retry_count < max_retries:
                return "retry"
            return "end"

        # If an image was generated, verify it
        if res.get("image_base64"):
            return "verify"

        # Text-only output, mark as solved
        return "end"

    def critic_router(self, state: AgentState) -> str:
        """Routes after visual criticism."""
        is_solved = state.get("is_solved", True)
        retry_count = state.get("retry_count", 0)
        max_retries = self.config["agent"].get("max_retries", 3)

        if not is_solved and retry_count < max_retries:
            return "refine"
        return "end"

    def _extract_code(self, content: str) -> str:
        """
        Extracts Python code from LLM response.
        Handles markdown code blocks and raw code.
        """
        if not content:
            return ""

        # Try to extract from markdown code block
        patterns = [
            r"```python\s*(.*?)\s*```",  # ```python ... ```
            r"```\s*(.*?)\s*```",  # ``` ... ```
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                return match.group(1).strip()

        # If no code block found, assume entire content is code
        # Remove any leading/trailing explanation text
        lines = content.strip().split("\n")
        code_lines = []
        in_code = False

        for line in lines:
            # Skip obvious explanation lines
            if line.strip().startswith("#") or any(
                line.strip().lower().startswith(word)
                for word in ["here", "this", "the", "i ", "note:", "output:"]
            ):
                if not in_code:
                    continue

            # Detect start of actual code
            if any(keyword in line for keyword in ["import ", "df", "plt.", "print(", "pd.", "="]):
                in_code = True

            if in_code:
                code_lines.append(line)

        return "\n".join(code_lines).strip() if code_lines else content.strip()
