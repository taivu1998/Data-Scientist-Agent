import logging
import operator
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, List, Optional, TypedDict

# Annotated was added in Python 3.9, use typing_extensions for older versions
if sys.version_info >= (3, 9):
    from typing import Annotated
else:
    from typing_extensions import Annotated

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:  # pragma: no cover - exercised in dependency-light environments
    ChatAnthropic = None

try:
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
except ImportError:  # pragma: no cover - exercised in dependency-light environments
    @dataclass
    class BaseMessage:
        content: Any

    @dataclass
    class HumanMessage(BaseMessage):
        pass

    @dataclass
    class SystemMessage(BaseMessage):
        pass

    @dataclass
    class AIMessage(BaseMessage):
        pass

try:
    from langgraph.graph import END, StateGraph
except ImportError:  # pragma: no cover - exercised in dependency-light environments
    END = "__end__"
    StateGraph = None

from pydantic import BaseModel, Field

from src.sandbox import SandboxWrapper

logger = logging.getLogger(__name__)


class VisualCritique(BaseModel):
    """Structured output for visual verification."""

    is_valid: bool = Field(description="Whether the chart correctly answers the query")
    has_title: bool = Field(description="Whether the chart has a readable title")
    has_labels: bool = Field(description="Whether axes are properly labeled")
    has_data: bool = Field(description="Whether the chart contains visible data (not empty)")
    feedback: str = Field(description="Specific feedback for improvement if needed")


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    context_data: str
    generated_code: str
    execution_result: dict
    retry_count: int
    is_solved: bool
    original_query: str
    task_type: str


class AnalystAgent:
    """
    A Graph-based Autonomous Agent for Data Analysis.
    Architecture: Plan -> Code -> Execute -> VisualVerify -> Refine -> END
    """

    VISUAL_TASK_TYPES = {"plot", "plot_log_check"}
    VISUAL_QUERY_KEYWORDS = (
        "plot",
        "chart",
        "graph",
        "histogram",
        "scatter",
        "heatmap",
        "box plot",
        "boxplot",
        "bar chart",
        "visualize",
        "visualise",
    )

    def __init__(self, config: dict):
        self.config = config
        self._ensure_runtime_dependencies()
        self.model = ChatAnthropic(
            model=config["agent"]["model_id"],
            temperature=config["agent"]["temperature"],
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
        self.visual_critic = self.model.with_structured_output(VisualCritique)
        self.sandbox_wrapper: Optional[SandboxWrapper] = None
        self._critic_failure_mode = config["agent"].get("critic_failure_mode", "best_effort")
        self.workflow = self._build_graph()

    def _ensure_runtime_dependencies(self):
        missing = []
        if ChatAnthropic is None:
            missing.append("langchain-anthropic")
        if StateGraph is None:
            missing.append("langgraph")

        if missing:
            raise RuntimeError(
                "AnalystAgent requires runtime dependencies that are not installed: "
                + ", ".join(missing)
            )

    def _build_graph(self):
        """Constructs the LangGraph state machine."""
        workflow = StateGraph(AgentState)

        workflow.add_node("planner", self.plan_node)
        workflow.add_node("coder", self.code_node)
        workflow.add_node("executor", self.execute_node)
        workflow.add_node("visual_critic", self.visual_critic_node)
        workflow.add_node("refiner", self.refine_node)

        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "coder")
        workflow.add_edge("coder", "executor")

        workflow.add_conditional_edges(
            "executor",
            self.should_continue,
            {"verify": "visual_critic", "retry": "refiner", "end": END},
        )
        workflow.add_conditional_edges(
            "visual_critic", self.critic_router, {"refine": "refiner", "end": END}
        )
        workflow.add_edge("refiner", "coder")

        return workflow.compile()

    def run(
        self,
        query: str,
        context: str,
        sandbox: SandboxWrapper,
        task_type: Optional[str] = None,
        critic_failure_mode: Optional[str] = None,
    ) -> AgentState:
        """
        Main entry point to run the agent.

        Args:
            query: The user's data analysis query
            context: Semantic context extracted from the dataset
            sandbox: The execution sandbox wrapper
            task_type: Optional task type hint (text, plot, plot_log_check)
            critic_failure_mode: Optional override for critic failures (strict or best_effort)
        """
        self.sandbox_wrapper = sandbox
        self._critic_failure_mode = (
            critic_failure_mode or self.config["agent"].get("critic_failure_mode", "best_effort")
        )

        self._setup_sandbox_environment()

        initial_state: AgentState = {
            "messages": [HumanMessage(content=query)],
            "context_data": context,
            "generated_code": "",
            "execution_result": {},
            "retry_count": 0,
            "is_solved": False,
            "original_query": query,
            "task_type": task_type or "",
        }

        try:
            final_state = self.workflow.invoke(initial_state)
            return final_state
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            return {
                **initial_state,
                "execution_result": {
                    "status": "error",
                    "error": str(e),
                    "stderr": str(e),
                    "stdout": "",
                    "image_base64": None,
                    "warnings": [],
                },
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
        if result.get("status") == "error":
            logger.warning(f"Sandbox setup warning: {result.get('stderr', '')}")

    def plan_node(self, state: AgentState) -> dict:
        """Analyze the request and create an execution plan."""
        query = state.get("original_query") or state["messages"][0].content
        context = state.get("context_data", "")

        plan_prompt = f"""
Analyze this data analysis request and create a brief execution plan.

Dataset Context:
{context}

User Query: {query}

Provide a concise 2-3 step plan for how to answer this query with Python code.
"""
        try:
            response = self.model.invoke([HumanMessage(content=plan_prompt)])
            plan = response.content
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            plan = "Proceeding with direct code generation due to planning error."

        return {"messages": [SystemMessage(content=f"Plan: {plan}")], "original_query": query}

    def code_node(self, state: AgentState) -> dict:
        """Generate Python code based on query, context, and prior guidance."""
        query = state.get("original_query", state["messages"][0].content)
        context = state.get("context_data", "")
        prev_res = state.get("execution_result", {})
        retry_count = state.get("retry_count", 0)
        guidance = self._build_guidance_from_messages(state.get("messages", []))

        error_context = ""
        if prev_res.get("error"):
            error_context += f"\nPrevious Runtime Error (fix this): {prev_res['error']}"
        if prev_res.get("stderr") and prev_res.get("status") == "error":
            error_context += f"\nPrevious Stderr (fix this): {prev_res['stderr']}"
        if prev_res.get("visual_feedback"):
            error_context += f"\nVisual Feedback (address this): {prev_res['visual_feedback']}"

        prompt = f"""You are an expert Python Data Analyst.

Dataset Context:
{context}

User Query: {query}

Structured Guidance:
{guidance}

Attempt Number: {retry_count + 1}
{error_context}

IMPORTANT RULES:
1. Write ONLY valid Python code - no markdown, no explanations
2. The DataFrame 'df' is already loaded from 'data.csv' - do NOT reload it
3. For plots: create the plot and call plt.show()
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
        """Run code in the execution sandbox."""
        code = state.get("generated_code", "")
        expects_visual = self._expects_visual_output(
            state.get("original_query", ""), state.get("task_type", "")
        )

        if not code:
            return {
                "execution_result": {
                    "status": "error",
                    "error": "No code to execute",
                    "stdout": "",
                    "stderr": "No code was generated",
                    "image_base64": None,
                    "warnings": [],
                },
                "is_solved": False,
            }

        try:
            result = self.sandbox_wrapper.run_code(code)
        except Exception as e:
            logger.error(f"Sandbox execution failed: {e}")
            result = {
                "status": "error",
                "error": str(e),
                "stdout": "",
                "stderr": str(e),
                "image_base64": None,
                "warnings": [],
            }

        is_solved = False
        if result.get("status") != "error":
            if expects_visual:
                if result.get("image_base64") and not self.config["agent"].get(
                    "enable_visual_critic", True
                ):
                    is_solved = True
            else:
                is_solved = True

        return {"execution_result": result, "is_solved": is_solved}

    def visual_critic_node(self, state: AgentState) -> dict:
        """Use the VLM with structured output to verify generated charts."""
        if not self.config["agent"].get("enable_visual_critic", True):
            return {
                "execution_result": {
                    **state["execution_result"],
                    "critic_status": "disabled",
                    "visual_feedback": "Visual critic disabled for this run.",
                },
                "is_solved": True,
            }

        image_data = state["execution_result"].get("image_base64")
        if not image_data:
            return {
                "execution_result": {
                    **state["execution_result"],
                    "critic_status": "missing_image",
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
                        "critic_status": "invalid",
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
                    "critic_status": "passed",
                    "visual_feedback": f"Validated: {critique.feedback}",
                },
                "is_solved": True,
            }

        except Exception as e:
            logger.error(f"Visual critic failed: {e}")
            updated_result = {
                **state["execution_result"],
                "critic_status": "error",
                "critic_error": str(e),
                "visual_feedback": f"Warning: Visual critic failed: {e}",
            }
            if self._critic_failure_mode == "strict":
                return {"execution_result": updated_result, "is_solved": False}
            return {"execution_result": updated_result, "is_solved": True}

    def refine_node(self, state: AgentState) -> dict:
        """Prepare targeted retry context from execution and visual feedback."""
        exec_result = state.get("execution_result", {})
        retry_count = state.get("retry_count", 0)
        error_summary = []

        if exec_result.get("error"):
            error_summary.append(f"Runtime Error: {exec_result['error']}")
        if exec_result.get("stderr") and exec_result.get("status") == "error":
            error_summary.append(f"Stderr: {exec_result['stderr']}")
        if exec_result.get("warnings"):
            error_summary.append(f"Warnings: {' | '.join(exec_result['warnings'])}")
        if exec_result.get("visual_feedback"):
            error_summary.append(f"Visual Feedback: {exec_result['visual_feedback']}")

        feedback_msg = " | ".join(error_summary) if error_summary else "Unknown issue"
        logger.info(f"Refine node (attempt {retry_count}): {feedback_msg}")
        return {"messages": [SystemMessage(content=f"Refinement needed: {feedback_msg}")]}

    def should_continue(self, state: AgentState) -> str:
        """Determine the next step after execution."""
        res = state.get("execution_result", {})
        retry_count = state.get("retry_count", 0)
        max_retries = self.config["agent"].get("max_retries", 3)
        expects_visual = self._expects_visual_output(
            state.get("original_query", ""), state.get("task_type", "")
        )

        if res.get("status") == "error":
            if retry_count < max_retries:
                return "retry"
            return "end"

        if expects_visual:
            if res.get("image_base64"):
                if self.config["agent"].get("enable_visual_critic", True):
                    return "verify"
                return "end"
            if retry_count < max_retries:
                return "retry"
            return "end"

        return "end"

    def critic_router(self, state: AgentState) -> str:
        """Route after visual criticism."""
        is_solved = state.get("is_solved", True)
        retry_count = state.get("retry_count", 0)
        max_retries = self.config["agent"].get("max_retries", 3)

        if not is_solved and retry_count < max_retries:
            return "refine"
        return "end"

    def _build_guidance_from_messages(self, messages: List[BaseMessage]) -> str:
        guidance_blocks = []
        for message in messages:
            if not isinstance(message, SystemMessage):
                continue
            content = getattr(message, "content", "")
            if isinstance(content, str) and (
                content.startswith("Plan:") or content.startswith("Refinement needed:")
            ):
                guidance_blocks.append(content)

        if not guidance_blocks:
            return "No prior planning or refinement context."
        return "\n".join(guidance_blocks[-3:])

    def _expects_visual_output(self, query: str, task_type: str) -> bool:
        if task_type in self.VISUAL_TASK_TYPES:
            return True

        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.VISUAL_QUERY_KEYWORDS)

    def _extract_code(self, content: str) -> str:
        """Extract Python code from an LLM response."""
        if not content:
            return ""

        patterns = [
            r"```python\s*(.*?)\s*```",
            r"```\s*(.*?)\s*```",
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                return match.group(1).strip()

        lines = content.strip().split("\n")
        code_lines = []
        in_code = False

        for line in lines:
            if line.strip().startswith("#") or any(
                line.strip().lower().startswith(word)
                for word in ["here", "this", "the", "i ", "note:", "output:"]
            ):
                if not in_code:
                    continue

            if any(keyword in line for keyword in ["import ", "df", "plt.", "print(", "pd.", "="]):
                in_code = True

            if in_code:
                code_lines.append(line)

        return "\n".join(code_lines).strip() if code_lines else content.strip()
