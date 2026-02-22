# Auto-Analyst: Multimodal Grounding for Self-Correcting Data Analysis Agents

<p align="center">
  <img src="https://img.shields.io/badge/Version-0.1.0-blue?style=flat-square" alt="Version">
  <img src="https://img.shields.io/badge/Python-3.10+-green?style=flat-square" alt="Python">
  <img src="https://img.shields.io/badge/Framework-LangGraph-purple?style=flat-square" alt="Framework">
  <img src="https://img.shields.io/badge/Sandbox-E2B%20Firecracker-red?style=flat-square" alt="Sandbox">
  <img src="https://img.shields.io/badge/License-MIT-orange?style=flat-square" alt="License">
</p>

<p align="center">
  <b>A research prototype that closes the visual grounding gap in LLM-powered data analysis through multimodal self-correction.</b>
</p>

---

## Abstract

LLM-based code generation agents have shown strong performance on tabular data analysis tasks, yet they exhibit a systematic failure mode on visualization tasks: **silent chart hallucination**. Unlike execution errors (which raise exceptions and can be trivially detected), hallucinated charts produce code that runs without errors but renders empty axes, mislabeled scales, truncated data, or semantically incorrect plot types. Standard ReAct-style agent loops that rely solely on execution status (exit code, stderr) are structurally unable to detect these failures, since the code *succeeds* from the interpreter's perspective.

**Auto-Analyst** addresses this gap by introducing a **Visual Critic** node into the agent's state machine -- a VLM-in-the-loop verifier that inspects rendered chart images and provides structured, actionable feedback for self-correction. The architecture is implemented as a LangGraph directed acyclic graph (DAG) with conditional routing, Pydantic-validated VLM outputs, sandboxed code execution via E2B Firecracker microVMs, and a semantic compression layer for context-efficient dataset representation. On our curated benchmark of 10 data analysis tasks across 3 difficulty tiers, the visual feedback loop demonstrates measurable improvement between Pass@1 and Pass@3, confirming that multimodal grounding enables recovery from failures invisible to text-only verification.

---

## Table of Contents

- [Motivation and Problem Statement](#motivation-and-problem-statement)
- [Architecture](#architecture)
- [Technical Deep Dive](#technical-deep-dive)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Evaluation Framework](#evaluation-framework)
- [Benchmark: The Golden Set](#benchmark-the-golden-set)
- [Testing](#testing)
- [Development](#development)
- [Dependencies](#dependencies)
- [Citation](#citation)
- [License](#license)

---

## Motivation and Problem Statement

Consider the following scenario: an LLM agent is asked to *"Create a bar chart of average salary by department."* The agent generates syntactically valid Python code, the sandbox executes it with exit code 0, and a PNG file is produced. A standard agent loop would mark this task as solved. But the chart is empty -- the `groupby` aggregation was applied to the wrong column, or `plt.show()` was called before `plt.savefig()`, or the DataFrame was inadvertently filtered to zero rows.

This class of failure is **pervasive and undetectable** by text-only verification:

| Failure Mode | Exception Raised? | Detectable by stderr? | Detectable by VLM? |
|:---|:---:|:---:|:---:|
| Empty axes (no data rendered) | No | No | **Yes** |
| Wrong plot type (histogram instead of bar) | No | No | **Yes** |
| Missing axis labels / title | No | No | **Yes** |
| Truncated or clipped data | No | No | **Yes** |
| Incorrect scale (linear vs. log) | No | No | **Yes** |
| Correct chart | No | No | **Yes** |

The core insight is that **visualization correctness is a visual property** -- it can only be verified by inspecting the rendered output. Auto-Analyst operationalizes this insight by integrating a VLM as a first-class verification node within the agent's control flow.

---

## Architecture

### High-Level System Design

```
                            ┌─────────────────────────────────────────────────────────────┐
                            │                    LangGraph State Machine                   │
                            │                                                             │
  User Query + CSV ───────▶ │  ┌──────────┐    ┌────────┐    ┌──────────┐                │
                            │  │ Planner  │───▶│ Coder  │───▶│ Executor │                │
                            │  └──────────┘    └────────┘    └─────┬────┘                │
                            │                      ▲               │                      │
                            │                      │          ┌────▼─────┐                │
                            │                 ┌────┴───┐      │ Router   │                │
                            │                 │ Refiner│      │ (cond.)  │                │
                            │                 └────┬───┘      └────┬─────┘                │
                            │                      │               │                      │
                            │                      │    ┌──────────▼──────────┐           │
                            │                      │    │   Visual Critic     │           │
                            │                      │    │  (VLM Structured    │           │
                            │                      │    │   Output w/ Pydantic)│           │
                            │                      │    └──────────┬──────────┘           │
                            │                      │               │                      │
                            │                      │    ┌──────────▼──────────┐           │
                            │                      └────┤   Critic Router     │──▶ END   │
                            │                           │ (valid? / retries?) │           │
                            │                           └─────────────────────┘           │
                            └─────────────────────────────────────────────────────────────┘
                                                           │
                                                           │ Code execution via API
                                                           ▼
                            ┌─────────────────────────────────────────────────────────────┐
                            │                   E2B Firecracker MicroVM                   │
                            │                                                             │
                            │   ┌─────────────────────────────────────────────────────┐   │
                            │   │              Persistent Jupyter Kernel               │   │
                            │   │                                                     │   │
                            │   │  • Stateful: variables persist across executions    │   │
                            │   │  • Pre-loaded: pandas, numpy, matplotlib, seaborn   │   │
                            │   │  • Auto-capture: matplotlib figures → base64 PNG    │   │
                            │   │  • Isolated: Firecracker VM-level sandboxing        │   │
                            │   └─────────────────────────────────────────────────────┘   │
                            └─────────────────────────────────────────────────────────────┘
```

### State Machine Specification

The agent is modeled as a **LangGraph DAG** with typed state propagation. Each node reads from and writes to a shared `AgentState` that accumulates messages, code artifacts, and execution results across the graph traversal.

```python
class AgentState(TypedDict):
    messages:          Annotated[List[BaseMessage], operator.add]  # Accumulated LLM context
    context_data:      str                                         # Semantic dataset profile
    generated_code:    str                                         # Latest generated Python code
    execution_result:  dict                                        # {stdout, stderr, error, image_base64}
    retry_count:       int                                         # Current refinement iteration
    is_solved:         bool                                        # Terminal condition flag
    original_query:    str                                         # Preserved user query
```

### Node Descriptions

| Node | Role | Input | Output |
|:---|:---|:---|:---|
| **Planner** | Decomposes the user query into a 2-3 step execution plan given the dataset profile. Prevents the coder from attempting everything in a single, error-prone code block. | `original_query`, `context_data` | Plan appended to `messages` |
| **Coder** | Generates Python code that implements the plan. On retries, receives error context and VLM feedback from prior iterations, enabling targeted self-correction. | `messages` (incl. plan + errors) | `generated_code` |
| **Executor** | Submits code to the E2B sandbox. Returns structured results including stdout, stderr, exception info, and base64-encoded PNG if a matplotlib figure was produced. | `generated_code` | `execution_result` |
| **Visual Critic** | Invokes Claude's vision capabilities on the rendered PNG. Returns a Pydantic-validated `VisualCritique` with boolean checks and natural language feedback. Only activated when the executor produces an image. | `execution_result.image_base64`, `original_query` | `VisualCritique` |
| **Refiner** | Aggregates execution errors and visual feedback into a structured error report. Increments `retry_count` and routes back to Coder for another attempt. | `execution_result`, `VisualCritique` | Error context appended to `messages` |

### Conditional Routing Logic

```
Executor ──┬── has image? ──────▶ Visual Critic ──┬── is_valid? ──▶ END
           │                                       │
           ├── text-only output ──▶ END            └── retries < max? ──▶ Refiner ──▶ Coder
           │                                                │
           └── execution error ──▶ Refiner ──▶ Coder        └── retries exhausted ──▶ END
```

---

## Technical Deep Dive

### 1. Visual Critic: Structured Multimodal Verification

The Visual Critic is the core research contribution. Rather than relying on free-form VLM descriptions (which are verbose and difficult to parse programmatically), we enforce structured output via Pydantic:

```python
class VisualCritique(BaseModel):
    is_valid:   bool   # Does the chart correctly and completely answer the query?
    has_title:  bool   # Is there a descriptive, readable title?
    has_labels: bool   # Are all axes properly labeled with units where applicable?
    has_data:   bool   # Is data actually visible (non-empty axes, rendered elements)?
    feedback:   str    # Specific, actionable feedback for the coder node
```

The VLM receives a multimodal prompt containing (1) the original user query, (2) the base64-encoded chart image, and (3) instructions to evaluate against the five criteria. Using `model.with_structured_output(VisualCritique)`, the response is parsed directly into a typed Python object -- eliminating brittle regex-based extraction and ensuring the feedback loop is deterministic and machine-readable.

**Design rationale**: Boolean fields (`is_valid`, `has_title`, `has_labels`, `has_data`) enable programmatic routing decisions, while the free-text `feedback` field provides rich context for the coder's next iteration. This hybrid structure balances machine-parseable control flow with human-readable diagnostic information.

### 2. Sandbox Architecture: E2B Firecracker MicroVMs

Executing LLM-generated code requires robust isolation. Auto-Analyst uses [E2B](https://e2b.dev/) Firecracker microVMs, the same technology that powers AWS Lambda:

- **VM-level isolation**: Each sandbox runs in a dedicated Firecracker microVM, providing hardware-grade security boundaries (not container-level).
- **Persistent Jupyter kernel**: Unlike stateless execution (subprocess per code block), the kernel maintains state across invocations. Variables defined in block *n* are available in block *n+1*, enabling multi-step analysis.
- **Automatic figure capture**: The wrapper intercepts matplotlib's rendering pipeline to extract figures as base64-encoded PNGs, which are then passed to the Visual Critic.
- **Configurable timeout**: Prevents infinite loops and runaway computations (default: 45s).

```python
class SandboxWrapper:
    def upload_data(self, local_path: str) -> str          # Upload CSV to sandbox filesystem
    def run_code(self, code: str) -> dict                  # Execute code, return structured result
    def list_files(self, path: str = "/") -> List[str]     # Inspect sandbox filesystem
    def read_file(self, path: str) -> str                  # Read file from sandbox
```

### 3. Semantic Compression: Context-Efficient Dataset Representation

A naive approach to dataset grounding -- injecting raw CSV rows into the prompt -- is wasteful and often counterproductive. A 1,000-row CSV consumes thousands of tokens while providing redundant information. Auto-Analyst instead computes a **semantic profile** of the dataset:

```python
def get_semantic_context(csv_path: str) -> str:
    """
    Generates a statistical profile of the dataset that provides
    sufficient information for code generation while fitting within
    context windows.

    Output includes:
    - Shape: (n_rows, n_cols)
    - Column types and names
    - Numeric columns: min, max, mean, std, median
    - Categorical columns: unique count, top-5 most frequent values
    - Missing values: percentage per column
    """
```

This compression achieves two objectives: (1) it fits within the LLM's context window regardless of dataset size, and (2) it provides the exact information needed for correct code generation -- column names, data types, value distributions, and edge cases (nulls, cardinality).

### 4. Multi-Tier Validation System

The `OutputValidator` implements a layered validation strategy that handles the heterogeneous nature of data analysis outputs:

| Validation Layer | Mechanism | Target Task Type |
|:---|:---|:---|
| **Text output matching** | Substring search against `expected_output_contains` | Text queries (shape, column names, aggregations) |
| **Plot type detection** | Keyword inspection in generated code (`plt.bar`, `sns.histplot`, `plt.scatter`, etc.) | Visualization tasks |
| **Log scale verification** | Regex pattern matching for `plt.yscale('log')`, `ax.set_yscale('log')`, and variants | Scale-specific tasks |
| **Visual critique integration** | Structured VLM feedback via `VisualCritique.is_valid` | All visualization tasks |

### 5. Self-Healing Code Generation

The agent's refinement loop is the mechanism by which visual grounding translates into improved outputs. When the Visual Critic flags an issue, the Refiner node constructs a targeted error report:

```
Previous code produced the following issues:
- Execution: [stdout/stderr if relevant]
- Visual Critique: is_valid=False, has_data=False
- Feedback: "The chart axes are empty. The groupby operation used 'Dept'
  but the column is named 'Department'. Use df.columns to verify column names."
```

This error context is appended to the message history and passed to the Coder node, which regenerates the code with awareness of the specific failure. The loop continues for up to `max_retries` attempts (default: 3), with each iteration receiving cumulative feedback.

---

## Project Structure

```
auto-analyst/
├── src/
│   ├── model.py               # AnalystAgent: LangGraph DAG, node definitions,
│   │                          #   conditional routing, VisualCritique schema
│   ├── sandbox.py             # SandboxWrapper: E2B Firecracker integration,
│   │                          #   code execution, file I/O, PNG capture
│   ├── dataset.py             # AnalysisTaskDataset: benchmark loader, task filtering
│   │                          #   get_semantic_context(): statistical profiling
│   ├── trainer.py             # Trainer: evaluation orchestration, metric computation,
│   │                          #   granular breakdowns (difficulty x task_type)
│   ├── validation.py          # OutputValidator: text matching, plot type detection,
│   │                          #   log scale verification, VLM feedback integration
│   ├── config_parser.py       # YAML config loading with CLI override merging
│   └── utils.py               # Deterministic seeding, structured logging setup
│
├── scripts/
│   ├── evaluate.py            # Benchmark runner: iterates Golden Set, computes metrics
│   └── inference.py           # Single-query interface for interactive use
│
├── tests/                     # 56+ unit tests, integration test suite
│   ├── conftest.py            # Fixtures, custom markers (@integration, @slow)
│   ├── test_model.py          # Code extraction, state structure, critic logic
│   ├── test_dataset.py        # Dataset loading, semantic compression, filtering
│   ├── test_validation.py     # 15 tests: text/plot/log_scale/visual validation
│   ├── test_sandbox.py        # Sandbox wrapper interface tests
│   ├── test_trainer.py        # Metric computation, result serialization
│   ├── test_config.py         # Config parsing, CLI override precedence
│   └── test_integration.py    # End-to-end tests (requires API keys)
│
├── configs/
│   └── default.yaml           # Experiment configuration (model, sandbox, data, logging)
│
├── data/
│   ├── csvs/
│   │   └── salaries.csv       # Sample dataset: 25 rows, 6 columns (salary data)
│   └── golden_set.json        # Curated benchmark: 10 tasks, 3 difficulty tiers
│
├── pyproject.toml             # Package metadata, dependencies, entry points
├── requirements.txt           # Pinned dependency versions
├── Makefile                   # Development workflow automation
└── .gitignore
```

---

## Getting Started

### Prerequisites

| Requirement | Purpose |
|:---|:---|
| Python >= 3.10 | Runtime (type hints, `match` statements) |
| [Anthropic API Key](https://console.anthropic.com/) | Claude for code generation + visual critique |
| [E2B API Key](https://e2b.dev/) | Firecracker microVM sandbox |

### Installation

```bash
git clone https://github.com/your-username/auto-analyst.git
cd auto-analyst

# Create isolated environment
python -m venv venv
source venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"
# Or: make install
```

### Environment Setup

```bash
cp .env.example .env
# Add your keys:
#   ANTHROPIC_API_KEY=sk-ant-...
#   E2B_API_KEY=e2b_...
```

### Single-Query Inference

```bash
python scripts/inference.py \
    --config configs/default.yaml \
    --query "Create a bar chart of average salary by department" \
    --csv_path data/csvs/salaries.csv
```

### Benchmark Evaluation

```bash
python scripts/evaluate.py --config configs/default.yaml

# Ablation: disable visual critic
python scripts/evaluate.py --config configs/default.yaml --enable_visual_critic false
```

---

## Evaluation Framework

### Metrics

The evaluation framework (`src/trainer.py`) computes the following metrics, inspired by the [HumanEval](https://arxiv.org/abs/2107.03374) Pass@k methodology:

| Metric | Definition | Significance |
|:---|:---|:---|
| **Pass@1** | Task solved on the first attempt (no refinement) | Baseline agent capability |
| **Pass@3** | Task solved within 3 attempts (with visual feedback) | Measures refinement effectiveness |
| **Pass (Refined)** | Task solved after all allowed retries | Upper bound on agent capability |
| **Execution Success Rate** | Code runs without raising exceptions | Code generation quality (syntax + runtime) |

**Key diagnostic**: The delta between Pass@1 and Pass@3 directly quantifies the value added by the Visual Critic. A large delta indicates that the VLM feedback loop is recovering from failures that text-only verification would miss.

### Granular Breakdowns

All metrics are disaggregated along two axes:

- **Difficulty** (Easy / Medium / Hard): Isolates performance on basic data exploration vs. complex multi-step visualizations.
- **Task Type** (Text / Plot): Separates text-output tasks (where visual critique is not triggered) from visualization tasks (where the Visual Critic is the primary differentiator).

---

## Benchmark: The Golden Set

The Golden Set (`data/golden_set.json`) is a curated benchmark of 10 data analysis tasks over the included `salaries.csv` dataset (25 rows, 6 columns: Job Title, Salary, Experience, Department, Location, Education).

### Task Distribution

| Difficulty | ID | Query | Output Type | Validation Method |
|:---|:---|:---|:---|:---|
| Easy | `easy_1` | Load the data and tell me the number of rows and columns | Text | Substring match (`25`, `6`) |
| Easy | `easy_2` | What are the column names in this dataset? | Text | Substring match (`Salary`, `Department`) |
| Easy | `easy_3` | Show me the first 5 rows of the data | Text | Substring match (`Job Title`) |
| Medium | `medium_1` | Group by Department, calculate average salary | Text | Substring match |
| Medium | `medium_2` | Bar chart of average salary by Department | Plot | Plot type + VLM critique |
| Medium | `medium_3` | Histogram of Salary distribution (10 bins) | Plot | Plot type + VLM critique |
| Medium | `medium_4` | Scatter plot of Experience vs Salary | Plot | Plot type + VLM critique |
| Hard | `hard_1` | Scatter plot with log Y-axis | Plot | Log scale regex + VLM critique |
| Hard | `hard_2` | Box plot of Salary by Department | Plot | Plot type + VLM critique |
| Hard | `hard_3` | Correlation heatmap of numeric columns | Plot | Plot type + VLM critique |

### Design Principles

- **Progressive difficulty**: Easy tasks test basic pandas operations; Medium tasks require aggregation + plotting; Hard tasks demand multi-step transformations with specific rendering constraints (log scales, correlation matrices).
- **Deterministic validation**: Each task specifies explicit expected outputs or plot types, enabling fully automated scoring without human evaluation.
- **Mixed modality**: The benchmark includes both text-output and visualization tasks, allowing controlled measurement of the Visual Critic's impact on the subset of tasks where it is activated.

---

## Testing

### Running Tests

```bash
# Unit tests only (no API keys required)
make test-unit
# Or: pytest tests/ -v --ignore=tests/test_integration.py

# Full suite including integration tests (requires API keys)
make test
# Or: pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=html
```

### Test Matrix

| Module | Tests | Scope |
|:---|:---:|:---|
| `test_validation.py` | 15 | Text output matching, plot type detection (bar, histogram, scatter, heatmap, boxplot, line, pie), log scale regex, execution error handling, visual feedback integration |
| `test_model.py` | 13 | Code extraction from markdown/raw blocks, AgentState structure, visual critic routing logic |
| `test_dataset.py` | 10 | Golden Set loading, dummy task generation, semantic compression output, task filtering by difficulty and type |
| `test_sandbox.py` | 9 | SandboxWrapper interface contracts |
| `test_trainer.py` | 6 | Metric calculation correctness, granular breakdown updates, result serialization |
| `test_config.py` | 3 | YAML parsing, CLI override precedence |
| `test_integration.py` | var. | End-to-end agent execution (marked `@pytest.mark.integration`) |

---

## Development

### Makefile Targets

```bash
make install       # Install package with dev dependencies
make evaluate      # Run full benchmark evaluation
make inference     # Run single-query inference (QUERY=, CSV_PATH=)
make test          # Run complete test suite
make test-unit     # Run unit tests only (no API keys)
make lint          # Lint with ruff + check formatting with black
make format        # Auto-format with black (line-length: 100)
make clean         # Remove logs, __pycache__, .pytest_cache
```

### Configuration Reference

```yaml
# configs/default.yaml
experiment_name: "visual_critic_v1"
seed: 42

agent:
  model_id: "claude-sonnet-4-20250514"   # LLM for code generation + visual critique
  temperature: 0.1                         # Low temperature for deterministic output
  max_retries: 3                          # Maximum refinement iterations
  enable_visual_critic: true              # Toggle VLM verification (false for ablation)

sandbox:
  template: "code-interpreter-v1"          # E2B sandbox template
  timeout: 45                             # Code execution timeout (seconds)

data:
  benchmark_path: "data/golden_set.json"
  csv_dir: "data/csvs/"

logging:
  log_dir: "logs/"
  save_artifacts: true                    # Persist generated code and plot PNGs
```

All YAML fields can be overridden via CLI flags:

```bash
python scripts/evaluate.py --config configs/default.yaml \
    --model_id "claude-sonnet-4-20250514" \
    --enable_visual_critic false \
    --max_retries 5
```

---

## Dependencies

### Core Stack

| Package | Version | Role |
|:---|:---|:---|
| `langgraph` | >= 0.2.0 | DAG-based state machine for agent control flow |
| `langchain-anthropic` | >= 0.2.0 | Claude API integration (chat + vision) |
| `langchain-core` | >= 0.3.0 | Base abstractions (messages, prompts, structured output) |
| `e2b-code-interpreter` | >= 0.0.11 | Firecracker microVM sandbox for code execution |
| `pydantic` | >= 2.0.0 | Structured output validation (VisualCritique schema) |
| `pandas` | latest | DataFrame operations and dataset profiling |
| `numpy` | < 2.0.0 | Numerical computation |
| `matplotlib` | latest | Chart generation and rendering |
| `seaborn` | latest | Statistical visualization |
| `pyyaml` | latest | Configuration file parsing |
| `python-dotenv` | latest | Environment variable management |

### Development

| Package | Role |
|:---|:---|
| `pytest` + `pytest-asyncio` | Test framework with async support |
| `black` | Code formatting (line-length: 100) |
| `ruff` | Fast Python linting |
| `mypy` | Static type checking |

---

## Citation

```bibtex
@software{auto_analyst_2024,
  title   = {Auto-Analyst: Multimodal Grounding for Self-Correcting Data Analysis Agents},
  author  = {Tai Vu Duc},
  year    = {2024},
  version = {0.1.0},
  url     = {https://github.com/your-username/auto-analyst},
  note    = {Research prototype demonstrating VLM-in-the-loop visual verification
             for LLM-powered data analysis agents}
}
```

---

## License

MIT License -- see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [LangGraph](https://langchain-ai.github.io/langgraph/) -- State machine framework for agentic workflows
- [E2B](https://e2b.dev/) -- Firecracker microVM infrastructure for secure code execution
- [Anthropic](https://www.anthropic.com/) -- Claude API (text generation and vision capabilities)

---

<p align="center">
  <b>Auto-Analyst</b> v0.1.0<br>
  <i>Multimodal Grounding for Self-Correcting Data Analysis Agents</i>
</p>
