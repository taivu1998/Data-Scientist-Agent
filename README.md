# Auto-Analyst: Grounding Data Agents with Visual Verification

<p align="center">
  <img src="https://img.shields.io/badge/Version-0.1.0-blue?style=flat-square" alt="Version">
  <img src="https://img.shields.io/badge/Python-3.10+-green?style=flat-square" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-orange?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/Status-Research%20Prototype-yellow?style=flat-square" alt="Status">
</p>

## Abstract

Large Language Models (LLMs) demonstrate remarkable capabilities in code generation, yet they suffer from a critical flaw when tasked with data visualization: **hallucinated charts**. These models frequently produce code that executes without errors but generates empty, mislabeled, or misleading visualizations - a problem that remains invisible to traditional text-based execution verification.

**Auto-Analyst** introduces a novel **Multimodal Grounding** architecture that addresses this fundamental limitation. By integrating a Vision-Language Model (VLM) as a "Visual Critic" within the agent loop, our system inspects generated visualizations and provides actionable feedback for self-correction. This research prototype achieves significantly higher success rates on complex plotting tasks compared to standard ReAct-style execution loops.

---

## 🚀 Key Features

| Feature | Description |
|---------|-------------|
| **Graph-Based State Machine** | LangGraph-powered DAG with 5 nodes: Plan → Code → Execute → VisualVerify → Refine |
| **Secure Sandbox Execution** | Firecracker microVM isolation via E2B (AWS Lambda-grade security) |
| **Visual Critic Loop** | VLM-powered multimodal verification using Claude's vision capabilities |
| **Semantic Compression** | Intelligent dataset profiling that fits context windows without dumping raw data |
| **Structured Output** | Pydantic-based validation schemas for reliable output parsing |
| **Comprehensive Validation** | Text output matching, plot type detection, and log scale verification |
| **Research-Grade Evaluation** | Pass@1, Pass@3, and granular metrics by difficulty and task type |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           LangGraph State Machine                              │
│  ┌────────┐    ┌────────┐    ┌─────────┐    ┌──────────────┐    ┌─────────┐ │
│  │ Planner │───▶│ Coder  │───▶│ Executor │───▶│ VisualCritic │───▶│   END   │ │
│  └────────┘    └────────┘    └─────────┘    └──────────────┘    └─────────┘ │
│       │                                                            ▲        │
│       │                           ┌──────────┐                     │        │
│       └──────────────────────────▶│ Refiner  │─────────────────────┘        │
│                                   └──────────┘                               │
└─────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        E2B Firecracker MicroVM                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                    Persistent Jupyter Kernel                              │ │
│  │  • pandas, numpy, matplotlib, seaborn pre-loaded                        │ │
│  │  • Variables persist across code execution blocks                        │ │
│  │  • Automatic PNG capture from matplotlib figures                         │ │
│  │  • AWS Lambda-grade isolation via Firecracker microVMs                  │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Agent Flow

1. **Planner Node**: Analyzes the query and creates a 2-3 step execution plan
2. **Coder Node**: Generates Python code based on query, context, and previous errors
3. **Executor Node**: Runs code in E2B sandbox, captures stdout and matplotlib PNGs
4. **VisualCritic Node**: Uses VLM to inspect generated charts and validate correctness
5. **Refiner Node**: Aggregates errors and visual feedback for code regeneration
6. **Loop**: Cycles back to Coder for up to `max_retries` attempts

---

## 📁 Project Structure

```
auto-analyst/
├── configs/
│   └── default.yaml                    # Default configuration
├── data/
│   ├── csvs/
│   │   └── salaries.csv                # Sample employee dataset (25 rows, 6 columns)
│   └── golden_set.json                # Benchmark tasks (10 tasks: 3 easy, 4 medium, 3 hard)
├── logs/                               # Evaluation logs and results
├── scripts/
│   ├── evaluate.py                     # Full benchmark evaluation
│   └── inference.py                    # Single-query inference
├── src/
│   ├── __init__.py
│   ├── config_parser.py               # YAML config + CLI overrides
│   ├── dataset.py                     # Golden Set loader + Semantic Compressor
│   ├── model.py                       # AnalystAgent (LangGraph + VisualCritique)
│   ├── sandbox.py                     # E2B SandboxWrapper
│   ├── trainer.py                     # Evaluation framework with granular metrics
│   ├── utils.py                       # Logging and seeding utilities
│   └── validation.py                  # OutputValidator (text + plot + log scale)
├── tests/                             # 56 unit tests + integration tests
├── .env.example                       # Environment template
├── pyproject.toml                     # Project configuration
├── requirements.txt                   # Dependencies
└── Makefile                           # Common commands
```

---

## 🛠️ Quick Start

### Prerequisites

- **Python** 3.10 or higher
- **Anthropic API Key** ([get one here](https://console.anthropic.com/))
- **E2B API Key** ([get one here](https://e2b.dev/))

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/auto-analyst.git
cd auto-analyst

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
make install
# Or: pip install -e ".[dev]"
```

### Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
# ANTHROPIC_API_KEY=sk-ant-...
# E2B_API_KEY=e2b_...
```

### Run Inference

```bash
# Single query with make
make inference QUERY="Create a bar chart of average salary by department" CSV_PATH="data/csvs/salaries.csv"

# Or directly with Python
python scripts/inference.py \
    --config configs/default.yaml \
    --query "What is the average salary by department?" \
    --csv_path data/csvs/salaries.csv
```

### Run Evaluation

```bash
# Full benchmark with make
make evaluate

# Or directly with Python
python scripts/evaluate.py --config configs/default.yaml
```

---

## ⚙️ Configuration

### YAML Configuration (`configs/default.yaml`)

```yaml
experiment_name: "visual_critic_v1"
seed: 42

agent:
  model_id: "claude-sonnet-4-20250514"  # Claude model
  temperature: 0.1                        # Lower = more deterministic
  max_retries: 3                         # Max refinement attempts
  enable_visual_critic: true             # Toggle visual verification

sandbox:
  template: "code-interpreter-v1"         # E2B sandbox template
  timeout: 45                            # Execution timeout (seconds)

data:
  benchmark_path: "data/golden_set.json" # Golden Set location
  csv_dir: "data/csvs/"                 # CSV files directory

logging:
  log_dir: "logs/"                       # Output directory
  save_artifacts: true                   # Save generated code/plots
```

### CLI Overrides

```bash
# Override model
python scripts/evaluate.py --config configs/default.yaml \
    --model_id "claude-sonnet-4-20250514"

# Disable visual critic (for ablation study)
python scripts/evaluate.py --config configs/default.yaml \
    --enable_visual_critic false

# Custom query and CSV
python scripts/inference.py --config configs/default.yaml \
    --query "Show me the data distribution" \
    --csv_path data/csvs/salaries.csv
```

---

## 📊 Evaluation Metrics

The benchmark produces comprehensive metrics:

### Overall Metrics

| Metric | Description |
|--------|-------------|
| **Pass@1** | Success rate on first attempt (no retries) |
| **Pass@3** | Success rate within 3 attempts (with refinement) |
| **Pass (Refined)** | Overall success rate after all refinement attempts |
| **Execution Success** | Percentage of code that runs without exceptions |

### Granular Metrics

Metrics are also broken down by:

- **Difficulty**: Easy, Medium, Hard
- **Task Type**: Text queries, Plot queries

### Expected Results

With visual critic enabled, you should observe improvement between Pass@1 and Pass@3, demonstrating the value of the multimodal verification loop.

---

## 📋 The Golden Set

The benchmark includes **10 tasks** across three difficulty levels:

### Easy (3 tasks) - Basic Data Exploration
- `"Load the data and tell me the number of rows and columns."`
- `"What are the column names in this dataset?"`
- `"Show me the first 5 rows of the data."`

### Medium (4 tasks) - Aggregation and Basic Plotting
- `"Group by 'Department' and calculate the average salary for each group."`
- `"Create a bar chart showing average salary by Department."`
- `"Plot a histogram of the Salary distribution with 10 bins."`
- `"Create a scatter plot of Experience (x-axis) vs Salary (y-axis)."`

### Hard (3 tasks) - Complex Visualizations
- `"Plot Salary vs Experience using a logarithmic scale on the Y-axis."`
- `"Create a box plot of Salary grouped by Department with proper title and labels."`
- `"Plot a heatmap showing the correlation matrix between all numeric columns."`

---

## 🔬 Technical Implementation

### VisualCritique Structured Output

```python
class VisualCritique(BaseModel):
    is_valid: bool      # Does the chart correctly answer the query?
    has_title: bool     # Does the chart have a readable title?
    has_labels: bool    # Are axes properly labeled?
    has_data: bool      # Is there visible data (not empty)?
    feedback: str       # Specific feedback for improvement
```

The VLM returns structured JSON that we parse using Pydantic, eliminating fragile keyword matching.

### OutputValidator

The validation system checks:

1. **Text Output**: Substring matching for `expected_output_contains`
2. **Plot Type**: Code inspection for plot type keywords (bar, hist, scatter, boxplot, heatmap)
3. **Log Scale**: Regex patterns for `plt.yscale('log')` and variants
4. **Visual Feedback**: Combined validation with VLM critique

### Semantic Compression

```python
def get_semantic_context(csv_path: str) -> str:
    # Instead of dumping raw rows, profiles the dataset:
    # - Shape: (25, 6)
    # - Numeric columns: Salary, Experience (with min/max/mean/std)
    # - Categorical columns: Job Title, Department, Location, Education (with top values)
```

This approach fits context windows while providing sufficient information for accurate code generation.

---

## 🧪 Testing

```bash
# Run unit tests (no API keys required)
pytest tests/ -v --ignore=tests/test_integration.py
# 56 tests covering model, dataset, validation, trainer, sandbox, config

# Run integration tests (requires API keys)
pytest tests/test_integration.py -v -m integration

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Test Coverage

| Module | Tests | Coverage |
|--------|-------|----------|
| `model.py` | 13 | Code extraction, state structure, visual critic logic |
| `dataset.py` | 10 | Golden Set loading, semantic compression |
| `validation.py` | 15 | Text, plot, and log scale validation |
| `sandbox.py` | 9 | Sandbox wrapper interface |
| `trainer.py` | 6 | Metrics calculation, result saving |
| `config.py` | 3 | Config parsing |

---

## 🔧 Makefile Commands

```bash
make install      # Install dependencies
make evaluate     # Run benchmark evaluation
make inference    # Run single inference
make test         # Run test suite
make lint         # Run linters (ruff, black, mypy)
make clean        # Clean logs and cache
```

---

## 🐛 Troubleshooting

### API Key Issues

```bash
# Verify keys are set
echo $ANTHROPIC_API_KEY
echo $E2B_API_KEY

# Ensure .env is loaded
source .env
```

### Sandbox Timeout

Increase the timeout in `configs/default.yaml`:

```yaml
sandbox:
  timeout: 60  # Increase from 45
```

### Import Errors

Ensure you're running from the project root:

```bash
cd /path/to/auto-analyst
python scripts/evaluate.py --config configs/default.yaml
```

---

## 📦 Dependencies

### Core

- **langgraph>=0.2.0** - DAG-based state machine
- **langchain-anthropic>=0.2.0** - Claude API integration
- **e2b-code-interpreter>=0.0.11** - Secure code execution
- **pandas, numpy** - Data manipulation
- **matplotlib, seaborn** - Visualization

### Development

- **pytest>=7.0.0** - Testing framework
- **black, ruff** - Code formatting and linting
- **mypy** - Type checking

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@software{auto_analyst,
  title = {Auto-Analyst: Grounding Data Agents with Visual Verification},
  author = {Research Engineer},
  year = {2024},
  version = {0.1.0},
  url = {https://github.com/your-username/auto-analyst}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Run tests: `make test`
4. Run linting: `make lint`
5. Submit a pull request

---

## 🙏 Acknowledgments

- [LangGraph](https://langchain-ai.github.io/langgraph/) for the state machine architecture
- [E2B](https://e2b.dev/) for secure code execution infrastructure
- [Anthropic](https://www.anthropic.com/) for Claude API and vision capabilities

---

<p align="center">
  <strong>Auto-Analyst</strong> — Research Prototype v0.1.0<br>
  Grounding Data Agents with Visual Verification
</p>
