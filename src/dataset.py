import json
import os
import logging
import pandas as pd
from typing import List, Dict, Iterator
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class AnalysisTaskDataset(Dataset):
    """
    Custom Dataset class to load the 'Golden Set' of data analysis tasks.
    Implements standard PyTorch Dataset interface for research consistency.

    The Golden Set is structured based on DSBench methodology:
    - Easy: Basic data loading and shape queries
    - Medium: Grouping, aggregation, and simple plots
    - Hard: Complex visualizations with specific requirements (log scale, etc.)
    """

    def __init__(self, benchmark_path: str):
        """
        Args:
            benchmark_path: Path to the JSON file containing tasks.
        """
        self.benchmark_path = benchmark_path
        self.tasks = self._load_tasks(benchmark_path)
        logger.info(f"Loaded {len(self.tasks)} tasks from {benchmark_path}")

    def _load_tasks(self, path: str) -> List[Dict]:
        """Load tasks from JSON file or generate dummy tasks if missing."""
        if not os.path.exists(path):
            logger.warning(f"Benchmark file not found: {path}. Using dummy tasks.")
            return self._generate_dummy_tasks()

        try:
            with open(path, "r") as f:
                tasks = json.load(f)
            return tasks
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in benchmark file: {e}")
            return self._generate_dummy_tasks()

    def _generate_dummy_tasks(self) -> List[Dict]:
        """
        Returns a synthetic Golden Set for testing and reproducibility.

        Task structure:
        - id: Unique identifier
        - query: Natural language data analysis request
        - csv_name: Dataset filename
        - type: Expected output type (text, plot, plot_log_check)
        - difficulty: easy/medium/hard
        - expected_output_contains: List of strings that should be in output
        - expected_visual: Type of visualization expected
        """
        return [
            # Easy tasks (3)
            {
                "id": "easy_1",
                "query": "Load the data and tell me the number of rows and columns.",
                "csv_name": "salaries.csv",
                "type": "text",
                "difficulty": "easy",
                "expected_output_contains": ["25", "6"],
            },
            {
                "id": "easy_2",
                "query": "What are the column names in this dataset?",
                "csv_name": "salaries.csv",
                "type": "text",
                "difficulty": "easy",
                "expected_output_contains": ["Job Title", "Salary", "Experience", "Department"],
            },
            {
                "id": "easy_3",
                "query": "Show me the first 5 rows of the data.",
                "csv_name": "salaries.csv",
                "type": "text",
                "difficulty": "easy",
                "expected_output_contains": ["Software Engineer", "Data Scientist"],
            },
            # Medium tasks (4)
            {
                "id": "medium_1",
                "query": "Group by 'Department' and calculate the average salary for each group. Show the results.",
                "csv_name": "salaries.csv",
                "type": "text",
                "difficulty": "medium",
                "expected_output_contains": ["Engineering", "Analytics"],
            },
            {
                "id": "medium_2",
                "query": "Create a bar chart showing average salary by Department.",
                "csv_name": "salaries.csv",
                "type": "plot",
                "difficulty": "medium",
                "expected_visual": "bar_chart",
            },
            {
                "id": "medium_3",
                "query": "Plot a histogram of the Salary distribution with 10 bins.",
                "csv_name": "salaries.csv",
                "type": "plot",
                "difficulty": "medium",
                "expected_visual": "histogram",
            },
            {
                "id": "medium_4",
                "query": "Create a scatter plot of Experience vs Salary.",
                "csv_name": "salaries.csv",
                "type": "plot",
                "difficulty": "medium",
                "expected_visual": "scatter",
            },
            # Hard tasks (3)
            {
                "id": "hard_1",
                "query": "Plot Salary vs Experience using a logarithmic scale on the Y-axis.",
                "csv_name": "salaries.csv",
                "type": "plot_log_check",
                "difficulty": "hard",
                "expected_visual": "scatter_log",
            },
            {
                "id": "hard_2",
                "query": "Create a box plot of Salary grouped by Department with proper title and labels.",
                "csv_name": "salaries.csv",
                "type": "plot",
                "difficulty": "hard",
                "expected_visual": "boxplot",
            },
            {
                "id": "hard_3",
                "query": "Plot a heatmap showing correlation between all numeric columns.",
                "csv_name": "salaries.csv",
                "type": "plot",
                "difficulty": "hard",
                "expected_visual": "heatmap",
            },
        ]

    def get_semantic_context(self, csv_path: str) -> str:
        """
        The 'Semantic Compressor': Profiles the CSV to fit into the Context Window.

        Instead of dumping raw rows (which would exhaust token limits), this method
        provides a schema + distribution summary that helps the LLM understand
        the data structure without seeing all the data.

        Key insight: For string columns, sampling top frequent values prevents
        hallucinations about column content (e.g., knowing "United States" vs "USA").

        Args:
            csv_path: Path to the CSV file to profile

        Returns:
            A formatted string containing the data profile
        """
        if not os.path.exists(csv_path):
            return "Context: CSV file not found locally. Cannot profile data."

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logger.error(f"Failed to read CSV {csv_path}: {e}")
            return f"Context: Error reading CSV file: {e}"

        if df.empty:
            return "Context: CSV file is empty (0 rows)."

        buffer = []
        buffer.append(f"Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        buffer.append(f"Columns: {list(df.columns)}")
        buffer.append("")
        buffer.append("Column Details:")

        for col in df.columns:
            dtype = df[col].dtype
            null_count = df[col].isnull().sum()
            null_pct = round(null_count / len(df) * 100, 1)

            if pd.api.types.is_numeric_dtype(dtype):
                # Numeric column: provide statistical summary
                try:
                    desc = df[col].describe()
                    buffer.append(
                        f"- {col} (Numeric, {dtype}): "
                        f"min={desc['min']:.2f}, max={desc['max']:.2f}, "
                        f"mean={desc['mean']:.2f}, std={desc['std']:.2f}"
                        f"{f', {null_pct}% missing' if null_count > 0 else ''}"
                    )
                except Exception:
                    buffer.append(f"- {col} (Numeric, {dtype}): Unable to compute statistics")

            elif pd.api.types.is_datetime64_any_dtype(dtype):
                # DateTime column: provide range
                try:
                    buffer.append(f"- {col} (DateTime): range [{df[col].min()} to {df[col].max()}]")
                except Exception:
                    buffer.append(f"- {col} (DateTime): Unable to compute range")

            else:
                # Categorical/Object column: provide top frequent values
                try:
                    unique_count = df[col].nunique()
                    top_k = df[col].value_counts().head(5).index.tolist()
                    # Truncate long values for readability
                    top_k_display = [
                        str(v)[:30] + "..." if len(str(v)) > 30 else str(v) for v in top_k
                    ]
                    buffer.append(
                        f"- {col} (Categorical, {unique_count} unique): "
                        f"Top values: {top_k_display}"
                        f"{f', {null_pct}% missing' if null_count > 0 else ''}"
                    )
                except Exception:
                    buffer.append(f"- {col} (Categorical): Unable to compute value counts")

        return "\n".join(buffer)

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, idx: int) -> Dict:
        return self.tasks[idx]

    def __iter__(self) -> Iterator[Dict]:
        return iter(self.tasks)

    def get_tasks_by_difficulty(self, difficulty: str) -> List[Dict]:
        """Filter tasks by difficulty level."""
        return [t for t in self.tasks if t.get("difficulty") == difficulty]

    def get_tasks_by_type(self, task_type: str) -> List[Dict]:
        """Filter tasks by output type (text, plot, etc.)."""
        return [t for t in self.tasks if t.get("type") == task_type]
