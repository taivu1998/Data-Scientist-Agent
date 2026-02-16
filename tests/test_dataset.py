"""Tests for the AnalysisTaskDataset class."""
import os
import json
import tempfile
import sys
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.dataset import AnalysisTaskDataset


class TestAnalysisTaskDataset:
    """Test suite for AnalysisTaskDataset."""

    def test_load_dummy_tasks_when_file_missing(self):
        """Should generate dummy tasks when benchmark file doesn't exist."""
        dataset = AnalysisTaskDataset("nonexistent_file.json")

        assert len(dataset) == 10  # 3 easy + 4 medium + 3 hard
        assert all('id' in task for task in dataset)
        assert all('query' in task for task in dataset)
        assert all('csv_name' in task for task in dataset)

    def test_load_tasks_from_json(self):
        """Should load tasks from a valid JSON file."""
        tasks = [
            {"id": "test_1", "query": "Test query", "csv_name": "test.csv", "type": "text"}
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(tasks, f)
            temp_path = f.name

        try:
            dataset = AnalysisTaskDataset(temp_path)
            assert len(dataset) == 1
            assert dataset[0]['id'] == "test_1"
        finally:
            os.unlink(temp_path)

    def test_get_tasks_by_difficulty(self):
        """Should filter tasks by difficulty level."""
        dataset = AnalysisTaskDataset("nonexistent.json")

        easy_tasks = dataset.get_tasks_by_difficulty('easy')
        medium_tasks = dataset.get_tasks_by_difficulty('medium')
        hard_tasks = dataset.get_tasks_by_difficulty('hard')

        assert len(easy_tasks) == 3
        assert len(medium_tasks) == 4
        assert len(hard_tasks) == 3

    def test_get_tasks_by_type(self):
        """Should filter tasks by output type."""
        dataset = AnalysisTaskDataset("nonexistent.json")

        text_tasks = dataset.get_tasks_by_type('text')
        plot_tasks = dataset.get_tasks_by_type('plot')

        assert len(text_tasks) >= 1
        assert len(plot_tasks) >= 1

    def test_semantic_context_extraction(self):
        """Should extract semantic context from CSV file."""
        # Create a temporary CSV
        df = pd.DataFrame({
            'Name': ['Alice', 'Bob', 'Charlie'],
            'Age': [25, 30, 35],
            'Salary': [50000, 60000, 70000]
        })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_path = f.name

        try:
            dataset = AnalysisTaskDataset("dummy")
            context = dataset.get_semantic_context(temp_path)

            assert "3 rows x 3 columns" in context
            assert "Name" in context
            assert "Age" in context
            assert "Salary" in context
            assert "Numeric" in context  # Age and Salary are numeric
            assert "Categorical" in context  # Name is categorical
        finally:
            os.unlink(temp_path)

    def test_semantic_context_missing_file(self):
        """Should handle missing CSV gracefully."""
        dataset = AnalysisTaskDataset("dummy")
        context = dataset.get_semantic_context("nonexistent.csv")

        assert "not found" in context.lower()

    def test_iteration(self):
        """Should support iteration over tasks."""
        dataset = AnalysisTaskDataset("nonexistent.json")

        task_ids = [task['id'] for task in dataset]
        assert len(task_ids) == 10
        assert 'easy_1' in task_ids

    def test_indexing(self):
        """Should support indexing."""
        dataset = AnalysisTaskDataset("nonexistent.json")

        task = dataset[0]
        assert 'id' in task
        assert 'query' in task
