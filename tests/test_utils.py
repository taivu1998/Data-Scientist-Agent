"""Tests for utility helpers."""

import logging
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import setup_logger


class TestSetupLogger:
    """Test logger setup behavior across repeated runs."""

    def test_repeated_setup_replaces_file_handler(self, tmp_path):
        logger_name = f"test_logger_{uuid.uuid4().hex}"
        first_dir = tmp_path / "run_one"
        second_dir = tmp_path / "run_two"

        logger = setup_logger(str(first_dir), logger_name)
        first_file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(first_file_handlers) == 1
        first_path = first_file_handlers[0].baseFilename
        assert str(first_dir) in first_path

        logger = setup_logger(str(second_dir), logger_name)
        second_file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        console_handlers = [
            h
            for h in logger.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]

        assert len(second_file_handlers) == 1
        assert str(second_dir) in second_file_handlers[0].baseFilename
        assert second_file_handlers[0].baseFilename != first_path
        assert len(console_handlers) == 1
