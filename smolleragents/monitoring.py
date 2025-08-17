#!/usr/bin/env python
# coding=utf-8

# Copyright 2025 Kurt Boden. All rights reserved.
# Based on the original SmolAgents by HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

YELLOW_HEX = "#FFD700"


class LogLevel(Enum):
    """Log levels for the agent logger."""

    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3


@dataclass
class TokenUsage:
    """Token usage information."""

    input_tokens: int
    output_tokens: int


@dataclass
class Timing:
    """Timing information for a step or run."""

    start_time: float
    end_time: float | None = None

    @property
    def duration(self):
        return None if self.end_time is None else self.end_time - self.start_time

    def dict(self):
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
        }

    def __repr__(self) -> str:
        return f"Timing(start_time={self.start_time}, end_time={self.end_time}, duration={self.duration})"


class AgentLogger:
    """Simple logger for agents that prints to console."""

    def __init__(self, level: LogLevel = LogLevel.INFO):
        self.level = level

    def log(self, message: str, level: LogLevel = LogLevel.INFO) -> None:
        """Log a message if the level is sufficient."""
        if level.value >= self.level.value:
            timestamp = time.strftime("%H:%M:%S")
            level_name = level.name
            print(f"[{timestamp}] {level_name}: {message}")

    def log_task(self, content: str, subtitle: str = "", level: LogLevel = LogLevel.INFO, title: str = "") -> None:
        """Log a task."""
        if level.value >= self.level.value:
            timestamp = time.strftime("%H:%M:%S")
            level_name = level.name
            print(f"[{timestamp}] {level_name}: TASK")
            if title:
                print(f"Title: {title}")
            if subtitle:
                print(f"Subtitle: {subtitle}")
            print(f"Content: {content}")
            print("-" * 80)

    def log_rule(self, message: str, level: LogLevel = LogLevel.INFO) -> None:
        """Log a rule."""
        if level.value >= self.level.value:
            timestamp = time.strftime("%H:%M:%S")
            level_name = level.name
            print(f"[{timestamp}] {level_name}: {message}")
            print("=" * 80)


class Monitor:
    """Simple monitor for tracking agent execution."""

    def __init__(self, level: LogLevel = LogLevel.INFO):
        self.level = level
        self.reset()

    def reset(self):
        """Reset the monitor."""
        self.start_time = time.time()
        self.steps = 0
        self.total_tokens = 0

    def log_step(self, step_number: int, tokens_used: int = 0):
        """Log a step."""
        self.steps = step_number
        self.total_tokens += tokens_used
        elapsed = time.time() - self.start_time
        if self.level.value <= LogLevel.INFO.value:
            print(f"Step {step_number} completed in {elapsed:.2f}s, total tokens: {self.total_tokens}")

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the execution."""
        elapsed = time.time() - self.start_time
        return {
            "total_steps": self.steps,
            "total_tokens": self.total_tokens,
            "total_time": elapsed,
            "tokens_per_second": self.total_tokens / elapsed if elapsed > 0 else 0,
        }
