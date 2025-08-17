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

"""
SmolAgents - A lightweight ReACT agent framework with OpenAI API integration

Quick Start:
-----------
from smolleragents import create_tool_calling_agent, OpenAIServerModel, LogLevel

# Basic usage
model = OpenAIServerModel("gpt-4o-mini")
agent = create_tool_calling_agent(model=model, tools=[])
result = agent.run("What is 2+2?")

# Control output verbosity
agent = create_tool_calling_agent(
    model=model, 
    tools=[],
    verbosity_level=LogLevel.WARNING  # Clean for production
)

Available Agent Types:
---------------------
• ToolCallingAgent: Fast, direct tool execution (recommended for most tasks)
• MultiStepAgent: Complex reasoning with planning capabilities

Verbosity Levels:
----------------
• LogLevel.WARNING: Only errors/warnings (production)
• LogLevel.INFO: Standard feedback (default)  
• LogLevel.DEBUG: Detailed output (development)

For detailed guidance, see README.md or AGENT_GUIDE.md
"""

__version__ = "1.0.0"

from .agent_types import *  # noqa: I001
from .agents import *  # Above noqa avoids a circular dependency due to cli.py
from .default_tools import *
from .local_python_executor import *
from .mcp_client import *
from .memory import *
from .models import *
from .monitoring import *
from .remote_executors import *
from .tools import *
from .utils import *
