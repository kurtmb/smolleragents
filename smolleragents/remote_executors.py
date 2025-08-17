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
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from .local_python_executor import PythonExecutor
from .monitoring import LogLevel
from .tools import Tool, get_tools_definition_code
from .utils import AgentError, _is_package_available

logger = logging.getLogger(__name__)


class RemotePythonExecutor(PythonExecutor):
    """Base class for remote Python executors."""
    
    def __init__(self, additional_imports: list[str], logger):
        self.additional_imports = additional_imports
        self.logger = logger
        self.logger.log("Initializing executor, hold on...")
        self.final_answer_pattern = re.compile(r"^final_answer\((.*)\)$", re.M)
        self.installed_packages = []

    def run_code_raise_errors(self, code: str, return_final_answer: bool = False) -> tuple[Any, str]:
        """Run code and raise errors. To be implemented by subclasses."""
        raise NotImplementedError

    def send_tools(self, tools: dict[str, Tool]):
        """Send tools to the executor."""
        # Install tool packages
        packages_to_install = {
            pkg
            for tool in tools.values()
            for pkg in tool.to_dict()["requirements"]
            if pkg not in self.installed_packages + ["smolleragents"]
        }
        if packages_to_install:
            self.installed_packages += self.install_packages(list(packages_to_install))
        # Get tool definitions
        code = get_tools_definition_code(tools)
        if code:
            execution = self.run_code_raise_errors(code)
            self.logger.log(execution[1])

    def send_variables(self, variables: dict):
        """Send variables to the kernel namespace."""
        # This would need to be implemented by subclasses
        pass

    def __call__(self, code_action: str) -> tuple[Any, str, bool]:
        """Check if code is a final answer and run it accordingly"""
        is_final_answer = bool(self.final_answer_pattern.search(code_action))
        output = self.run_code_raise_errors(code_action, return_final_answer=is_final_answer)
        return output[0], output[1], is_final_answer

    def install_packages(self, additional_imports: list[str]):
        """Install packages. To be implemented by subclasses."""
        if additional_imports:
            _, execution_logs = self.run_code_raise_errors(f"!pip install {' '.join(additional_imports)}")
            self.logger.log(execution_logs)
        return additional_imports


class E2BExecutor(RemotePythonExecutor):
    """Placeholder for E2B executor. Requires e2b_code_interpreter package."""
    
    def __init__(self, additional_imports: list[str], logger, **kwargs):
        raise ModuleNotFoundError(
            "Please install 'e2b' extra to use E2BExecutor: `pip install 'smolleragents[e2b]'`"
        )


class DockerExecutor(RemotePythonExecutor):
    """Placeholder for Docker executor. Requires docker and websocket packages."""
    
    def __init__(self, additional_imports: list[str], logger, **kwargs):
        raise ModuleNotFoundError(
            "Please install 'docker' extra to use DockerExecutor: `pip install 'smolleragents[docker]'`"
        )
