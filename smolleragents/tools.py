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
from __future__ import annotations

import ast
import inspect
import json
import logging
import os
import sys
import tempfile
import textwrap
import types
from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._function_type_hints_utils import (
    TypeHintParsingException,
    _convert_type_hints_to_json_schema,
    get_imports,
    get_json_schema,
)
from .tool_validation import MethodChecker, validate_tool_attributes
from .utils import BASE_BUILTIN_MODULES, _is_package_available, get_source, instance_to_source, is_valid_name


logger = logging.getLogger(__name__)


def validate_after_init(cls):
    original_init = cls.__init__

    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.validate_arguments()

    cls.__init__ = new_init
    return cls


AUTHORIZED_TYPES = [
    "string",
    "boolean",
    "integer",
    "number",
    "image",
    "audio",
    "array",
    "object",
    "any",
    "null",
]

CONVERSION_DICT = {"str": "string", "int": "integer", "float": "number"}


class Tool:
    """
    A base class for the functions used by the agent. Subclass this and implement the `forward` method as well as the
    following class attributes:

    - **description** (`str`) -- A short description of what your tool does, the inputs it expects and the output(s) it
      will return. For instance 'This is a tool that downloads a file from a `url`. It takes the `url` as input, and
      returns the text contained in the file'.
    - **name** (`str`) -- A performative name that will be used for your tool in the prompt to the agent. For instance
      `"text-classifier"` or `"image_generator"`.
    - **inputs** (`Dict[str, Dict[str, Union[str, type, bool]]]`) -- The dict of modalities expected for the inputs.
      It has one `type`key and a `description`key.
      This is used by `launch_gradio_demo` or to make a nice space from your tool, and also can be used in the generated
      description for your tool.
    - **output_type** (`type`) -- The type of the tool output. This is used by `launch_gradio_demo`
      or to make a nice space from your tool, and also can be used in the generated description for your tool.

    You can also override the method [`~Tool.setup`] if your tool has an expensive operation to perform before being
    usable (such as loading a model). [`~Tool.setup`] will be called the first time you use your tool, but not at
    instantiation.
    """

    name: str
    description: str
    inputs: dict[str, dict[str, str | type | bool]]
    output_type: str

    def __init__(self, *args, **kwargs):
        self.is_initialized = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        validate_after_init(cls)

    def validate_arguments(self):
        required_attributes = {
            "description": str,
            "name": str,
            "inputs": dict,
            "output_type": str,
        }
        # Validate class attributes
        for attr, expected_type in required_attributes.items():
            attr_value = getattr(self, attr, None)
            if attr_value is None:
                raise TypeError(f"You must set an attribute {attr}.")
            if not isinstance(attr_value, expected_type):
                raise TypeError(
                    f"Attribute {attr} should have type {expected_type.__name__}, got {type(attr_value)} instead."
                )
        # - Validate name
        if not is_valid_name(self.name):
            raise Exception(
                f"Invalid Tool name '{self.name}': must be a valid Python identifier and not a reserved keyword"
            )
        # Validate inputs
        for input_name, input_content in self.inputs.items():
            assert isinstance(input_content, dict), f"Input '{input_name}' should be a dictionary."
            assert "type" in input_content and "description" in input_content, (
                f"Input '{input_name}' should have keys 'type' and 'description', has only {list(input_content.keys())}."
            )
            if input_content["type"] not in AUTHORIZED_TYPES:
                raise Exception(
                    f"Input '{input_name}': type '{input_content['type']}' is not an authorized value, should be one of {AUTHORIZED_TYPES}."
                )
        # Validate output type
        assert getattr(self, "output_type", None) in AUTHORIZED_TYPES

        # Validate forward function signature, except for Tools that use a "generic" signature (PipelineTool, SpaceToolWrapper, LangChainToolWrapper)
        if not (
            hasattr(self, "skip_forward_signature_validation")
            and getattr(self, "skip_forward_signature_validation") is True
        ):
            signature = inspect.signature(self.forward)
            actual_keys = set(key for key in signature.parameters.keys() if key != "self")
            expected_keys = set(self.inputs.keys())
            if actual_keys != expected_keys:
                raise Exception(
                    f"In tool '{self.name}', 'forward' method parameters were {actual_keys}, but expected {expected_keys}. "
                    f"It should take 'self' as its first argument, then its next arguments should match the keys of tool attribute 'inputs'."
                )

            json_schema = _convert_type_hints_to_json_schema(self.forward, error_on_missing_type_hints=False)[
                "properties"
            ]  # This function will not raise an error on missing docstrings, contrary to get_json_schema
            for key, value in self.inputs.items():
                assert key in json_schema, (
                    f"Input '{key}' should be present in function signature, found only {json_schema.keys()}"
                )
                if "nullable" in value:
                    assert "nullable" in json_schema[key], (
                        f"Nullable argument '{key}' in inputs should have key 'nullable' set to True in function signature."
                    )
                if key in json_schema and "nullable" in json_schema[key]:
                    assert "nullable" in value, (
                        f"Nullable argument '{key}' in function signature should have key 'nullable' set to True in inputs."
                    )

    def forward(self, *args, **kwargs):
        return NotImplementedError("Write this method in your subclass of `Tool`.")

    def __call__(self, *args, sanitize_inputs_outputs: bool = False, **kwargs):
        if not self.is_initialized:
            self.setup()

        # Handle the arguments might be passed as a single dictionary
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], dict):
            potential_kwargs = args[0]

            # If the dictionary keys match our input parameters, convert it to kwargs
            if all(key in self.inputs for key in potential_kwargs):
                args = ()
                kwargs = potential_kwargs

        if sanitize_inputs_outputs:
            from .agent_types import handle_agent_input_types, handle_agent_output_types
            args, kwargs = handle_agent_input_types(*args, **kwargs)
        outputs = self.forward(*args, **kwargs)
        if sanitize_inputs_outputs:
            from .agent_types import handle_agent_output_types
            outputs = handle_agent_output_types(outputs, self.output_type)
        return outputs

    def setup(self):
        """
        Overwrite this method here for any operation that is expensive and needs to be executed before you start using
        your tool. Such as loading a big model.
        """
        self.is_initialized = True

    def to_dict(self) -> dict:
        """Returns a dictionary representing the tool"""
        class_name = self.__class__.__name__
        if type(self).__name__ == "SimpleTool":
            # Check that imports are self-contained
            source_code = get_source(self.forward).replace("@tool", "")
            forward_node = ast.parse(source_code)
            # If tool was created using '@tool' decorator, it has only a forward pass, so it's simpler to just get its code
            method_checker = MethodChecker(set())
            method_checker.visit(forward_node)

            if len(method_checker.errors) > 0:
                errors = [f"- {error}" for error in method_checker.errors]
                raise (ValueError(f"SimpleTool validation failed for {self.name}:\n" + "\n".join(errors)))

            forward_source_code = get_source(self.forward)
            tool_code = textwrap.dedent(
                f"""
            from smolleragents import Tool
            from typing import Any, Optional

            class {class_name}(Tool):
                name = "{self.name}"
                description = {json.dumps(textwrap.dedent(self.description).strip())}
                inputs = {repr(self.inputs)}
                output_type = "{self.output_type}"
            """
            ).strip()
            import re

            def add_self_argument(source_code: str) -> str:
                """Add 'self' as first argument to a function definition if not present."""
                pattern = r"def forward\(((?!self)[^)]*)\)"

                def replacement(match):
                    args = match.group(1).strip()
                    if args:  # If there are other arguments
                        return f"def forward(self, {args})"
                    return "def forward(self)"

                return re.sub(pattern, replacement, source_code)

            forward_source_code = forward_source_code.replace(self.name, "forward")
            forward_source_code = add_self_argument(forward_source_code)
            forward_source_code = forward_source_code.replace("@tool", "").strip()
            tool_code += "\n\n" + textwrap.indent(forward_source_code, "    ")

        else:  # If the tool was not created by the @tool decorator, it was made by subclassing Tool
            if type(self).__name__ in [
                "SpaceToolWrapper",
                "LangChainToolWrapper",
                "GradioToolWrapper",
            ]:
                raise ValueError(
                    "Cannot save objects created with from_space, from_langchain or from_gradio, as this would create errors."
                )

            validate_tool_attributes(self.__class__)

            tool_code = "from typing import Any, Optional\n" + instance_to_source(self, base_cls=Tool)

        requirements = {el for el in get_imports(tool_code) if el not in sys.stdlib_module_names} | {"smolleragents"}

        return {"name": self.name, "code": tool_code, "requirements": sorted(requirements)}

    def save(self, output_dir: str | Path, tool_file_name: str = "tool", make_gradio_app: bool = True):
        """
        Saves the relevant code files for your tool. This will copy the code of your tool in `output_dir` as well as autogenerate:

        - a `requirements.txt` containing the names of the modules used by your tool (as detected when inspecting its
          code)

        Args:
            output_dir (`str` or `Path`): The folder in which you want to save your tool.
            tool_file_name (`str`, *optional*, defaults to `"tool"`): The name of the file to save the tool code to.
            make_gradio_app (`bool`, *optional*, defaults to `True`): Whether to create a Gradio app file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save tool code
        tool_file_path = output_dir / f"{tool_file_name}.py"
        self._write_file(tool_file_path, self.to_dict()["code"])

        # Save requirements
        requirements_file_path = output_dir / "requirements.txt"
        self._write_file(requirements_file_path, "\n".join(self.to_dict()["requirements"]))

    def _write_file(self, file_path: Path, content: str) -> None:
        """Write content to a file."""
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

    @classmethod
    def from_code(cls, tool_code: str, **kwargs):
        module = types.ModuleType("dynamic_tool")

        exec(tool_code, module.__dict__)

        # Find the Tool subclass
        tool_class = next(
            (
                obj
                for _, obj in inspect.getmembers(module, inspect.isclass)
                if issubclass(obj, Tool) and obj is not Tool
            ),
            None,
        )

        if tool_class is None:
            raise ValueError("No Tool subclass found in the code.")

        if not isinstance(tool_class.inputs, dict):
            tool_class.inputs = ast.literal_eval(tool_class.inputs)

        return tool_class(**kwargs)

    @staticmethod
    def from_gradio(gradio_tool):
        """
        Creates a [`Tool`] from a Gradio tool.

        Args:
            gradio_tool: The Gradio tool to convert.
        Returns:
            [`Tool`]: The Gradio tool, as a tool.
        """
        class GradioToolWrapper(Tool):
            def __init__(self, _gradio_tool):
                self.gradio_tool = _gradio_tool
                super().__init__()

            def forward(self, *args, **kwargs):
                return self.gradio_tool(*args, **kwargs)

        return GradioToolWrapper(gradio_tool)

    @staticmethod
    def from_langchain(langchain_tool):
        """
        Creates a [`Tool`] from a LangChain tool.

        Args:
            langchain_tool: The LangChain tool to convert.
        Returns:
            [`Tool`]: The LangChain tool, as a tool.
        """
        class LangChainToolWrapper(Tool):
            skip_forward_signature_validation = True

            def __init__(self, _langchain_tool):
                self.langchain_tool = _langchain_tool
                super().__init__()

            def forward(self, *args, **kwargs):
                return self.langchain_tool.run(*args, **kwargs)

        return LangChainToolWrapper(langchain_tool)


def add_description(description):
    """
    Decorator to add a description to a function.

    Args:
        description (`str`): The description to add.
    """

    def inner(func):
        func.description = description
        return func

    return inner


class ToolCollection:
    """A collection of tools."""

    def __init__(self, tools: list[Tool]):
        self.tools = tools

    @classmethod
    def from_mcp(
        cls, server_parameters: dict, trust_remote_code: bool = False
    ) -> "ToolCollection":
        """
        Creates a [`ToolCollection`] from an MCP server.

        Args:
            server_parameters: The server parameters for the MCP server.
            trust_remote_code: Whether to trust remote code.
        Returns:
            [`ToolCollection`]: The tool collection from the MCP server.
        """
        if not trust_remote_code:
            raise ValueError(
                "Loading tools from MCP requires to acknowledge you trust its code: to do so, pass `trust_remote_code=True`."
            )

        # This would need to be implemented based on MCP integration
        # For now, return an empty collection
        return cls([])


def tool(tool_function: Callable) -> Tool:
    """
    Decorator to create a tool from a function.

    Args:
        tool_function (`Callable`): The function to convert to a tool.
    Returns:
        [`Tool`]: The tool created from the function.
    """

    class SimpleTool(Tool):
        skip_forward_signature_validation = True
        
        def __init__(self):
            # Get function signature
            sig = inspect.signature(tool_function)
            self.inputs = {}
            
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                    
                param_type = param.annotation if param.annotation != inspect.Parameter.empty else str
                
                # Convert Python types to string types
                if param_type == int:
                    param_type_str = "integer"
                elif param_type == float:
                    param_type_str = "number"
                elif param_type == bool:
                    param_type_str = "boolean"
                elif param_type == str:
                    param_type_str = "string"
                else:
                    param_type_str = "any"
                
                self.inputs[param_name] = {
                    "type": param_type_str,
                    "description": getattr(tool_function, 'description', f"Parameter {param_name}")
                }
            
            self.name = tool_function.__name__
            self.description = getattr(tool_function, 'description', tool_function.__doc__ or "A tool created from a function")
            self.output_type = "any"
            super().__init__()

        @wraps(tool_function)
        def forward(self, *args, **kwargs):
            return tool_function(*args, **kwargs)

    return SimpleTool()


def get_tools_definition_code(tools: dict[str, Tool]) -> str:
    """
    Get the code definition for a collection of tools.

    Args:
        tools (`dict[str, Tool]`): The tools to get definitions for.
    Returns:
        `str`: The code definition for the tools.
    """
    code_lines = []
    for tool_name, tool in tools.items():
        code_lines.append(f"def {tool_name}({', '.join(tool.inputs.keys())}):")
        code_lines.append(f'    """{tool.description}"""')
        code_lines.append("    pass")
        code_lines.append("")
    
    return "\n".join(code_lines)


__all__ = [
    "AUTHORIZED_TYPES",
    "Tool",
    "tool",
    "ToolCollection",
]