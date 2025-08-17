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
Agent Decision Guide
===================

Which agent should you use?

1. ToolCallingAgent (Recommended for most use cases)
   - Use when: You need quick answers to simple questions
   - Examples: "What is 2+2?", "Convert USD to EUR", "Get weather data"
   - Benefits: Fast, efficient, no planning overhead
   - Creation: create_tool_calling_agent(model, tools)

2. MultiStepAgent (For complex reasoning)
   - Use when: You need research, analysis, or multi-step problem solving
   - Examples: "Research AI trends", "Solve complex math step-by-step"
   - Benefits: Planning, thorough reasoning, state management
   - Creation: create_multi_step_agent(model, tools)

Quick Decision Tree:
- Simple calculation or API call? → ToolCallingAgent
- Need to research or analyze? → MultiStepAgent
- Unsure? Start with ToolCallingAgent, upgrade if needed

For detailed examples, see the README.md file.
"""

import importlib
import inspect
import json
import os
import re
import tempfile
import textwrap
import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import jinja2
import yaml
from jinja2 import StrictUndefined, Template


if TYPE_CHECKING:
    pass

from .agent_types import handle_agent_output_types
from .default_tools import TOOL_MAPPING, FinalAnswerTool
from .local_python_executor import BASE_BUILTIN_MODULES, LocalPythonExecutor, PythonExecutor, fix_final_answer_code
from .memory import (
    ActionStep,
    AgentMemory,
    FinalAnswerStep,
    Message,
    PlanningStep,
    SystemPromptStep,
    TaskStep,
    Timing,
    TokenUsage,
    ToolCall,
)
from .models import (
    CODEAGENT_RESPONSE_FORMAT,
    ChatMessage,
    ChatMessageStreamDelta,
    MessageRole,
    Model,
    parse_json_if_needed,
    OpenAIServerModel,
)
from .monitoring import (
    YELLOW_HEX,
    AgentLogger,
    LogLevel,
    Monitor,
)
from .remote_executors import DockerExecutor, E2BExecutor
from .tools import Tool
from .utils import (
    AgentError,
    AgentExecutionError,
    AgentGenerationError,
    AgentMaxStepsError,
    AgentParsingError,
    AgentToolCallError,
    AgentToolExecutionError,
    extract_code_from_text,
    is_valid_name,
    make_init_file,
    parse_code_blobs,
    truncate_content,
)


logger = getLogger(__name__)


def get_variable_names(self, template: str) -> set[str]:
    pattern = re.compile(r"\{\{([^{}]+)\}\}")
    return {match.group(1).strip() for match in pattern.finditer(template)}


def populate_template(template: str, variables: dict[str, Any]) -> str:
    compiled_template = Template(template, undefined=StrictUndefined)
    try:
        return compiled_template.render(**variables)
    except Exception as e:
        raise Exception(f"Error during jinja template rendering: {type(e).__name__}: {e}")


@dataclass
class FinalOutput:
    output: Any | None


class PlanningPromptTemplate(TypedDict):
    """
    Prompt templates for the planning step.

    Args:
        plan (`str`): Initial plan prompt.
        update_plan_pre_messages (`str`): Update plan pre-messages prompt.
        update_plan_post_messages (`str`): Update plan post-messages prompt.
    """

    initial_plan: str
    update_plan_pre_messages: str
    update_plan_post_messages: str


class ManagedAgentPromptTemplate(TypedDict):
    """
    Prompt templates for the managed agent.

    Args:
        task (`str`): Task prompt.
        report (`str`): Report prompt.
    """

    task: str
    report: str


class FinalAnswerPromptTemplate(TypedDict):
    """
    Prompt templates for the final answer.

    Args:
        pre_messages (`str`): Pre-messages prompt.
        post_messages (`str`): Post-messages prompt.
    """

    pre_messages: str
    post_messages: str


class PromptTemplates(TypedDict):
    """
    Prompt templates for the agent.

    Args:
        system_prompt (`str`): System prompt.
        planning ([`~agents.PlanningPromptTemplate`]): Planning prompt templates.
        managed_agent ([`~agents.ManagedAgentPromptTemplate`]): Managed agent prompt templates.
        final_answer ([`~agents.FinalAnswerPromptTemplate`]): Final answer prompt templates.
    """

    system_prompt: str
    planning: PlanningPromptTemplate
    managed_agent: ManagedAgentPromptTemplate
    final_answer: FinalAnswerPromptTemplate


EMPTY_PROMPT_TEMPLATES = PromptTemplates(
    system_prompt="",
    planning=PlanningPromptTemplate(
        initial_plan="",
        update_plan_pre_messages="",
        update_plan_post_messages="",
    ),
    managed_agent=ManagedAgentPromptTemplate(task="", report=""),
    final_answer=FinalAnswerPromptTemplate(pre_messages="", post_messages=""),
)


@dataclass
class RunResult:
    """Holds extended information about an agent run.

    Attributes:
        output (Any | None): The final output of the agent run, if available.
        state (Literal["success", "max_steps_error"]): The final state of the agent after the run.
        messages (list[dict]): The agent's memory, as a list of messages.
        token_usage (TokenUsage | None): Count of tokens used during the run.
        timing (Timing): Timing details of the agent run: start time, end time, duration.
    """

    output: Any | None
    state: Literal["success", "max_steps_error"]
    messages: list[dict]
    token_usage: TokenUsage | None
    timing: Timing


class MultiStepAgent(ABC):
    """
    Agent class that solves the given task step by step, using the ReAct framework:
    While the objective is not reached, the agent will perform a cycle of action (given by the LLM) and observation (obtained from the environment).

    Args:
        tools (`list[Tool]`): [`Tool`]s that the agent can use.
        model (`Callable[[list[dict[str, str]]], ChatMessage]`): Model that will generate the agent's actions.
        prompt_templates ([`~agents.PromptTemplates`], *optional*): Prompt templates.
        max_steps (`int`, default `20`): Maximum number of steps the agent can take to solve the task.
        add_base_tools (`bool`, default `False`): Whether to add the base tools to the agent's tools.
        verbosity_level (`LogLevel`, default `LogLevel.INFO`): Level of verbosity of the agent's logs.
        grammar (`dict[str, str]`, *optional*): Grammar used to parse the LLM output.
            <Deprecated version="1.17.0">
            Parameter `grammar` is deprecated and will be removed in version 1.20.
            </Deprecated>
        managed_agents (`list`, *optional*): Managed agents that the agent can call.
        step_callbacks (`list[Callable]`, *optional*): Callbacks that will be called at each step.
        planning_interval (`int`, *optional*): Interval at which the agent will run a planning step.
        name (`str`, *optional*): Necessary for a managed agent only - the name by which this agent can be called.
        description (`str`, *optional*): Necessary for a managed agent only - the description of this agent.
        provide_run_summary (`bool`, *optional*): Whether to provide a run summary when called as a managed agent.
        final_answer_checks (`list`, *optional*): List of Callables to run before returning a final answer for checking validity.
    """

    def __init__(
        self,
        tools: list[Tool],
        model: Model,
        prompt_templates: PromptTemplates | None = None,
        max_steps: int = 20,
        add_base_tools: bool = False,
        verbosity_level: LogLevel = LogLevel.INFO,
        grammar: dict[str, str] | None = None,
        managed_agents: list | None = None,
        step_callbacks: list[Callable] | None = None,
        planning_interval: int | None = None,
        name: str | None = None,
        description: str | None = None,
        provide_run_summary: bool = False,
        final_answer_checks: list[Callable] | None = None,
        return_full_result: bool = False,
        logger: AgentLogger | None = None,
    ):
        self.agent_name = self.__class__.__name__
        self.model = model
        self.prompt_templates = prompt_templates or EMPTY_PROMPT_TEMPLATES
        if prompt_templates is not None:
            missing_keys = set(EMPTY_PROMPT_TEMPLATES.keys()) - set(prompt_templates.keys())
            assert not missing_keys, (
                f"Some prompt templates are missing from your custom `prompt_templates`: {missing_keys}"
            )
            for key, value in EMPTY_PROMPT_TEMPLATES.items():
                if isinstance(value, dict):
                    for subkey in value.keys():
                        assert key in prompt_templates.keys() and (subkey in prompt_templates[key].keys()), (
                            f"Some prompt templates are missing from your custom `prompt_templates`: {subkey} under {key}"
                        )

        self.max_steps = max_steps
        self.step_number = 0
        if grammar is not None:
            warnings.warn(
                "Parameter 'grammar' is deprecated and will be removed in version 1.20.",
                FutureWarning,
            )
        self.grammar = grammar
        self.planning_interval = planning_interval
        self.state: dict[str, Any] = {}
        self.name = self._validate_name(name)
        self.description = description
        self.provide_run_summary = provide_run_summary
        self.final_answer_checks = final_answer_checks
        self.return_full_result = return_full_result

        self._setup_managed_agents(managed_agents)
        self._setup_tools(tools, add_base_tools)
        self._validate_tools_and_managed_agents(tools, managed_agents)

        self.system_prompt = self.initialize_system_prompt()
        self.task: str | None = None
        self.memory = AgentMemory(self.system_prompt)

        if logger is None:
            self.logger = AgentLogger(level=verbosity_level)
        else:
            self.logger = logger

        self.monitor = Monitor(level=verbosity_level)
        self.step_callbacks = step_callbacks if step_callbacks is not None else []
        self.stream_outputs = False

    def _validate_name(self, name: str | None) -> str | None:
        if name is not None and not is_valid_name(name):
            raise ValueError(f"Agent name '{name}' must be a valid Python identifier and not a reserved keyword.")
        return name

    def _setup_managed_agents(self, managed_agents: list | None = None) -> None:
        """Setup managed agents with proper logging."""
        self.managed_agents = {}
        if managed_agents:
            assert all(agent.name and agent.description for agent in managed_agents), (
                "All managed agents need both a name and a description!"
            )
            self.managed_agents = {agent.name: agent for agent in managed_agents}

    def _setup_tools(self, tools, add_base_tools):
        assert all(isinstance(tool, Tool) for tool in tools), "All elements must be instance of Tool (or a subclass)"
        self.tools = {tool.name: tool for tool in tools}
        if add_base_tools:
            self.tools.update(
                {
                    name: cls()
                    for name, cls in TOOL_MAPPING.items()
                    if name != "python_interpreter" or self.__class__.__name__ == "ToolCallingAgent"
                }
            )
        self.tools.setdefault("final_answer", FinalAnswerTool())

    def _validate_tools_and_managed_agents(self, tools, managed_agents):
        tool_and_managed_agent_names = [tool.name for tool in tools]
        if managed_agents is not None:
            tool_and_managed_agent_names += [agent.name for agent in managed_agents]
        if self.name:
            tool_and_managed_agent_names.append(self.name)
        if len(tool_and_managed_agent_names) != len(set(tool_and_managed_agent_names)):
            raise ValueError(
                "Each tool or managed_agent should have a unique name! You passed these duplicate names: "
                f"{[name for name in tool_and_managed_agent_names if tool_and_managed_agent_names.count(name) > 1]}"
            )

    def run(
        self,
        task: str,
        stream: bool = False,
        reset: bool = True,
        images: list[Any] | None = None,
        additional_args: dict | None = None,
        max_steps: int | None = None,
    ):
        """
        Run the agent for the given task.

        Args:
            task (`str`): Task to perform.
            stream (`bool`): Whether to run in streaming mode.
                If `True`, returns a generator that yields each step as it is executed. You must iterate over this generator to process the individual steps (e.g., using a for loop or `next()`).
                If `False`, executes all steps internally and returns only the final answer after completion.
            reset (`bool`): Whether to reset the conversation or keep it going from previous run.
            images (`list[PIL.Image.Image]`, *optional*): Image(s) objects.
            additional_args (`dict`, *optional*): Any other variables that you want to pass to the agent run, for instance images or dataframes. Give them clear names!
            max_steps (`int`, *optional*): Maximum number of steps the agent can take to solve the task. if not provided, will use the agent's default value.

        Example:
        ```py
        from smolleragents import CodeAgent
        agent = CodeAgent(tools=[])
        agent.run("What is the result of 2 power 3.7384?")
        ```
        """
        max_steps = max_steps or self.max_steps
        self.task = task
        self.interrupt_switch = False
        if additional_args is not None:
            self.state.update(additional_args)
            self.task += f"""
You have been provided with these additional arguments, that you can access using the keys as variables in your python code:
{str(additional_args)}."""

        self.system_prompt = self.initialize_system_prompt()
        self.memory.system_prompt = SystemPromptStep(system_prompt=self.system_prompt)
        if reset:
            self.memory.reset()
            self.monitor.reset()

        self.logger.log_task(
            content=self.task.strip(),
            subtitle=f"{type(self.model).__name__} - {(self.model.model_id if hasattr(self.model, 'model_id') else '')}",
            level=LogLevel.INFO,
            title=self.name if hasattr(self, "name") else None,
        )
        self.memory.steps.append(TaskStep(task=self.task, task_images=images))

        if getattr(self, "python_executor", None):
            self.python_executor.send_variables(variables=self.state)
            self.python_executor.send_tools({**self.tools, **self.managed_agents})

        if stream:
            # The steps are returned as they are executed through a generator to iterate on.
            return self._run_stream(task=self.task, max_steps=max_steps, images=images)
        run_start_time = time.time()
        # Outputs are returned only at the end. We only look at the last step.

        steps = list(self._run_stream(task=self.task, max_steps=max_steps, images=images))
        assert isinstance(steps[-1], FinalAnswerStep)
        output = steps[-1].output

        if self.return_full_result:
            total_input_tokens = 0
            total_output_tokens = 0
            correct_token_usage = True
            for step in self.memory.steps:
                if isinstance(step, (ActionStep, PlanningStep)):
                    if step.token_usage is None:
                        correct_token_usage = False
                        break
                    else:
                        total_input_tokens += step.token_usage.input_tokens
                        total_output_tokens += step.token_usage.output_tokens
            if correct_token_usage:
                token_usage = TokenUsage(input_tokens=total_input_tokens, output_tokens=total_output_tokens)
            else:
                token_usage = None

            if self.memory.steps and isinstance(getattr(self.memory.steps[-1], "error", None), AgentMaxStepsError):
                state = "max_steps_error"
            else:
                state = "success"

            messages = self.memory.get_full_steps()

            return RunResult(
                output=output,
                token_usage=token_usage,
                messages=messages,
                timing=Timing(start_time=run_start_time, end_time=time.time()),
                state=state,
            )

        return output

    def _run_stream(
        self, task: str, max_steps: int, images: list[Any] | None = None
    ) -> Generator[ActionStep | PlanningStep | FinalAnswerStep | ChatMessageStreamDelta]:
        final_answer = None
        self.step_number = 1
        while final_answer is None and self.step_number <= max_steps:
            if self.interrupt_switch:
                raise AgentError("Agent interrupted.", self.logger)

            # Run a planning step if scheduled
            if self.planning_interval is not None and (
                self.step_number == 1 or (self.step_number - 1) % self.planning_interval == 0
            ):
                planning_start_time = time.time()
                planning_step = None
                for element in self._generate_planning_step(
                    task, is_first_step=(self.step_number == 1), step=self.step_number
                ):
                    yield element
                    planning_step = element
                assert isinstance(planning_step, PlanningStep)  # Last yielded element should be a PlanningStep
                self.memory.steps.append(planning_step)
                planning_end_time = time.time()
                planning_step.timing = Timing(
                    start_time=planning_start_time,
                    end_time=planning_end_time,
                )

            # Start action step!
            action_step_start_time = time.time()
            action_step = ActionStep(
                step_number=self.step_number,
                timing=Timing(start_time=action_step_start_time),
                observations_images=images,
            )
            try:
                for el in self._execute_step(action_step):
                    yield el
                final_answer = el
            except AgentGenerationError as e:
                # Agent generation errors are not caused by a Model error but an implementation error: so we should raise them and exit.
                raise e
            except AgentError as e:
                # Other AgentError types are caused by the Model, so we should log them and iterate.
                action_step.error = e
            finally:
                self._finalize_step(action_step)
                self.memory.steps.append(action_step)
                yield action_step
                self.step_number += 1

        if final_answer is None and self.step_number == max_steps + 1:
            final_answer = self._handle_max_steps_reached(task)
            yield action_step
        yield FinalAnswerStep(handle_agent_output_types(final_answer))

    def _execute_step(self, memory_step: ActionStep) -> Generator[ChatMessageStreamDelta | FinalOutput]:
        self.logger.log_rule(f"Step {self.step_number}", level=LogLevel.INFO)
        for el in self._step_stream(memory_step):
            final_answer = el
            if isinstance(el, ChatMessageStreamDelta):
                yield el
            elif isinstance(el, FinalOutput):
                final_answer = el.output
                if self.final_answer_checks:
                    self._validate_final_answer(final_answer)
                yield final_answer

    def _validate_final_answer(self, final_answer: Any):
        for check_function in self.final_answer_checks:
            try:
                assert check_function(final_answer, self.memory)
            except Exception as e:
                raise AgentError(f"Check {check_function.__name__} failed with error: {e}")

    def _finalize_step(self, memory_step: ActionStep):
        memory_step.timing.end_time = time.time()
        for callback in self.step_callbacks:
            # For compatibility with old callbacks that don't take the agent as an argument
            callback(memory_step) if len(inspect.signature(callback).parameters) == 1 else callback(
                memory_step, agent=self
            )

    def _handle_max_steps_reached(self, task: str) -> Any:
        action_step_start_time = time.time()
        final_answer = self.provide_final_answer(task)
        final_memory_step = ActionStep(
            step_number=self.step_number,
            error=AgentMaxStepsError("Reached max steps."),
            timing=Timing(start_time=action_step_start_time, end_time=time.time()),
            token_usage=None,
        )
        final_memory_step.action_output = final_answer.content
        self._finalize_step(final_memory_step)
        self.memory.steps.append(final_memory_step)
        return final_answer.content

    def _generate_planning_step(
        self, task, is_first_step: bool, step: int
    ) -> Generator[ChatMessageStreamDelta | PlanningStep]:
        start_time = time.time()
        if is_first_step:
            input_messages = [
                {
                    "role": MessageRole.USER,
                    "content": [
                        {
                            "type": "text",
                            "text": populate_template(
                                self.prompt_templates["planning"]["initial_plan"],
                                variables={"task": task, "tools": self.tools, "managed_agents": self.managed_agents},
                            ),
                        }
                    ],
                }
            ]
            if self.stream_outputs and hasattr(self.model, "generate_stream"):
                plan_message_content = ""
                output_stream = self.model.generate_stream(input_messages, stop_sequences=["<end_plan>"])  # type: ignore
                input_tokens, output_tokens = 0, 0
                for event in output_stream:
                    if event.content is not None:
                        plan_message_content += event.content
                        if self.logger.level.value <= LogLevel.INFO.value:
                            print(f"Planning: {event.content}")
                        if event.token_usage:
                            output_tokens += event.token_usage.output_tokens
                            input_tokens = event.token_usage.input_tokens
                    yield event
            else:
                plan_message = self.model.generate(input_messages, stop_sequences=["<end_plan>"])
                plan_message_content = plan_message.content
                input_tokens, output_tokens = (
                    plan_message.token_usage.input_tokens,
                    plan_message.token_usage.output_tokens,
                )
            plan = textwrap.dedent(
                f"""Here are the facts I know and the plan of action that I will follow to solve the task:\n```\n{plan_message_content}\n```"""
            )
        else:
            # Summary mode removes the system prompt and previous planning messages output by the model.
            # Removing previous planning messages avoids influencing too much the new plan.
            memory_messages = self.write_memory_to_messages(summary_mode=True)
            plan_update_pre = {
                "role": MessageRole.SYSTEM,
                "content": [
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["planning"]["update_plan_pre_messages"], variables={"task": task}
                        ),
                    }
                ],
            }
            plan_update_post = {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["planning"]["update_plan_post_messages"],
                            variables={
                                "task": task,
                                "tools": self.tools,
                                "managed_agents": self.managed_agents,
                                "remaining_steps": (self.max_steps - step),
                            },
                        ),
                    }
                ],
            }
            input_messages = [plan_update_pre] + memory_messages + [plan_update_post]
            if self.stream_outputs and hasattr(self.model, "generate_stream"):
                plan_message_content = ""
                input_tokens, output_tokens = 0, 0
                for event in self.model.generate_stream(
                    input_messages,
                    stop_sequences=["<end_plan>"],
                ):  # type: ignore
                    if event.content is not None:
                        plan_message_content += event.content
                        if self.logger.level.value <= LogLevel.INFO.value:
                            print(f"Planning Update: {event.content}")
                        if event.token_usage:
                            output_tokens += event.token_usage.output_tokens
                            input_tokens = event.token_usage.input_tokens
                    yield event
            else:
                plan_message = self.model.generate(input_messages, stop_sequences=["<end_plan>"])
                plan_message_content = plan_message.content
                input_tokens, output_tokens = (
                    plan_message.token_usage.input_tokens,
                    plan_message.token_usage.output_tokens,
                )
            plan = textwrap.dedent(
                f"""I still need to solve the task I was given:\n```\n{self.task}\n```\n\nHere are the facts I know and my new/updated plan of action to solve the task:\n```\n{plan_message_content}\n```"""
            )
        log_headline = "Initial plan" if is_first_step else "Updated plan"
        if self.logger.level.value <= LogLevel.INFO.value:
            print(f"\n{log_headline}:")
            print(plan)
        yield PlanningStep(
            model_input_messages=input_messages,
            plan=plan,
            model_output_message=ChatMessage(role=MessageRole.ASSISTANT, content=plan_message_content),
            token_usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
            timing=Timing(start_time=start_time, end_time=time.time()),
        )

    @property
    def logs(self):
        logger.warning(
            "The 'logs' attribute is deprecated and will soon be removed. Please use 'self.memory.steps' instead."
        )
        return [self.memory.system_prompt] + self.memory.steps

    @abstractmethod
    def initialize_system_prompt(self) -> str:
        """To be implemented in child classes"""
        ...

    def interrupt(self):
        """Interrupts the agent execution."""
        self.interrupt_switch = True

    def write_memory_to_messages(
        self,
        summary_mode: bool | None = False,
    ) -> list[Message]:
        """
        Reads past llm_outputs, actions, and observations or errors from the memory into a series of messages
        that can be used as input to the LLM. Adds a number of keywords (such as PLAN, error, etc) to help
        the LLM.
        """
        messages = self.memory.system_prompt.to_messages(summary_mode=summary_mode)
        for memory_step in self.memory.steps:
            messages.extend(memory_step.to_messages(summary_mode=summary_mode))
        return messages

    def _step_stream(self, memory_step: ActionStep) -> Generator[FinalOutput]:
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Yields either None if the step is not final, or the final answer.
        """
        memory_messages = self.write_memory_to_messages()

        input_messages = memory_messages.copy()

        # Add new step in logs
        memory_step.model_input_messages = input_messages

        try:
            chat_message: ChatMessage = self.model.generate(
                input_messages,
                stop_sequences=["Observation:", "Calling tools:"],
                tools_to_call_from=list(self.tools.values()),
            )
            memory_step.model_output_message = chat_message
            model_output = chat_message.content
            self.logger.log(
                message=model_output if model_output else str(chat_message.raw),
                level=LogLevel.DEBUG,
            )

            memory_step.model_output_message.content = model_output
            memory_step.model_output = model_output
        except Exception as e:
            raise AgentGenerationError(f"Error while generating output:\n{e}") from e

        if chat_message.tool_calls is None or len(chat_message.tool_calls) == 0:
            try:
                chat_message = self.model.parse_tool_calls(chat_message)
            except Exception as e:
                raise AgentParsingError(f"Error while parsing tool call from model output: {e}")
        else:
            for tool_call in chat_message.tool_calls:
                tool_call.function.arguments = parse_json_if_needed(tool_call.function.arguments)
        tool_call = chat_message.tool_calls[0]  # type: ignore
        tool_name, tool_call_id = tool_call.function.name, tool_call.id
        tool_arguments = tool_call.function.arguments
        memory_step.model_output = str(f"Called Tool: '{tool_name}' with arguments: {tool_arguments}")
        memory_step.tool_calls = [ToolCall(name=tool_name, arguments=tool_arguments, id=tool_call_id)]
        memory_step.token_usage = chat_message.token_usage

        # Execute
        if self.logger.level.value <= LogLevel.INFO.value:
            print(f"Calling tool: '{tool_name}' with arguments: {tool_arguments}")
        if tool_name == "final_answer":
            if isinstance(tool_arguments, dict):
                if "answer" in tool_arguments:
                    answer = tool_arguments["answer"]
                else:
                    answer = tool_arguments
            else:
                answer = tool_arguments
            final_answer = self.execute_tool_call("final_answer", {"answer": answer})
            if self.logger.level.value <= LogLevel.INFO.value:
                print(f"Final answer: {final_answer}")

            memory_step.action_output = final_answer
            yield FinalOutput(output=final_answer)
        else:
            if tool_arguments is None:
                tool_arguments = {}
            observation = self.execute_tool_call(tool_name, tool_arguments)
            updated_information = str(observation).strip()
            if self.logger.level.value <= LogLevel.INFO.value:
                print(f"Observations: {updated_information.replace('[', '|')}")  # escape potential rich-tag-like components
            memory_step.observations = updated_information
            yield FinalOutput(output=None)

    def execute_tool_call(self, tool_name: str, arguments: dict[str, str] | str) -> Any:
        """
        Execute a tool or managed agent with the provided arguments.

        The arguments are replaced with the actual values from the state if they refer to state variables.

        Args:
            tool_name (`str`): Name of the tool or managed agent to execute.
            arguments (dict[str, str] | str): Arguments passed to the tool call.
        """
        # Check if the tool exists
        available_tools = {**self.tools, **self.managed_agents}
        if tool_name not in available_tools:
            raise AgentToolExecutionError(
                f"Unknown tool {tool_name}, should be one of: {', '.join(available_tools)}."
            )

        # Get the tool and substitute state variables in arguments
        tool = available_tools[tool_name]
        arguments = self._substitute_state_variables(arguments)
        is_managed_agent = tool_name in self.managed_agents

        try:
            # Call tool with appropriate arguments
            if isinstance(arguments, dict):
                return tool(**arguments) if is_managed_agent else tool(**arguments, sanitize_inputs_outputs=True)
            elif isinstance(arguments, str):
                return tool(arguments) if is_managed_agent else tool(arguments, sanitize_inputs_outputs=True)
            else:
                raise TypeError(f"Unsupported arguments type: {type(arguments)}")

        except TypeError as e:
            # Handle invalid arguments
            description = getattr(tool, "description", "No description")
            if is_managed_agent:
                error_msg = (
                    f"Invalid request to team member '{tool_name}' with arguments {json.dumps(arguments)}: {e}\n"
                    "You should call this team member with a valid request.\n"
                    f"Team member description: {description}"
                )
            else:
                error_msg = (
                    f"Invalid call to tool '{tool_name}' with arguments {json.dumps(arguments)}: {e}\n"
                    "You should call this tool with correct input arguments.\n"
                    f"Expected inputs: {json.dumps(tool.inputs)}\n"
                    f"Returns output type: {tool.output_type}\n"
                    f"Tool description: '{description}'"
                )
            raise AgentToolCallError(error_msg) from e

        except Exception as e:
            # Handle execution errors
            if is_managed_agent:
                error_msg = (
                    f"Error executing request to team member '{tool_name}' with arguments {json.dumps(arguments)}: {e}\n"
                    "Please try again or request to another team member"
                )
            else:
                error_msg = (
                    f"Error executing tool '{tool_name}' with arguments {json.dumps(arguments)}: {type(e).__name__}: {e}\n"
                    "Please try again or use another tool"
                )
            raise AgentToolExecutionError(error_msg) from e

    def _substitute_state_variables(self, arguments: dict[str, str] | str) -> dict[str, Any] | str:
        """Replace string values in arguments with their corresponding state values if they exist."""
        # Since we're not using state management, just return arguments as-is
        return arguments

    def step(self, memory_step: ActionStep) -> Any:
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Returns either None if the step is not final, or the final answer.
        """
        return list(self._step_stream(memory_step))[-1]

    def extract_action(self, model_output: str, split_token: str) -> tuple[str, str]:
        """
        Parse action from the LLM output

        Args:
            model_output (`str`): Output of the LLM
            split_token (`str`): Separator for the action. Should match the example in the system prompt.
        """
        try:
            split = model_output.split(split_token)
            rationale, action = (
                split[-2],
                split[-1],
            )  # NOTE: using indexes starting from the end solves for when you have more than one split_token in the output
        except Exception:
            raise AgentParsingError(
                f"No '{split_token}' token provided in your output.\nYour output:\n{model_output}\n. Be sure to include an action, prefaced with '{split_token}'!",
                self.logger,
            )
        return rationale.strip(), action.strip()

    def provide_final_answer(self, task: str, images: list[Any] | None = None) -> ChatMessage:
        """
        Provide the final answer to the task, based on the logs of the agent's interactions.

        Args:
            task (`str`): Task to perform.
            images (`list[PIL.Image.Image]`, *optional*): Image(s) objects.

        Returns:
            `str`: Final answer to the task.
        """
        messages = [
            {
                "role": MessageRole.SYSTEM,
                "content": [
                    {
                        "type": "text",
                        "text": self.prompt_templates["final_answer"]["pre_messages"],
                    }
                ],
            }
        ]
        if images:
            messages[0]["content"].append({"type": "image"})
        messages += self.write_memory_to_messages()[1:]
        messages += [
            {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["final_answer"]["post_messages"], variables={"task": task}
                        ),
                    }
                ],
            }
        ]
        try:
            chat_message: ChatMessage = self.model.generate(messages)
            return chat_message
        except Exception as e:
            return f"Error in generating final LLM output:\n{e}"

    def visualize(self):
        """Prints a pretty visualization of the agent's steps."""
        print("=" * 80)
        print(f"Agent: {self.name or self.__class__.__name__}")
        print("=" * 80)
        
        for i, step in enumerate(self.memory.steps):
            print(f"\nStep {i + 1}:")
            print("-" * 40)
            
            if hasattr(step, 'model_output') and step.model_output:
                print("Thought:")
                print(step.model_output)
                print()
            
            if hasattr(step, 'tool_calls') and step.tool_calls:
                for tool_call in step.tool_calls:
                    print(f"Tool Call: {tool_call.name}")
                    print(f"Arguments: {tool_call.arguments}")
                    print()
            
            if hasattr(step, 'observations') and step.observations:
                print("Observation:")
                print(step.observations)
                print()
            
            if hasattr(step, 'error') and step.error:
                print(f"Error: {step.error}")
                print()
            
            if hasattr(step, 'timing') and step.timing:
                print(f"Duration: {step.timing.duration:.2f}s")
                if step.token_usage:
                    print(f"Tokens: {step.token_usage.input_tokens} in, {step.token_usage.output_tokens} out")
                print()

    def replay(self, detailed: bool = False):
        """Prints a pretty replay of the agent's steps.

        Args:
            detailed (bool, optional): If True, also displays the memory at each step. Defaults to False.
                Careful: will increase log length exponentially. Use only for debugging.
        """
        self.memory.replay(self.logger, detailed=detailed)

    def __call__(self, task: str, **kwargs):
        """Adds additional prompting for the managed agent, runs it, and wraps the output.
        This method is called only by a managed agent.
        """
        full_task = populate_template(
            self.prompt_templates["managed_agent"]["task"],
            variables=dict(name=self.name, task=task),
        )
        result = self.run(full_task, **kwargs)
        if isinstance(result, RunResult):
            report = result.output
        else:
            report = result
        answer = populate_template(
            self.prompt_templates["managed_agent"]["report"], variables=dict(name=self.name, final_answer=report)
        )
        if self.provide_run_summary:
            answer += "\n\nFor more detail, find below a summary of this agent's work:\n<summary_of_work>\n"
            for message in self.write_memory_to_messages(summary_mode=True):
                content = message["content"]
                answer += "\n" + truncate_content(str(content)) + "\n---"
            answer += "\n</summary_of_work>"
        return answer

    def save(self, output_dir: str | Path, relative_path: str | None = None):
        """
        Saves the relevant code files for your agent. This will copy the code of your agent in `output_dir` as well as autogenerate:

        - a `tools` folder containing the logic for each of the tools under `tools/{tool_name}.py`.
        - a `managed_agents` folder containing the logic for each of the managed agents.
        - an `agent.json` file containing a dictionary representing your agent.
        - a `prompt.yaml` file containing the prompt templates used by your agent.
        - a `requirements.txt` containing the names of the modules used by your tool (as detected when inspecting its
          code)

        Args:
            output_dir (`str` or `Path`): The folder in which you want to save your agent.
        """
        make_init_file(output_dir)

        # Recursively save managed agents
        if self.managed_agents:
            make_init_file(os.path.join(output_dir, "managed_agents"))
            for agent_name, agent in self.managed_agents.items():
                agent_suffix = f"managed_agents.{agent_name}"
                if relative_path:
                    agent_suffix = relative_path + "." + agent_suffix
                agent.save(os.path.join(output_dir, "managed_agents", agent_name), relative_path=agent_suffix)

        class_name = self.__class__.__name__

        # Save tools to different .py files
        for tool in self.tools.values():
            make_init_file(os.path.join(output_dir, "tools"))
            tool.save(os.path.join(output_dir, "tools"), tool_file_name=tool.name, make_gradio_app=False)

        # Save prompts to yaml
        yaml_prompts = yaml.safe_dump(
            self.prompt_templates,
            default_style="|",  # This forces block literals for all strings
            default_flow_style=False,
            width=float("inf"),
            sort_keys=False,
            allow_unicode=True,
            indent=2,
        )

        with open(os.path.join(output_dir, "prompts.yaml"), "w", encoding="utf-8") as f:
            f.write(yaml_prompts)

        # Save agent dictionary to json
        agent_dict = self.to_dict()
        agent_dict["tools"] = [tool.name for tool in self.tools.values()]
        agent_dict["managed_agents"] = {agent.name: agent.__class__.__name__ for agent in self.managed_agents.values()}
        with open(os.path.join(output_dir, "agent.json"), "w", encoding="utf-8") as f:
            json.dump(agent_dict, f, indent=4)

        # Save requirements
        with open(os.path.join(output_dir, "requirements.txt"), "w", encoding="utf-8") as f:
            f.writelines(f"{r}\n" for r in agent_dict["requirements"])

    def to_dict(self) -> dict[str, Any]:
        """Convert the agent to a dictionary representation.

        Returns:
            `dict`: Dictionary representation of the agent.
        """
        # TODO: handle serializing step_callbacks and final_answer_checks
        for attr in ["final_answer_checks", "step_callbacks"]:
            if getattr(self, attr, None):
                self.logger.log(f"This agent has {attr}: they will be ignored by this method.", LogLevel.INFO)

        tool_dicts = [tool.to_dict() for tool in self.tools.values()]
        tool_requirements = {req for tool in self.tools.values() for req in tool.to_dict()["requirements"]}
        managed_agents_requirements = {
            req for managed_agent in self.managed_agents.values() for req in managed_agent.to_dict()["requirements"]
        }
        requirements = tool_requirements | managed_agents_requirements
        if hasattr(self, "authorized_imports"):
            requirements.update(
                {package.split(".")[0] for package in self.authorized_imports if package not in BASE_BUILTIN_MODULES}
            )

        agent_dict = {
            "class": self.__class__.__name__,
            "tools": tool_dicts,
            "model": {
                "class": self.model.__class__.__name__,
                "data": self.model.to_dict(),
            },
            "managed_agents": [managed_agent.to_dict() for managed_agent in self.managed_agents.values()],
            "prompt_templates": self.prompt_templates,
            "max_steps": self.max_steps,
            "verbosity_level": int(self.logger.level),
            "grammar": self.grammar,
            "planning_interval": self.planning_interval,
            "name": self.name,
            "description": self.description,
            "requirements": sorted(requirements),
        }
        return agent_dict

    @classmethod
    def from_dict(cls, agent_dict: dict[str, Any], **kwargs) -> "MultiStepAgent":
        """Create agent from a dictionary representation.

        Args:
            agent_dict (`dict[str, Any]`): Dictionary representation of the agent.
            **kwargs: Additional keyword arguments that will override agent_dict values.

        Returns:
            `MultiStepAgent`: The created agent.
        """
        # Import here to avoid circular imports
        from .models import Model

        # Override with kwargs
        agent_dict = {**agent_dict, **kwargs}

        # Create model
        model_class_name = agent_dict["model"]["class"]
        model_data = agent_dict["model"]["data"]
        
        # Import model class dynamically
        if model_class_name == "OpenAIServerModel":
            from .models import OpenAIServerModel
            model = OpenAIServerModel(**model_data)
        elif model_class_name == "ApiModel":
            from .models import ApiModel
            model = ApiModel(**model_data)
        else:
            raise ValueError(f"Unsupported model class: {model_class_name}")

        # Create tools
        tools = []
        for tool_dict in agent_dict["tools"]:
            tool_class_name = tool_dict["class"]
            if tool_class_name == "PythonInterpreterTool":
                from .default_tools import PythonInterpreterTool
                tool = PythonInterpreterTool(**tool_dict["data"])
            elif tool_class_name == "FinalAnswerTool":
                from .default_tools import FinalAnswerTool
                tool = FinalAnswerTool()
            else:
                # For custom tools, we'll need to import them dynamically
                # This is a simplified version - in practice you might need more sophisticated tool loading
                raise ValueError(f"Unsupported tool class: {tool_class_name}")
            tools.append(tool)

        # Create managed agents
        managed_agents = []
        for managed_agent_dict in agent_dict.get("managed_agents", []):
            # This would need to be implemented based on your managed agent structure
            pass

        # Create agent
        agent = cls(
            tools=tools,
            model=model,
            prompt_templates=agent_dict["prompt_templates"],
            max_steps=agent_dict["max_steps"],
            verbosity_level=agent_dict["verbosity_level"],
            grammar=agent_dict.get("grammar"),
            planning_interval=agent_dict.get("planning_interval"),
            name=agent_dict.get("name"),
            description=agent_dict.get("description"),
        )

        return agent

    @classmethod
    def from_folder(cls, folder: str | Path, **kwargs):
        """Create agent from a folder containing agent files.

        Args:
            folder (`str` or `Path`): Path to the folder containing agent files.
            **kwargs: Additional keyword arguments.

        Returns:
            `MultiStepAgent`: The created agent.
        """
        folder = Path(folder)
        
        # Load agent.json
        with open(folder / "agent.json", "r") as f:
            agent_dict = json.load(f)
        
        # Load prompt templates
        with open(folder / "prompts.yaml", "r") as f:
            agent_dict["prompt_templates"] = yaml.safe_load(f)
        
        return cls.from_dict(agent_dict, **kwargs)


class ToolCallingAgent(MultiStepAgent):
    """
    A concrete implementation of MultiStepAgent for tool calling.
    
    This agent is designed for simple, single-step tool execution.
    It uses JSON tool calls and doesn't require planning.
    
    Key Features:
    - Single-step execution (one tool call per step)
    - JSON-based tool calling format
    - No planning by default (planning_interval=None)
    - Automatic loading of toolcalling_agent.yaml prompts
    
    Use Cases:
    - Simple calculations (e.g., "What is 2+2?")
    - Single tool operations (e.g., "Update this record")
    - Direct tool execution without complex reasoning
    
    Example:
        agent = ToolCallingAgent(
            model=model,
            tools=[PythonInterpreterTool(), FinalAnswerTool()]
        )
        result = agent.run("Calculate 2 + 2")
    """
    
    def __init__(self, *args, **kwargs):
        # Auto-load default prompts if none provided
        if 'prompt_templates' not in kwargs:
            try:
                kwargs['prompt_templates'] = load_default_prompts("ToolCallingAgent")
            except Exception as e:
                # Fall back to empty templates if loading fails
                pass
        
        # Ensure planning is disabled for tool calling
        kwargs.setdefault('planning_interval', None)
        
        super().__init__(*args, **kwargs)
    
    def initialize_system_prompt(self) -> str:
        """Initialize system prompt from loaded templates."""
        if self.prompt_templates and self.prompt_templates.get('system_prompt'):
            return self.prompt_templates['system_prompt']
        return "You are a helpful AI assistant that can use tools to solve tasks."

    def _step_stream(self, memory_step: ActionStep) -> Generator[FinalOutput]:
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        """
        memory_messages = self.write_memory_to_messages()

        input_messages = memory_messages.copy()

        # Add new step in logs
        memory_step.model_input_messages = input_messages

        try:
            chat_message: ChatMessage = self.model.generate(
                input_messages,
                stop_sequences=["Observation:", "Calling tools:"],
                tools_to_call_from=list(self.tools.values()),
            )
            memory_step.model_output_message = chat_message
            model_output = chat_message.content
            self.logger.log(
                message=model_output if model_output else str(chat_message.raw),
                level=LogLevel.DEBUG,
            )

            memory_step.model_output_message.content = model_output
            memory_step.model_output = model_output
        except Exception as e:
            raise AgentGenerationError(f"Error while generating output:\n{e}") from e

        if chat_message.tool_calls is None or len(chat_message.tool_calls) == 0:
            try:
                chat_message = self.model.parse_tool_calls(chat_message)
            except Exception as e:
                raise AgentParsingError(f"Error while parsing tool call from model output: {e}")
        else:
            for tool_call in chat_message.tool_calls:
                tool_call.function.arguments = parse_json_if_needed(tool_call.function.arguments)
        tool_call = chat_message.tool_calls[0]  # type: ignore
        tool_name, tool_call_id = tool_call.function.name, tool_call.id
        tool_arguments = tool_call.function.arguments
        memory_step.model_output = str(f"Called Tool: '{tool_name}' with arguments: {tool_arguments}")
        memory_step.tool_calls = [ToolCall(name=tool_name, arguments=tool_arguments, id=tool_call_id)]
        memory_step.token_usage = chat_message.token_usage

        # Execute
        if self.logger.level.value <= LogLevel.INFO.value:
            print(f"Calling tool: '{tool_name}' with arguments: {tool_arguments}")
        if tool_name == "final_answer":
            if isinstance(tool_arguments, dict):
                if "answer" in tool_arguments:
                    answer = tool_arguments["answer"]
                else:
                    answer = tool_arguments
            else:
                answer = tool_arguments
            final_answer = self.execute_tool_call("final_answer", {"answer": answer})
            if self.logger.level.value <= LogLevel.INFO.value:
                print(f"Final answer: {final_answer}")

            memory_step.action_output = final_answer
            yield FinalOutput(output=final_answer)
        else:
            if tool_arguments is None:
                tool_arguments = {}
            observation = self.execute_tool_call(tool_name, tool_arguments)
            updated_information = str(observation).strip()
            if self.logger.level.value <= LogLevel.INFO.value:
                print(f"Observations: {updated_information.replace('[', '|')}")  # escape potential rich-tag-like components
            memory_step.observations = updated_information
            yield FinalOutput(output=None)


def load_default_prompts(agent_type: str) -> PromptTemplates:
    """
    Load default prompt templates based on agent type.
    
    Args:
        agent_type: "ToolCallingAgent" or "MultiStepAgent"
    
    Returns:
        PromptTemplates: Loaded from appropriate YAML file
    """
    # Get the directory where this file is located
    current_dir = Path(__file__).parent
    prompts_dir = current_dir / "prompts"
    
    if agent_type == "ToolCallingAgent":
        yaml_file = prompts_dir / "toolcalling_agent.yaml"
    elif agent_type == "MultiStepAgent":
        # Use structured_code_agent.yaml for MultiStepAgent for better structured output
        yaml_file = prompts_dir / "structured_code_agent.yaml"
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    if not yaml_file.exists():
        raise FileNotFoundError(f"Prompt template file not found: {yaml_file}")
    
    with open(yaml_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_tool_calling_agent(model: Model, tools: list[Tool], **kwargs) -> "ToolCallingAgent":
    """
    Create ToolCallingAgent with default prompts loaded automatically.
    
    This agent is designed for simple, single-step tool execution.
    It uses JSON tool calls and doesn't require planning.
    
    Args:
        model: The model to use for generation
        tools: List of tools the agent can use
        **kwargs: Additional arguments passed to ToolCallingAgent
    
    Returns:
        ToolCallingAgent: Configured with default tool calling prompts
    """
    # Load default prompts for tool calling
    prompt_templates = load_default_prompts("ToolCallingAgent")
    
    # Set planning_interval to None for tool calling (no planning needed)
    kwargs.setdefault('planning_interval', None)
    
    return ToolCallingAgent(
        model=model,
        tools=tools,
        prompt_templates=prompt_templates,
        **kwargs
    )


def create_multi_step_agent(model: Model, tools: list[Tool], **kwargs) -> "MultiStepAgent":
    """
    Create MultiStepAgent with structured code prompts and planning enabled.
    
    This agent is designed for complex multi-step reasoning tasks.
    It uses structured JSON output and includes planning steps.
    
    Args:
        model: The model to use for generation
        tools: List of tools the agent can use
        **kwargs: Additional arguments passed to MultiStepAgent
    
    Returns:
        MultiStepAgent: Configured with default structured code prompts and planning
    """
    # Load default prompts for multi-step reasoning
    prompt_templates = load_default_prompts("MultiStepAgent")
    
    # Enable planning every step for complex reasoning
    # TODO: Can be modified to planning_interval=N for different planning frequency
    kwargs.setdefault('planning_interval', 1)
    
    return MultiStepAgent(
        model=model,
        tools=tools,
        prompt_templates=prompt_templates,
        **kwargs
    )
