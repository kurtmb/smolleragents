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
from dataclasses import asdict, dataclass
from logging import getLogger
from typing import TYPE_CHECKING, Any, TypedDict

from .models import ChatMessage, MessageRole
from .monitoring import AgentLogger, LogLevel, Timing, TokenUsage
from .utils import AgentError, make_json_serializable


if TYPE_CHECKING:
    from .models import ChatMessage
    from .monitoring import AgentLogger


logger = getLogger(__name__)


class Message(TypedDict):
    role: MessageRole
    content: str | list[dict[str, Any]]


@dataclass
class ToolCall:
    name: str
    arguments: Any
    id: str

    def dict(self):
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": make_json_serializable(self.arguments),
            },
        }


@dataclass
class MemoryStep:
    def dict(self):
        return asdict(self)

    def to_messages(self, summary_mode: bool = False) -> list[Message]:
        raise NotImplementedError


@dataclass
class ActionStep(MemoryStep):
    step_number: int
    timing: Timing
    model_input_messages: list[Message] | None = None
    tool_calls: list[ToolCall] | None = None
    error: AgentError | None = None
    model_output_message: ChatMessage | None = None
    model_output: str | None = None
    observations: str | None = None
    observations_images: list[Any] | None = None
    action_output: Any = None
    token_usage: TokenUsage | None = None

    def dict(self):
        # We overwrite the method to parse the tool_calls and action_output manually
        return {
            "model_input_messages": self.model_input_messages,
            "tool_calls": [tc.dict() for tc in self.tool_calls] if self.tool_calls else [],
            "timing": self.timing.dict(),
            "token_usage": asdict(self.token_usage) if self.token_usage else None,
            "step": self.step_number,
            "error": self.error.dict() if self.error else None,
            "model_output_message": self.model_output_message.dict() if self.model_output_message else None,
            "model_output": self.model_output,
            "observations": self.observations,
            "action_output": make_json_serializable(self.action_output),
        }

    def to_messages(self, summary_mode: bool = False) -> list[Message]:
        messages = []
        if self.model_output is not None and not summary_mode:
            messages.append(
                Message(role=MessageRole.ASSISTANT, content=[{"type": "text", "text": self.model_output.strip()}])
            )

        if self.tool_calls is not None:
            messages.append(
                Message(
                    role=MessageRole.TOOL_CALL,
                    content=[
                        {
                            "type": "text",
                            "text": "Calling tools:\n" + str([tc.dict() for tc in self.tool_calls]),
                        }
                    ],
                )
            )

        if self.observations_images:
            messages.append(
                Message(
                    role=MessageRole.USER,
                    content=[
                        {
                            "type": "image",
                            "image": image,
                        }
                        for image in self.observations_images
                    ],
                )
            )

        if self.observations is not None:
            messages.append(
                Message(
                    role=MessageRole.TOOL_RESPONSE,
                    content=[
                        {
                            "type": "text",
                            "text": f"Observation:\n{self.observations}",
                        }
                    ],
                )
            )
        if self.error is not None:
            error_message = (
                "Error:\n"
                + str(self.error)
                + "\nNow let's retry: take care not to repeat previous errors! If you have retried several times, try a completely different approach.\n"
            )
            message_content = f"Call id: {self.tool_calls[0].id}\n" if self.tool_calls else ""
            message_content += error_message
            messages.append(
                Message(role=MessageRole.TOOL_RESPONSE, content=[{"type": "text", "text": message_content}])
            )

        return messages


@dataclass
class PlanningStep(MemoryStep):
    model_input_messages: list[Message]
    model_output_message: ChatMessage
    plan: str
    timing: Timing
    token_usage: TokenUsage | None = None

    def to_messages(self, summary_mode: bool = False) -> list[Message]:
        if summary_mode:
            return []
        return [
            Message(role=MessageRole.ASSISTANT, content=[{"type": "text", "text": self.plan.strip()}]),
            Message(role=MessageRole.USER, content=[{"type": "text", "text": "Now proceed and carry out this plan."}]),
            # This second message creates a role change to prevent models models from simply continuing the plan message
        ]


@dataclass
class TaskStep(MemoryStep):
    task: str
    task_images: list[Any] | None = None

    def to_messages(self, summary_mode: bool = False) -> list[Message]:
        content = [{"type": "text", "text": f"New task:\n{self.task}"}]
        if self.task_images:
            for image in self.task_images:
                content.append({"type": "image", "image": image})

        return [Message(role=MessageRole.USER, content=content)]


@dataclass
class SystemPromptStep(MemoryStep):
    system_prompt: str

    def to_messages(self, summary_mode: bool = False) -> list[Message]:
        if summary_mode:
            return []
        return [Message(role=MessageRole.SYSTEM, content=[{"type": "text", "text": self.system_prompt}])]


@dataclass
class FinalAnswerStep(MemoryStep):
    output: Any

    def to_messages(self, summary_mode: bool = False) -> list[Message]:
        if summary_mode:
            return []
        return [
            Message(
                role=MessageRole.ASSISTANT,
                content=[{"type": "text", "text": f"Final answer:\n{self.output}"}],
            )
        ]


class AgentMemory:
    def __init__(self, system_prompt: str):
        self.system_prompt = SystemPromptStep(system_prompt=system_prompt)
        self.steps: list[MemoryStep] = []

    def reset(self):
        self.steps = []

    def get_succinct_steps(self) -> list[dict]:
        return [step.dict() for step in self.steps]

    def get_full_steps(self) -> list[dict]:
        return [step.dict() for step in self.steps]

    def replay(self, logger: AgentLogger, detailed: bool = False):
        """Prints a replay of the agent's steps."""
        logger.log("Agent Memory Replay:", level=LogLevel.INFO)
        logger.log(f"System Prompt: {self.system_prompt.system_prompt}", level=LogLevel.INFO)
        
        for i, step in enumerate(self.steps):
            logger.log(f"Step {i + 1}:", level=LogLevel.INFO)
            
            if isinstance(step, ActionStep):
                if step.model_output:
                    logger.log(f"Thought: {step.model_output}", level=LogLevel.INFO)
                if step.tool_calls:
                    for tool_call in step.tool_calls:
                        logger.log(f"Tool Call: {tool_call.name} with args: {tool_call.arguments}", level=LogLevel.INFO)
                if step.observations:
                    logger.log(f"Observation: {step.observations}", level=LogLevel.INFO)
                if step.error:
                    logger.log(f"Error: {step.error}", level=LogLevel.INFO)
                if step.timing:
                    logger.log(f"Duration: {step.timing.duration:.2f}s", level=LogLevel.INFO)
            elif isinstance(step, PlanningStep):
                logger.log(f"Planning: {step.plan}", level=LogLevel.INFO)
                if step.timing:
                    logger.log(f"Duration: {step.timing.duration:.2f}s", level=LogLevel.INFO)
            elif isinstance(step, TaskStep):
                logger.log(f"Task: {step.task}", level=LogLevel.INFO)
            elif isinstance(step, FinalAnswerStep):
                logger.log(f"Final Answer: {step.output}", level=LogLevel.INFO)
            
            logger.log("", level=LogLevel.INFO)


__all__ = ["AgentMemory"]
