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
import os
import pathlib
import tempfile
import uuid
from io import BytesIO

import requests

from .utils import _is_package_available


logger = logging.getLogger(__name__)


class AgentType:
    """
    Abstract class to be reimplemented to define types that can be returned by agents.

    These objects serve three purposes:

    - They behave as they were the type they're meant to be, e.g., a string for text, a PIL.Image.Image for images
    - They can be stringified: str(object) in order to return a string defining the object
    - They should be displayed correctly in ipython notebooks/colab/jupyter
    """

    def __init__(self, value):
        self._value = value

    def __str__(self):
        return self.to_string()

    def to_raw(self):
        logger.error(
            "This is a raw AgentType of unknown type. Display in notebooks and string conversion will be unreliable"
        )
        return self._value

    def to_string(self) -> str:
        logger.error(
            "This is a raw AgentType of unknown type. Display in notebooks and string conversion will be unreliable"
        )
        return str(self._value)


class AgentText(AgentType, str):
    """
    Text type returned by the agent. Behaves as a string.
    """

    def to_raw(self):
        return self._value

    def to_string(self):
        return str(self._value)


class AgentAudio(AgentType, str):
    """
    Audio type returned by the agent.
    """

    def __init__(self, value, samplerate=16_000):
        if not _is_package_available("soundfile") or not _is_package_available("torch"):
            raise ModuleNotFoundError(
                "Please install 'audio' extra to use AgentAudio: `pip install 'smolleragents[audio]'`"
            )
        import numpy as np

        super().__init__(value)

        self._path = None
        self._tensor = None

        self.samplerate = samplerate
        if isinstance(value, (str, pathlib.Path)):
            self._path = value
        elif isinstance(value, tuple):
            self.samplerate = value[0]
            if isinstance(value[1], np.ndarray):
                self._tensor = value[1]
            else:
                raise TypeError(f"Unsupported type for {self.__class__.__name__}: {type(value)}")
        else:
            raise TypeError(f"Unsupported type for {self.__class__.__name__}: {type(value)}")

    def _ipython_display_(self, include=None, exclude=None):
        """
        Displays correctly this type in an ipython notebook (ipython, colab, jupyter, ...)
        """
        from IPython.display import Audio, display

        display(Audio(self.to_string(), rate=self.samplerate))

    def to_raw(self):
        """
        Returns the "raw" version of that object. In the case of an AgentAudio, it is a numpy array.
        """
        if self._tensor is not None:
            return self._tensor

        if self._path is not None:
            import soundfile as sf

            self._tensor, self.samplerate = sf.read(self._path)
            return self._tensor

    def to_string(self):
        """
        Returns the stringified version of that object. In the case of an AgentAudio, it is a path to the serialized
        version of the audio.
        """
        if self._path is not None:
            return self._path

        if self._tensor is not None:
            import soundfile as sf

            directory = tempfile.mkdtemp()
            self._path = os.path.join(directory, str(uuid.uuid4()) + ".wav")
            sf.write(self._path, self._tensor, self.samplerate)
            return self._path


def handle_agent_input_types(*args, **kwargs):
    """Handle agent input types."""
    return args, kwargs


def handle_agent_output_types(output, output_type=None):
    """Handle agent output types."""
    if output_type == "text":
        return AgentText(output)
    elif output_type == "audio":
        return AgentAudio(output)
    else:
        return output
