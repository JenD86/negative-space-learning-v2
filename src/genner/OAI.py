import ast
import json
import re
from typing import Any, List, Optional, Protocol, Sequence, Tuple, cast

from openai import OpenAI
from result import Result, Ok, Err, UnwrapError

from src.observability.types import InferenceResult, UsageInfo
from src.typing.message import Message
from .Base import Genner
from dataclasses import dataclass, field

from typing import Dict, NamedTuple, Protocol, runtime_checkable
from pprint import pformat
from dataclasses import dataclass, field


@runtime_checkable
class OAICompatibleConfig(Protocol):
    name: str
    model: str
    max_tokens: int
    temperature: float


@dataclass
class PList:
    messages: List[Message] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.messages)

    def __add__(self, other: "PList") -> "PList":
        return PList(messages=self.messages + other.messages)

    def __repr__(self) -> str:
        messages_repr = pformat(self.messages)
        return f"PList(\n\tmessages=[\n\t\t{messages_repr}\n\t\t]\n)"


class OAIConfig(NamedTuple):
    name: str = "OpenAI"
    model: str = "gpt-3.5-turbo"
    max_tokens: int = 500
    temperature: float = 0.5


class OAIUsageResponse(Protocol):
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


class OAIChoiceResponse(Protocol):
    finish_reason: str | None


class OAIChatResponse(Protocol):
    usage: OAIUsageResponse | None
    choices: Sequence[OAIChoiceResponse]
    model: str


class OAIGenner(Genner):
    def __init__(
        self,
        client: OpenAI,
        config: OAICompatibleConfig,
        identifier: str = "oai",
    ):
        super().__init__(identifier)

        self.client = client
        self.config = config

    def plist_completion(self, messages: List[Message]) -> Result[InferenceResult, str]:
        try:
            print(self.config.model)
            response = self.client.chat.completions.create(
                model=self.config.model.strip(),
                # response_format={"type": "json_object"},
                messages=cast(Any, messages),
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

            assert isinstance(response.choices[0].message.content, str)

            return Ok(
                InferenceResult(
                    content=response.choices[0].message.content,
                    usage=self.get_usage_info(response),
                )
            )
        except Exception as e:
            import traceback

            print(traceback.format_exc())
            return Err(
                "OAIGenner.plist_completion: Unexpected error,\n"
                f"`messages`: \n{messages}\n"
                f"`e`: \n{e}"
            )

    def generate_code(
        self, messages: List[Message]
    ) -> Result[Tuple[str, str], Tuple[str, Optional[str]]]:
        raw_response: Optional[str] = None
        try:
            raw_response = self.plist_completion(messages).unwrap().content
            extracted_code = self.extract_code(raw_response).unwrap()
            return Ok((extracted_code, raw_response))
        except UnwrapError as e:
            error_message = (
                "OAIGenner.generate_code: Unwrap error,\n"
                f"`self.config.model`: {self.config.model}\n"
                f"`e.result.err()`: \n{e.result.err()}\n"
            )
            return Err((error_message, raw_response))
        except Exception as e:
            error_message = (
                "OAIGenner.generate_code: Unexpected error,\n"
                f"`self.config.model`: {self.config.model}\n"
                f"`messages`: \n{messages}\n"
                f"`e`: \n{e}\n"
            )
            return Err((error_message, raw_response))

    @staticmethod
    def extract_code(response: str) -> Result[str, str]:
        try:
            # Extract code from the response
            regex_pattern = r"```python\n([\s\S]*?)```"
            code_match = re.search(regex_pattern, response, re.DOTALL)
            assert code_match is not None, "`code_match` is None"

            code_string = code_match.group(1)
            assert code_string is not None, "`code_string` is None"

            return Ok(code_string)
        except Exception as e:
            return Err(
                "OAIGenner.extract_code: Unexpected error,\n"
                f"`response`: \n{response}\n"
                f"`e`: \n{e}\n"
            )

    def generate_list(
        self, messages: List[Message]
    ) -> Result[Tuple[List[str], str], Tuple[str, Optional[str]]]:
        raw_response: Optional[str] = None
        try:
            raw_response = self.plist_completion(messages).unwrap().content
            processed_list = self.extract_list(raw_response).unwrap()
            return Ok((processed_list, raw_response))
        except UnwrapError as e:
            error_message = (
                "OAIGenner.generate_list: Unwrap error,\n"
                f"`self.config.model`: {self.config.model}\n"
                f"`e.result.err()`: \n{e.result.err()}\n"
            )
            return Err((error_message, raw_response))
        except Exception as e:
            error_message = (
                "OAIGenner.generate_list: Unexpected error,\n"
                f"`self.config.model`: {self.config.model}\n"
                f"`messages`: \n{messages}\n"
                f"`e`: \n{e}\n"
            )
            return Err((error_message, raw_response))

    @staticmethod
    def extract_list(response: str) -> Result[List[str], str]:
        try:
            # Remove markdown code block markers and "json" label
            json_str = response.replace("```json", "").replace("```", "").strip()

            expected_keys = ["strategies", "strats", "strategy", "Strategies", "Strats"]

            for key in expected_keys:
                if key in json_str:
                    processed_list = json.loads(json_str)[key]
                    break
            else:
                return Err(
                    f"No strategies found in the response, expected keys: {expected_keys}"
                )

            # Validate types
            assert isinstance(processed_list, list), "`processed_list` is not a `list`"
            assert all(isinstance(item, str) for item in processed_list), (
                "All items in `processed_list` must be strings"
            )

            return Ok(processed_list)
        except Exception as e:
            return Err(
                "OAIGenner.extract_list: Unexpected error,\n"
                f"`response`: \n{response}\n"
                f"`e`: \n{e}\n"
            )

    @staticmethod
    def get_usage_info(response: object) -> UsageInfo:
        response = cast(OAIChatResponse, response)
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage is not None else None
        completion_tokens = usage.completion_tokens if usage is not None else None
        total_tokens = usage.total_tokens if usage is not None else None
        if total_tokens is None and (
            prompt_tokens is not None or completion_tokens is not None
        ):
            total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

        stop_reason = response.choices[0].finish_reason if response.choices else None

        return UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            model=response.model,
            stop_reason=stop_reason,
        )
