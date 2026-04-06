import ast
import json
import re
from pprint import pformat
from typing import List, Optional, Protocol, Tuple, cast

import anthropic
from result import Result, Ok, Err, UnwrapError

from src.observability.types import InferenceResult, UsageInfo
from src.typing.message import Message
from .Base import Genner
from dataclasses import dataclass, field

from typing import Dict, TypedDict, Any
from typing import Dict, NamedTuple


class ClaudeUsageResponse(Protocol):
    input_tokens: int | None
    output_tokens: int | None


class ClaudeMessageResponse(Protocol):
    usage: ClaudeUsageResponse
    model: str
    stop_reason: str | None


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


class ClaudeConfig(NamedTuple):
    name: str = "Claude"
    model: str = "claude-3-5-sonnet-20241022"  # Latest Claude model
    max_tokens: int = 1000
    temperature: float = 0.5


class ClaudeGenner(Genner):
    def __init__(self, client: anthropic.Anthropic, config: ClaudeConfig):
        super().__init__("claude")

        self.client = client
        self.config = config

    def plist_completion(self, messages: List[Message]) -> Result[InferenceResult, str]:
        try:
            # Convert messages format for Claude
            claude_messages = []
            system_message = ""

            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    claude_messages.append(
                        {"role": msg["role"], "content": msg["content"]}
                    )

            # Create Claude API call
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=system_message
                if system_message
                else "You are a helpful assistant.",
                messages=claude_messages,
            )

            return Ok(
                InferenceResult(
                    content=response.content[0].text,
                    usage=self.get_usage_info(response),
                )
            )

        except Exception as e:
            import traceback

            print(traceback.format_exc())
            return Err(
                "ClaudeGenner.plist_completion: Unexpected error,\n"
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
                "ClaudeGenner.generate_code: Unwrap error,\n"
                f"`self.config.model`: {self.config.model}\n"
                f"`e.result.err()`: \n{e.result.err()}\n"
            )
            return Err((error_message, raw_response))
        except Exception as e:
            error_message = (
                "ClaudeGenner.generate_code: Unexpected error,\n"
                f"`self.config.model`: {self.config.model}\n"
                f"`messages`: \n{messages}\n"
                f"`e`: \n{e}\n"
            )
            return Err((error_message, raw_response))

    def generate_list(
        self, messages: List[Message]
    ) -> Result[Tuple[List[str], str], Tuple[str, Optional[str]]]:
        raw_response: Optional[str] = None
        try:
            raw_response = self.plist_completion(messages).unwrap().content
            extracted_list = self.extract_list(raw_response).unwrap()
            return Ok((extracted_list, raw_response))
        except UnwrapError as e:
            error_message = (
                "ClaudeGenner.generate_list: Unwrap error,\n"
                f"`self.config.model`: {self.config.model}\n"
                f"`e.result.err()`: \n{e.result.err()}\n"
            )
            return Err((error_message, raw_response))
        except Exception as e:
            error_message = (
                "ClaudeGenner.generate_list: Unexpected error,\n"
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
            return Err(f"ClaudeGenner.extract_code: {e}")

    @staticmethod
    def extract_list(response: str) -> Result[List[str], str]:
        # Try JSON parsing first (similar to Qwen implementation)
        pass_1_exception: Exception | None = None
        try:
            import json

            # Remove markdown code block markers and "json" label
            json_str = response.replace("```json", "").replace("```", "").strip()

            # Extract only the JSON part (from first { to matching })
            json_start = json_str.find("{")
            if json_start == -1:
                raise ValueError("No JSON object found")

            brace_count = 0
            json_end = json_start
            for i in range(json_start, len(json_str)):
                if json_str[i] == "{":
                    brace_count += 1
                elif json_str[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break

            if brace_count != 0:
                raise ValueError("Unmatched braces in JSON")

            clean_json = json_str[json_start:json_end]

            expected_keys = ["strategies", "strats", "strategy", "Strategies", "Strats"]

            parsed_json = json.loads(clean_json)
            for key in expected_keys:
                if key in parsed_json:
                    processed_list = parsed_json[key]
                    break
            else:
                raise ValueError("No matching strategy keys found")

            # Validate types
            assert isinstance(processed_list, list), "`processed_list` is not a `list`"
            assert all(isinstance(item, str) for item in processed_list), (
                "All items in `processed_list` must be strings"
            )

            return Ok(processed_list)
        except Exception as exc:
            pass_1_exception = exc

        # Fallback: extract from markdown list format
        assert pass_1_exception is not None
        try:
            match = re.search(r"```(.*)```", response, re.DOTALL)

            if not match:
                # Try to find numbered or bulleted lists directly
                strategies = []
                lines = response.split("\n")

                for line in lines:
                    line = line.strip()
                    # Check for numbered list
                    if re.match(r"^\d+\.", line):
                        strategy = re.sub(r"^\d+\.\s*", "", line)
                        if strategy:
                            strategies.append(strategy)
                    # Check for bullet points
                    elif line.startswith("- ") or line.startswith("* "):
                        strategy = line[2:].strip()
                        if strategy:
                            strategies.append(strategy)

                if strategies:
                    return Ok(strategies)
                else:
                    return Err("No list format found in response")

            # Extract the content inside the backticks
            content_inside_backticks = match.group(1).strip()

            processed_list = []
            lines = content_inside_backticks.strip().split("\n")

            for line in lines:
                if line.strip().startswith("-"):
                    list_item = line.strip().lstrip("- ").strip()
                    processed_list.append(list_item)

            return Ok(processed_list)

        except Exception as pass_2_exception:
            return Err(
                "ClaudeGenner.extract_list: Failed to parse response as JSON and also failed to extract from markdown, "
                f"`pass_1_exception`: \n{pass_1_exception}\n"
                f"`pass_2_exception`: \n{pass_2_exception}\n"
                f"`response`: \n{response}\n"
            )

    @staticmethod
    def get_usage_info(response: object) -> UsageInfo:
        response = cast(ClaudeMessageResponse, response)
        usage = response.usage
        prompt_tokens = usage.input_tokens
        completion_tokens = usage.output_tokens
        total_tokens = None
        if prompt_tokens is not None or completion_tokens is not None:
            total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

        return UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            model=response.model,
            stop_reason=response.stop_reason,
        )
