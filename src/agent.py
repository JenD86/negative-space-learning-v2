from typing import Dict, List, Optional, Tuple

from loguru import logger
from result import Err, Ok, Result

from src.genner.Base import Genner
from src.prompt import (
    get_regen_code_req_prompt,
    get_regen_list_req_prompt,
    get_sp_egc_req_prompt,
    get_strategy_code_req_prompt,
    get_strategy_list_req_prompt,
    get_system_prompt,
    get_env_discovery_regen_prompt
)
from src.typing.message import Message
from src.typing.alias import RawResponse, ParsedCode


def _message(role: str, content: str, phase: str) -> Message:
    return {
        "role": role,
        "content": content,
        "meta": {"phase": phase},
    }


def generate_special_environment_getter_code(
    genner: Genner,
    env_infos: List[str],
    special_env_infos: List[str],
) -> Result[Tuple[RawResponse, List[Message]], str]:
    messages: List[Message] = [
        _message(
            "system",
            get_system_prompt().formatted_prompt,
            "special_environment_getter_code",
        ),
        _message(
            "user",
            get_sp_egc_req_prompt(
                basic_env_infos=env_infos, special_env_infos=special_env_infos
            ).formatted_prompt,
            "special_environment_getter_code",
        ),
    ]

    match genner.plist_completion(messages):
        case Ok(inference_result):
            return Ok((inference_result.content, messages))
        case Err(err):
            return Err(err)

def regenerate_env_discovery_code(
    genner: Genner,
    regen_count: int,
    error_sources: List[str],
    error_contexts: List[str],
    latest_generation: str,
) -> Result[Tuple[RawResponse, List[Message]], str]:
    messages: List[Message] = [
        _message(
            "system",
            get_system_prompt().formatted_prompt,
            "special_environment_getter_code_regeneration",
        ),
        _message(
            "user",
            get_env_discovery_regen_prompt(
                regen_count=regen_count,
                error_sources=error_sources,
                error_contexts=error_contexts,
                latest_generation=latest_generation,
            ).formatted_prompt,
            "special_environment_getter_code_regeneration",
        ),
    ]

    match genner.plist_completion(messages):
        case Ok(inference_result):
            return Ok((inference_result.content, messages))
        case Err(err):
            return Err(err)
        
def generate_strategy_list(
    genner: Genner,
    env_infos: List[str],
    special_env_infos: List[str],
    previous_strategies: List[str],
) -> Result[Tuple[RawResponse, List[Message]], str]:
    messages: List[Message] = [
        _message(
            "system",
            get_system_prompt().formatted_prompt,
            "strategy_list",
        ),
        _message(
            "user",
            get_strategy_list_req_prompt(
                basic_env_infos=env_infos,
                special_env_infos=special_env_infos,
                previous_strategies=previous_strategies,
            ).formatted_prompt,
            "strategy_list",
        ),
    ]

    match genner.plist_completion(messages):
        case Ok(inference_result):
            return Ok((inference_result.content, messages))
        case Err(err):
            return Err(err)


def generate_strategy_code(
    genner: Genner,
    strategy: str,
    env_infos: List[str],
    special_env_infos: List[str],
) -> Result[Tuple[RawResponse, List[Message]], str]:
    messages: List[Message] = [
        _message(
            "system",
            get_system_prompt().formatted_prompt,
            "strategy_code",
        ),
        _message(
            "user",
            get_strategy_code_req_prompt(
                strategy=strategy,
                basic_env_infos=env_infos,
                special_env_infos=special_env_infos,
            ).formatted_prompt,
            "strategy_code",
        ),
    ]

    match genner.plist_completion(messages):
        case Ok(inference_result):
            return Ok((inference_result.content, messages))
        case Err(err):
            return Err(err)


def regenerate_code(
    genner: Genner,
    regen_count: int,
    error_sources: List[str],
    error_contexts: List[str],
    latest_generation: str,
) -> Result[Tuple[RawResponse, List[Message]], str]:
    messages: List[Message] = [
        _message(
            "system",
            get_system_prompt().formatted_prompt,
            "strategy_code_regeneration",
        ),
        _message(
            "user",
            get_regen_code_req_prompt(
                regen_count=regen_count,
                error_sources=error_sources,
                error_contexts=error_contexts,
                latest_generation=latest_generation,
            ).formatted_prompt,
            "strategy_code_regeneration",
        ),
    ]

    match genner.plist_completion(messages):
        case Ok(inference_result):
            return Ok((inference_result.content, messages))
        case Err(err):
            return Err(err)


def regenerate_list(
    genner: Genner,
    regen_count: int,
    error_contexts: List[str],
    latest_generation: str,
) -> Result[Tuple[RawResponse, List[Message]], str]:
    messages: List[Message] = [
        _message(
            "system",
            get_system_prompt().formatted_prompt,
            "strategy_list_regeneration",
        ),
        _message(
            "user",
            get_regen_list_req_prompt(
                regen_count=regen_count,
                error_contexts=error_contexts,
                latest_generation=latest_generation,
            ).formatted_prompt,
            "strategy_list_regeneration",
        ),
    ]

    match genner.plist_completion(messages):
        case Ok(inference_result):
            return Ok((inference_result.content, messages))
        case Err(err):
            return Err(err)
