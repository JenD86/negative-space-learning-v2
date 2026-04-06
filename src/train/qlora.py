from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAX_SEQ_LENGTH = 2048
DEFAULT_ADAPTER_EXPORT_ROOT = PROJECT_ROOT / "models" / "adapters"
DEFAULT_MERGED_EXPORT_ROOT = PROJECT_ROOT / "models" / "merged"
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _export_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_base_model(
    base_model: str,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
) -> tuple[Any, Any, Any]:
    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise RuntimeError(
            "unsloth is required for qlora export commands. Install the ML dependencies first."
        ) from exc

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    return model, tokenizer, FastLanguageModel


def _attach_lora_adapter(fast_language_model: Any, model: Any) -> Any:
    return fast_language_model.get_peft_model(
        model,
        r=32,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )


def _write_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _save_lora_adapter(model: Any, tokenizer: Any, output_dir: Path) -> None:
    save_pretrained_merged = getattr(model, "save_pretrained_merged", None)
    if callable(save_pretrained_merged):
        save_pretrained_merged(str(output_dir), tokenizer, save_method="lora")
        return

    save_pretrained = getattr(model, "save_pretrained", None)
    if not callable(save_pretrained):
        raise RuntimeError("Model does not expose a supported LoRA export method")

    save_pretrained(str(output_dir))
    tokenizer_save = getattr(tokenizer, "save_pretrained", None)
    if callable(tokenizer_save):
        tokenizer_save(str(output_dir))


def export_lora_adapter(
    base_model: str,
    output_dir: str,
    *,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
) -> Path:
    """Export a PEFT LoRA adapter directory for vLLM or llama.cpp."""

    resolved_output_dir = Path(output_dir).expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, fast_language_model = _load_base_model(
        base_model,
        max_seq_length=max_seq_length,
    )
    model = _attach_lora_adapter(fast_language_model, model)
    _save_lora_adapter(model, tokenizer, resolved_output_dir)

    _write_metadata(
        resolved_output_dir / "adapter_info.json",
        {
            "base_model": base_model,
            "export_method": "lora",
            "max_seq_length": max_seq_length,
            "exported_at": _export_timestamp(),
        },
    )
    return resolved_output_dir


def export_merged_model(
    base_model: str,
    output_dir: str,
    *,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
) -> Path:
    """Merge the adapter into the base model and export fp16 safetensors."""

    resolved_output_dir = Path(output_dir).expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, fast_language_model = _load_base_model(
        base_model,
        max_seq_length=max_seq_length,
    )
    model = _attach_lora_adapter(fast_language_model, model)

    save_pretrained_merged = getattr(model, "save_pretrained_merged", None)
    if not callable(save_pretrained_merged):
        raise RuntimeError("Model does not expose save_pretrained_merged()")

    save_pretrained_merged(
        str(resolved_output_dir),
        tokenizer,
        save_method="merged_16bit",
    )
    _write_metadata(
        resolved_output_dir / "model_info.json",
        {
            "base_model": base_model,
            "export_method": "merged_16bit",
            "max_seq_length": max_seq_length,
            "exported_at": _export_timestamp(),
        },
    )
    return resolved_output_dir


def _resolve_convert_lora_script() -> Path:
    candidates = [
        PROJECT_ROOT / ".llama" / "convert_lora_to_gguf.py",
        PROJECT_ROOT / ".llama" / "llama.cpp" / "convert_lora_to_gguf.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    script_on_path = shutil.which("convert_lora_to_gguf.py")
    if script_on_path is not None:
        return Path(script_on_path)

    raise FileNotFoundError(
        "Could not find convert_lora_to_gguf.py. Install llama.cpp or place the script under .llama/."
    )


def convert_adapter_to_gguf(
    adapter_dir: str,
    output_path: str,
    quantize: str = "f16",
) -> Path:
    """Convert a PEFT LoRA adapter directory into GGUF format for llama.cpp."""

    resolved_adapter_dir = Path(adapter_dir).expanduser().resolve()
    if not resolved_adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {resolved_adapter_dir}")

    resolved_output_path = Path(output_path).expanduser().resolve()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    convert_script = _resolve_convert_lora_script()

    command = [
        sys.executable,
        str(convert_script),
        "--outfile",
        str(resolved_output_path),
        "--outtype",
        quantize,
        str(resolved_adapter_dir),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to convert LoRA adapter to GGUF: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return resolved_output_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="QLoRA export helpers for the NSL finetuning pipeline."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--export-lora", action="store_true")
    mode.add_argument("--export-merged", action="store_true")
    mode.add_argument("--convert-adapter", action="store_true")
    parser.add_argument("--base-model")
    parser.add_argument("--adapter-dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--quantize", default="f16")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.convert_adapter:
        if not args.adapter_dir:
            parser.error("--adapter-dir is required with --convert-adapter")
        output_path = convert_adapter_to_gguf(
            args.adapter_dir,
            args.output,
            quantize=args.quantize,
        )
    else:
        if not args.base_model:
            parser.error("--base-model is required for export commands")
        if args.export_lora:
            output_path = export_lora_adapter(
                args.base_model,
                args.output,
                max_seq_length=args.max_seq_length,
            )
        else:
            output_path = export_merged_model(
                args.base_model,
                args.output,
                max_seq_length=args.max_seq_length,
            )

    print(output_path)
    return 0


__all__ = [
    "DEFAULT_ADAPTER_EXPORT_ROOT",
    "DEFAULT_MERGED_EXPORT_ROOT",
    "convert_adapter_to_gguf",
    "export_lora_adapter",
    "export_merged_model",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
