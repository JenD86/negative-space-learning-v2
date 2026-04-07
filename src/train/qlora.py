from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence, cast

from unsloth import FastLanguageModel
from datasets import Dataset
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAX_SEQ_LENGTH = 2048
DEFAULT_ADAPTER_EXPORT_ROOT = PROJECT_ROOT / "models" / "adapters"
DEFAULT_MERGED_EXPORT_ROOT = PROJECT_ROOT / "models" / "merged"
TrainingExportFormat = Literal["peft", "merged_16bit", "gguf"]
ConfiguredTrainingExportFormat = Literal["auto", "peft", "merged_16bit", "gguf"]
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

SUPPORTED_TRAINING_BASE_MODEL_EXAMPLES = (
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
)


def _export_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_training_base_model(base_model: str) -> None:
    lower_model = base_model.lower()
    invalid_format = None
    if lower_model.endswith(".gguf") or "gguf" in lower_model:
        invalid_format = "GGUF"
    elif "awq" in lower_model:
        invalid_format = "AWQ"
    elif "gptq" in lower_model:
        invalid_format = "GPTQ"

    if invalid_format is None:
        return

    examples = ", ".join(
        repr(example) for example in SUPPORTED_TRAINING_BASE_MODEL_EXAMPLES
    )
    raise ValueError(
        f"Unsupported training base model '{base_model}': {invalid_format} is an inference/export format, not a Transformers training checkpoint. "
        f"Use a Transformers or Unsloth BnB model instead, for example {examples}."
    )


def _load_base_model(
    base_model: str,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
):
    _validate_training_base_model(base_model)

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
    except RuntimeError as exc:
        if "No config file found" in str(exc):
            examples = ", ".join(
                repr(example) for example in SUPPORTED_TRAINING_BASE_MODEL_EXAMPLES
            )
            raise RuntimeError(
                f"Failed to load training base model '{base_model}'. "
                f"Use a Transformers-compatible training checkpoint or an Unsloth BnB checkpoint, for example {examples}."
            ) from exc
        raise
    return model, tokenizer


def _attach_lora_adapter(model: Any, config: dict[str, Any]) -> Any:
    return FastLanguageModel.get_peft_model(
        model,
        r=config.get("rank", 32),
        target_modules=config.get("target_modules", LORA_TARGET_MODULES),
        lora_alpha=config.get("alpha", 32),
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )


def _write_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def resolve_training_export_format(
    model_name: str,
    configured_format: ConfiguredTrainingExportFormat = "auto",
) -> TrainingExportFormat:
    if configured_format != "auto":
        return cast(TrainingExportFormat, configured_format)

    backend = model_name.strip().split(":", 1)[0]
    if backend == "llama":
        return "gguf"
    return "peft"


def _save_training_artifact(
    model: Any,
    tokenizer: Any,
    output_path: Path,
    *,
    export_format: TrainingExportFormat,
    gguf_quantize: str = "f16",
) -> Path:

    output_path.mkdir(parents=True, exist_ok=True)

    if "gguf" not in export_format:
        model.save_pretrained_merged(output_path, tokenizer, save_method=export_format)
        return output_path

    model.save_pretrained_gguf(
        output_path, tokenizer, quantization_method=gguf_quantize
    )
    return output_path


def _training_metadata_path(artifact_path: Path) -> Path:
    if artifact_path.suffix == ".gguf":
        return artifact_path.with_suffix(".training_info.json")
    return artifact_path / "training_info.json"


def _load_sft_dataset(training_data_paths: Sequence[Path], tokenizer: Any) -> Any:
    """Load one or more generation JSONL files into a chat-formatted Dataset."""

    if Dataset is None:
        raise RuntimeError("datasets is required for SFT training")

    rows: list[dict[str, str]] = []
    for training_data_path in training_data_paths:
        resolved_path = Path(training_data_path).expanduser().resolve()
        with open(resolved_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue

                payload = json.loads(line)
                if not payload.get("success"):
                    continue

                prompt = payload.get("prompt")
                raw_response = payload.get("raw_response")
                if not isinstance(prompt, str) or not isinstance(raw_response, str):
                    continue

                text = tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": raw_response},
                    ],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                rows.append({"text": text})

    if not rows:
        raise ValueError("No training rows found")

    return Dataset.from_list(rows)


def train_sft(
    base_model: str,
    training_data_paths: Sequence[str],
    output_dir: str,
    *,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    max_steps: int = 50,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    warmup_steps: int = 5,
    logging_steps: int = 1,
    report_to: str = "none",
    export_format: TrainingExportFormat = "peft",
    gguf_quantize: str = "f16",
) -> Path:
    """Run SFT on the provided generation window and save a trained LoRA adapter."""

    model, tokenizer = _load_base_model(
        base_model,
        max_seq_length=max_seq_length,
    )
    model = _attach_lora_adapter(model, {})

    resolved_output_dir = Path(output_dir).expanduser().resolve()
    if export_format != "gguf":
        resolved_output_dir.mkdir(parents=True, exist_ok=True)

    resolved_training_paths = [
        str(Path(path).expanduser().resolve()) for path in training_data_paths
    ]
    dataset = _load_sft_dataset(
        [Path(path) for path in resolved_training_paths],
        tokenizer,
    )
    row_count = len(dataset)

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=str(resolved_output_dir),
            max_steps=max_steps,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_strategy="no",
            report_to=report_to,
            dataset_text_field="text",
            max_length=max_seq_length,
        ),
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    artifact_path = _save_training_artifact(
        model,
        tokenizer,
        resolved_output_dir,
        export_format=export_format,
        gguf_quantize=gguf_quantize,
    )
    _write_metadata(
        _training_metadata_path(artifact_path),
        {
            "base_model": base_model,
            "max_steps": max_steps,
            "learning_rate": learning_rate,
            "row_count": row_count,
            "training_data_paths": resolved_training_paths,
            "export_format": export_format,
            "exported_at": _export_timestamp(),
        },
    )
    return artifact_path


def export_lora_adapter(
    base_model: str,
    output_dir: str,
    *,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
) -> Path:
    """Export a PEFT LoRA adapter directory for vLLM or llama.cpp."""

    resolved_output_dir = Path(output_dir).expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = _load_base_model(
        base_model,
        max_seq_length=max_seq_length,
    )
    model = _attach_lora_adapter(model, {})
    model.save_pretrained_merged(resolved_output_dir, tokenizer, save_method="lora")

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

    model, tokenizer = _load_base_model(
        base_model,
        max_seq_length=max_seq_length,
    )
    model = _attach_lora_adapter(model, {})

    model.save_pretrained_merged(
        resolved_output_dir, tokenizer, save_method="merged_16bit"
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
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--export-lora", action="store_true")
    mode.add_argument("--export-merged", action="store_true")
    mode.add_argument("--convert-adapter", action="store_true")
    parser.add_argument("--base-model")
    parser.add_argument("--adapter-dir")
    parser.add_argument("--training-data", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--report-to", default="none")
    parser.add_argument(
        "--export-format",
        choices=("peft", "merged_16bit", "gguf"),
        default="peft",
    )
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
        if args.train:
            if not args.training_data:
                parser.error("--training-data is required with --train")
            output_path = train_sft(
                args.base_model,
                args.training_data,
                args.output,
                max_seq_length=args.max_seq_length,
                max_steps=args.max_steps,
                per_device_train_batch_size=args.per_device_train_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                learning_rate=args.learning_rate,
                warmup_steps=args.warmup_steps,
                logging_steps=args.logging_steps,
                report_to=args.report_to,
                export_format=args.export_format,
                gguf_quantize=args.quantize,
            )
        elif args.export_lora:
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
    "resolve_training_export_format",
    "train_sft",
    "convert_adapter_to_gguf",
    "export_lora_adapter",
    "export_merged_model",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
