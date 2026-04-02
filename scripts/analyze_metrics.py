from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Sequence


def discover_metrics_files(paths: Sequence[str | Path]) -> list[Path]:
    discovered: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if path.is_file():
            if path.name.startswith("metrics_") and path.suffix == ".jsonl":
                discovered.add(path.resolve())
            continue
        if path.is_dir():
            for match in path.rglob("metrics_*.jsonl"):
                if match.is_file():
                    discovered.add(match.resolve())
    return sorted(discovered)


def discover_run_log_files(paths: Sequence[str | Path]) -> list[Path]:
    discovered: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if path.is_file():
            if _is_run_log_file(path):
                discovered.add(path.resolve())
            continue
        if path.is_dir():
            for pattern in ("LAST_RUN_LOG.json", "*_full_run_*.json"):
                for match in path.rglob(pattern):
                    if match.is_file() and _is_run_log_file(match):
                        discovered.add(match.resolve())
    return sorted(discovered)


def analyze_paths(paths: Sequence[str | Path]) -> dict[str, Any]:
    metric_files = discover_metrics_files(paths)
    run_log_files = discover_run_log_files(paths)
    run_metadata = load_run_metadata(run_log_files)

    records_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_inference_records: list[dict[str, Any]] = []
    all_phase_records: list[dict[str, Any]] = []

    for metric_file in metric_files:
        for record in load_metrics_file(metric_file):
            run_id = record.get("run_id")
            if not isinstance(run_id, str) or not run_id:
                continue
            records_by_run[run_id].append(record)
            metric_type = record.get("metric_type")
            if metric_type == "inference":
                all_inference_records.append(record)
            elif metric_type == "phase":
                all_phase_records.append(record)

    run_summaries = [
        summarize_run(run_id, records, run_metadata.get(run_id, {}))
        for run_id, records in sorted(records_by_run.items())
    ]

    return {
        "run_count": len(run_summaries),
        "metric_file_count": len(metric_files),
        "run_log_count": len(run_log_files),
        "runs": run_summaries,
        "comparison_tables": {
            "latency_by_model": build_latency_by_model_table(run_summaries),
            "retry_rates_by_phase": build_retry_rates_by_phase_table(all_phase_records),
            "memory_profiles_by_model": build_memory_profiles_by_model_table(
                all_inference_records
            ),
            "generation_benchmarks_by_backend": build_generation_benchmarks_by_backend_table(
                all_inference_records
            ),
        },
    }


def load_metrics_file(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    rows: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def load_run_metadata(paths: Sequence[str | Path]) -> dict[str, dict[str, Any]]:
    metadata_by_run: dict[str, dict[str, Any]] = {}
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            continue
        run_id = payload.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            continue
        current = metadata_by_run.setdefault(run_id, {})
        for key in ("commit_id", "config", "metrics_path"):
            value = payload.get(key)
            if value not in (None, ""):
                current[key] = value
    return metadata_by_run


def summarize_run(
    run_id: str,
    records: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    inference_records = [
        record for record in records if record.get("metric_type") == "inference"
    ]
    phase_records = [record for record in records if record.get("metric_type") == "phase"]

    inference_calls = len(inference_records)
    successful_inference_calls = sum(1 for record in inference_records if record.get("success"))
    failed_inference_calls = inference_calls - successful_inference_calls
    phase_count = len(phase_records)
    total_retry_count = sum(_as_int(record.get("retry_count")) for record in phase_records)

    total_inference_latency_ms = sum(
        _as_float(record.get("latency_ms")) for record in inference_records
    )
    average_inference_latency_ms = (
        total_inference_latency_ms / inference_calls if inference_calls else 0.0
    )
    total_phase_duration_ms = sum(
        _as_float(record.get("duration_ms")) for record in phase_records
    )

    usage_dicts = [_usage(record) for record in inference_records]
    total_input_tokens = sum(_as_int(usage.get("prompt_tokens")) for usage in usage_dicts)
    total_output_tokens = sum(
        _as_int(usage.get("completion_tokens")) for usage in usage_dicts
    )
    total_tokens = sum(_usage_total_tokens(usage) for usage in usage_dicts)

    output_tokens_per_second = 0.0
    if total_inference_latency_ms > 0:
        output_tokens_per_second = total_output_tokens / (total_inference_latency_ms / 1000)

    prompt_rates = _collect_generation_rates(inference_records, "prompt_tokens_per_second")
    output_rates = _collect_generation_rates(inference_records, "output_tokens_per_second")
    total_rates = _collect_generation_rates(inference_records, "total_tokens_per_second")

    host_memory_samples = [
        _as_float(record.get("host_memory_mb"))
        for record in inference_records
        if _has_number(record.get("host_memory_mb"))
    ]
    gpu_memory_samples = [
        _as_float(record.get("gpu_memory_mb"))
        for record in inference_records
        if _has_number(record.get("gpu_memory_mb"))
    ]

    backend = infer_backend(inference_records)
    model = infer_model(inference_records)

    return {
        "run_id": run_id,
        "commit_id": metadata.get("commit_id", "unknown"),
        "config": metadata.get("config", "unknown"),
        "backend": backend,
        "model": model,
        "inference_calls": inference_calls,
        "successful_inference_calls": successful_inference_calls,
        "failed_inference_calls": failed_inference_calls,
        "success_rate": (
            successful_inference_calls / inference_calls if inference_calls else 0.0
        ),
        "phase_count": phase_count,
        "total_retry_count": total_retry_count,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "total_inference_latency_ms": total_inference_latency_ms,
        "average_inference_latency_ms": average_inference_latency_ms,
        "total_phase_duration_ms": total_phase_duration_ms,
        "output_tokens_per_second": output_tokens_per_second,
        "generation_count": len(total_rates),
        "average_prompt_tokens_per_second": _average_or_zero(prompt_rates),
        "average_output_tokens_per_second": _average_or_zero(output_rates),
        "average_total_tokens_per_second": _average_or_zero(total_rates),
        "median_total_tokens_per_second": _median_or_none(total_rates),
        "peak_total_tokens_per_second": max(total_rates) if total_rates else None,
        "average_host_memory_mb": _average_or_none(host_memory_samples),
        "peak_host_memory_mb": max(host_memory_samples) if host_memory_samples else None,
        "average_gpu_memory_mb": _average_or_none(gpu_memory_samples),
        "peak_gpu_memory_mb": max(gpu_memory_samples) if gpu_memory_samples else None,
    }


def build_latency_by_model_table(run_summaries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary in run_summaries:
        grouped[str(summary.get("model", "unknown"))].append(summary)

    rows: list[dict[str, Any]] = []
    for model, summaries in sorted(grouped.items()):
        total_inference_calls = sum(_as_int(item.get("inference_calls")) for item in summaries)
        total_latency_ms = sum(
            _as_float(item.get("total_inference_latency_ms")) for item in summaries
        )
        total_output_tokens = sum(
            _as_int(item.get("total_output_tokens")) for item in summaries
        )
        average_latency_ms = (
            total_latency_ms / total_inference_calls if total_inference_calls else 0.0
        )
        average_output_tokens_per_second = 0.0
        if total_latency_ms > 0:
            average_output_tokens_per_second = total_output_tokens / (total_latency_ms / 1000)
        rows.append(
            {
                "model": model,
                "run_count": len(summaries),
                "total_inference_calls": total_inference_calls,
                "total_latency_ms": total_latency_ms,
                "average_latency_ms": average_latency_ms,
                "average_output_tokens_per_second": average_output_tokens_per_second,
            }
        )
    return rows


def build_retry_rates_by_phase_table(phase_records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in phase_records:
        phase_name = record.get("phase_name")
        if isinstance(phase_name, str) and phase_name:
            grouped[phase_name].append(record)

    rows: list[dict[str, Any]] = []
    for phase_name, records in sorted(grouped.items()):
        total_retry_count = sum(_as_int(record.get("retry_count")) for record in records)
        runs_with_retries = {
            str(record.get("run_id"))
            for record in records
            if _as_int(record.get("retry_count")) > 0 and record.get("run_id")
        }
        failure_count = sum(1 for record in records if not record.get("success"))
        rows.append(
            {
                "phase_name": phase_name,
                "phase_occurrences": len(records),
                "run_count": len({str(record.get("run_id")) for record in records if record.get("run_id")}),
                "total_retry_count": total_retry_count,
                "average_retry_count": total_retry_count / len(records) if records else 0.0,
                "runs_with_retries": len(runs_with_retries),
                "failure_count": failure_count,
                "failure_rate": failure_count / len(records) if records else 0.0,
            }
        )
    return rows


def build_memory_profiles_by_model_table(
    inference_records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    run_model_map = infer_models_by_run(inference_records)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in inference_records:
        run_id = record.get("run_id")
        if isinstance(run_id, str) and run_id in run_model_map:
            model = run_model_map[run_id]
        else:
            model = infer_model([record])
        grouped[model].append(record)

    rows: list[dict[str, Any]] = []
    for model, records in sorted(grouped.items()):
        host_memory_samples = [
            _as_float(record.get("host_memory_mb"))
            for record in records
            if _has_number(record.get("host_memory_mb"))
        ]
        gpu_memory_samples = [
            _as_float(record.get("gpu_memory_mb"))
            for record in records
            if _has_number(record.get("gpu_memory_mb"))
        ]
        rows.append(
            {
                "model": model,
                "run_count": len({str(record.get("run_id")) for record in records if record.get("run_id")}),
                "sample_count": len(records),
                "average_host_memory_mb": _average_or_none(host_memory_samples),
                "peak_host_memory_mb": max(host_memory_samples) if host_memory_samples else None,
                "average_gpu_memory_mb": _average_or_none(gpu_memory_samples),
                "peak_gpu_memory_mb": max(gpu_memory_samples) if gpu_memory_samples else None,
            }
        )
    return rows


def build_generation_benchmarks_by_backend_table(
    inference_records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    run_backend_map = infer_backends_by_run(inference_records)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in inference_records:
        run_id = record.get("run_id")
        if isinstance(run_id, str) and run_id in run_backend_map:
            backend = run_backend_map[run_id]
        else:
            backend = infer_backend([record])
        grouped[backend].append(record)

    rows: list[dict[str, Any]] = []
    for backend, records in sorted(grouped.items()):
        prompt_rates = _collect_generation_rates(records, "prompt_tokens_per_second")
        output_rates = _collect_generation_rates(records, "output_tokens_per_second")
        total_rates = _collect_generation_rates(records, "total_tokens_per_second")
        latency_samples = [
            _as_float(record.get("latency_ms"))
            for record in records
            if _has_number(record.get("latency_ms"))
        ]
        rows.append(
            {
                "backend": backend,
                "run_count": len({str(record.get("run_id")) for record in records if record.get("run_id")}),
                "inference_call_count": len(records),
                "generation_count": len(total_rates),
                "success_rate": (
                    sum(1 for record in records if record.get("success")) / len(records)
                    if records
                    else 0.0
                ),
                "average_latency_ms": _average_or_zero(latency_samples),
                "median_latency_ms": _median_or_none(latency_samples),
                "p95_latency_ms": _percentile_or_none(latency_samples, 95.0),
                "average_prompt_tokens_per_second": _average_or_zero(prompt_rates),
                "median_prompt_tokens_per_second": _median_or_none(prompt_rates),
                "peak_prompt_tokens_per_second": max(prompt_rates) if prompt_rates else None,
                "average_output_tokens_per_second": _average_or_zero(output_rates),
                "median_output_tokens_per_second": _median_or_none(output_rates),
                "peak_output_tokens_per_second": max(output_rates) if output_rates else None,
                "average_total_tokens_per_second": _average_or_zero(total_rates),
                "median_total_tokens_per_second": _median_or_none(total_rates),
                "peak_total_tokens_per_second": max(total_rates) if total_rates else None,
            }
        )
    return rows


def infer_backend(inference_records: Sequence[dict[str, Any]]) -> str:
    candidates = [
        str(record.get("backend"))
        for record in inference_records
        if record.get("backend") not in (None, "")
    ]
    if not candidates:
        return "unknown"
    return Counter(candidates).most_common(1)[0][0]


def infer_model(inference_records: Sequence[dict[str, Any]]) -> str:
    candidates = [
        str(record.get("model") or _usage(record).get("model"))
        for record in inference_records
        if (record.get("model") or _usage(record).get("model")) not in (None, "")
    ]
    if not candidates:
        return "unknown"
    return Counter(candidates).most_common(1)[0][0]


def infer_backends_by_run(
    inference_records: Sequence[dict[str, Any]],
) -> dict[str, str]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in inference_records:
        run_id = record.get("run_id")
        if isinstance(run_id, str) and run_id:
            grouped[run_id].append(record)

    return {
        run_id: infer_backend(records)
        for run_id, records in grouped.items()
    }


def infer_models_by_run(
    inference_records: Sequence[dict[str, Any]],
) -> dict[str, str]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in inference_records:
        run_id = record.get("run_id")
        if isinstance(run_id, str) and run_id:
            grouped[run_id].append(record)

    return {
        run_id: infer_model(records)
        for run_id, records in grouped.items()
    }


def render_text_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("Metrics analysis")
    lines.append(f"Runs analyzed: {report['run_count']}")
    lines.append(f"Metrics files: {report['metric_file_count']}")
    lines.append("")
    lines.append("Runs")
    for run in report["runs"]:
        lines.append(
            " - "
            f"{run['run_id']} | backend={run['backend']} | model={run['model']} | commit={run['commit_id']} | "
            f"inference_calls={run['inference_calls']} | avg_latency_ms={run['average_inference_latency_ms']:.2f} | "
            f"avg_total_tok_s={run['average_total_tokens_per_second']:.2f}"
        )
    lines.append("")
    lines.append("Latency by model")
    for row in report["comparison_tables"]["latency_by_model"]:
        lines.append(
            " - "
            f"{row['model']} | runs={row['run_count']} | total_inference_calls={row['total_inference_calls']} | "
            f"avg_latency_ms={row['average_latency_ms']:.2f} | avg_output_tokens_per_second={row['average_output_tokens_per_second']:.2f}"
        )
    lines.append("")
    lines.append("Generation benchmarks by backend")
    for row in report["comparison_tables"]["generation_benchmarks_by_backend"]:
        lines.append(
            " - "
            f"{row['backend']} | runs={row['run_count']} | generations={row['generation_count']} | "
            f"avg_output_tok_s={row['average_output_tokens_per_second']:.2f} | "
            f"avg_total_tok_s={row['average_total_tokens_per_second']:.2f} | "
            f"median_total_tok_s={_format_optional_float(row['median_total_tokens_per_second'])} | "
            f"p95_latency_ms={_format_optional_float(row['p95_latency_ms'])}"
        )
    lines.append("")
    lines.append("Retry rates by phase")
    for row in report["comparison_tables"]["retry_rates_by_phase"]:
        lines.append(
            " - "
            f"{row['phase_name']} | occurrences={row['phase_occurrences']} | total_retry_count={row['total_retry_count']} | "
            f"avg_retry_count={row['average_retry_count']:.2f} | failure_rate={row['failure_rate']:.2%}"
        )
    lines.append("")
    lines.append("Memory profiles by model")
    for row in report["comparison_tables"]["memory_profiles_by_model"]:
        lines.append(
            " - "
            f"{row['model']} | avg_host_memory_mb={_format_optional_float(row['average_host_memory_mb'])} | "
            f"peak_host_memory_mb={_format_optional_float(row['peak_host_memory_mb'])} | "
            f"avg_gpu_memory_mb={_format_optional_float(row['average_gpu_memory_mb'])} | "
            f"peak_gpu_memory_mb={_format_optional_float(row['peak_gpu_memory_mb'])}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--output")
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args(argv)

    report = analyze_paths(args.paths)
    if args.format == "json":
        rendered = json.dumps(report, indent=args.indent, sort_keys=True)
    else:
        rendered = render_text_report(report)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + ("\n" if not rendered.endswith("\n") else ""), encoding="utf-8")
    else:
        print(rendered)

    return 0


def _is_run_log_file(path: Path) -> bool:
    return path.name == "LAST_RUN_LOG.json" or (
        path.suffix == ".json" and "_full_run_" in path.name
    )


def _usage(record: dict[str, Any]) -> dict[str, Any]:
    usage = record.get("usage")
    if isinstance(usage, dict):
        return usage
    return {}


def _collect_generation_rates(
    records: Sequence[dict[str, Any]],
    rate_key: str,
) -> list[float]:
    rates: list[float] = []
    for record in records:
        rate = _generation_rate(record, rate_key)
        if rate is not None:
            rates.append(rate)
    return rates


def _generation_rate(record: dict[str, Any], rate_key: str) -> float | None:
    precomputed = record.get(rate_key)
    if _has_number(precomputed):
        return float(precomputed)

    usage = _usage(record)
    latency_ms = _as_float(record.get("latency_ms"))
    if latency_ms <= 0:
        return None

    latency_seconds = latency_ms / 1000
    prompt_tokens = _optional_int(usage.get("prompt_tokens"))
    completion_tokens = _optional_int(usage.get("completion_tokens"))
    total_tokens = _optional_int(usage.get("total_tokens"))
    if total_tokens is None and (
        prompt_tokens is not None or completion_tokens is not None
    ):
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

    if rate_key == "prompt_tokens_per_second" and prompt_tokens is not None:
        return float(prompt_tokens) / latency_seconds
    if rate_key == "output_tokens_per_second" and completion_tokens is not None:
        return float(completion_tokens) / latency_seconds
    if rate_key == "total_tokens_per_second" and total_tokens is not None:
        return float(total_tokens) / latency_seconds
    return None


def _usage_total_tokens(usage: dict[str, Any]) -> int:
    total_tokens = usage.get("total_tokens")
    if _has_number(total_tokens):
        return _as_int(total_tokens)
    return _as_int(usage.get("prompt_tokens")) + _as_int(usage.get("completion_tokens"))


def _average_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _average_or_zero(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _median_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(median(values))


def _percentile_or_none(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])

    sorted_values = sorted(values)
    percentile = min(max(percentile, 0.0), 100.0)
    rank = (percentile / 100) * (len(sorted_values) - 1)
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    fraction = rank - lower_index
    return float(lower_value + (upper_value - lower_value) * fraction)


def _has_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_int(value: Any) -> int:
    if _has_number(value):
        return int(value)
    return 0


def _optional_int(value: Any) -> int | None:
    if _has_number(value):
        return int(value)
    return None


def _as_float(value: Any) -> float:
    if _has_number(value):
        return float(value)
    return 0.0


def _format_optional_float(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
