from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


EXIT_SUCCESS = 0
EXIT_USAGE = 2
EXIT_DATASET = 3
EXIT_FILTER = 4
EXIT_OUTPUT = 5
EXIT_PLUGIN = 10


HELP_ENTRIES = {
    "open": "/view open <path> [as <key>] - load a NetCDF file and register a dataset key",
    "list_vars": "/view list_vars [<key>] - list coordinates and variables for the dataset",
    "preview": "/view preview <key> [as <subset>] [--cols --filter --bbox --start --end --order] - tabular preview with optional subset registration",
    "plot": "/view plot <key> <timeseries|profile|map> [--x --y --bbox --start --end --order --cmap --out] - render matplotlib charts (maps default to LONGITUDE/LATITUDE axes)",
    "export": "/view export <key> csv [--cols ... --filter ... --out ...] - stream a filtered CSV export",
    "close": "/view close [<key>] - close and release a dataset from memory",
    "help": "/view help [command] - show available /view commands and usage",
}


@dataclass
class ParsedCommand:
    request_type: str
    request_payload: Dict[str, Any]
    out_path: Optional[Path] = None
    echo_json: bool = False
    description: str = ""


def split_commands(raw: str) -> List[str]:
    commands: List[str] = []
    buf: List[str] = []
    quote: Optional[str] = None
    escape = False
    for ch in raw:
        if escape:
            buf.append(ch)
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch in {'"', "'"}:
            if quote is None:
                quote = ch
            elif quote == ch:
                quote = None
            buf.append(ch)
            continue
        if ch == ";" and quote is None:
            command = "".join(buf).strip()
            if command:
                commands.append(command)
            buf.clear()
            continue
        buf.append(ch)
    if buf:
        command = "".join(buf).strip()
        if command:
            commands.append(command)
    return commands


def parse_order(value: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for piece in value.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if ":" in piece:
            col, direction = piece.split(":", 1)
        else:
            col, direction = piece, "asc"
        entries.append({"col": col, "dir": direction})
    return entries


def parse_size(value: str) -> Tuple[int, int]:
    if "x" not in value:
        raise ValueError("size must look like WIDTHxHEIGHT")
    width, height = value.lower().split("x", 1)
    return int(width), int(height)


def parse_bins(value: str) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for pair in value.split(","):
        if "=" not in pair:
            continue
        key, val = pair.split("=", 1)
        result[key.strip()] = float(val)
    return result


def parse_slash_command(raw: str, fallback_dataset: Optional[str]) -> ParsedCommand:
    tokens = shlex.split(raw)
    if not tokens or not tokens[0].startswith("/"):
        raise ValueError("Slash commands must start with '/'")
    if tokens[0] != "/view":
        raise ValueError("Unsupported slash namespace; only /view is implemented")
    if len(tokens) == 1:
        raise ValueError("Missing /view subcommand")
    sub = tokens[1]
    idx = 2

    def take() -> str:
        nonlocal idx
        if idx >= len(tokens):
            raise ValueError("Missing argument")
        token = tokens[idx]
        idx += 1
        return token

    def take_optional(default: Optional[str] = None) -> Optional[str]:
        nonlocal idx
        if idx >= len(tokens):
            return default
        token = tokens[idx]
        idx += 1
        return token

    if sub == "help":
        topic = take_optional()
        if idx < len(tokens):
            raise ValueError("Unexpected arguments after help topic")
        payload = {"topic": topic}
        description = "Show /view help"
        return ParsedCommand("view.help", payload, description=description)

    options: Dict[str, Any] = {}
    out_path: Optional[Path] = None
    echo_json = False

    def parse_options() -> None:
        nonlocal idx, out_path, echo_json
        while idx < len(tokens):
            token = tokens[idx]
            if not token.startswith("--"):
                raise ValueError(f"Unexpected token '{token}'")
            idx += 1
            name = token[2:]
            if name in {
                "filter",
                "json-filter",
                "order",
                "cols",
                "cursor",
                "size",
                "dpi",
                "title",
                "x",
                "y",
                "z",
                "limit",
                "out",
                "bins",
                "filename",
                "bbox",
                "box",
                "start",
                "end",
                "cmap",
                "order",
            }:
                value = take()
                if name == "filter":
                    options.setdefault("filter", {})["dsl"] = value
                elif name == "json-filter":
                    options.setdefault("filter", {})["json"] = json.loads(value)
                elif name == "order":
                    options["orderBy"] = parse_order(value)
                elif name == "cols":
                    options["columns"] = [col.strip() for col in value.split(",") if col.strip()]
                elif name == "cursor":
                    options["cursor"] = value
                elif name == "size":
                    width, height = parse_size(value)
                    options.setdefault("style", {})["width"] = width
                    options.setdefault("style", {})["height"] = height
                elif name == "dpi":
                    options.setdefault("style", {})["dpi"] = int(value)
                elif name == "title":
                    options.setdefault("style", {})["title"] = value
                elif name == "x":
                    options["x"] = value
                elif name == "y":
                    options["y"] = value
                elif name == "z":
                    options["z"] = value
                elif name == "limit":
                    options["limit"] = int(value)
                elif name == "out":
                    out_path = Path(value)
                elif name == "bins":
                    options.setdefault("style", {})["bins"] = parse_bins(value)
                elif name == "filename":
                    options["filename"] = value
                elif name in {"bbox", "box"}:
                    parts = [piece.strip() for piece in value.split(",") if piece.strip()]
                    if len(parts) != 4:
                        raise ValueError("--bbox/--box expects four comma-separated values")
                    try:
                        bbox_vals = [float(piece) for piece in parts]
                    except ValueError as exc:
                        raise ValueError("--bbox/--box values must be numeric") from exc
                    options["bbox"] = bbox_vals
                elif name == "start":
                    options["start"] = value
                elif name == "end":
                    options["end"] = value
                elif name == "cmap":
                    options.setdefault("style", {})["cmap"] = value
            elif name in {"invert-y", "no-invert-y", "grid", "echo-json"}:
                if name == "invert-y":
                    options.setdefault("style", {})["invert_y"] = True
                elif name == "no-invert-y":
                    options.setdefault("style", {})["invert_y"] = False
                elif name == "grid":
                    options.setdefault("style", {})["grid"] = True
                elif name == "echo-json":
                    echo_json = True
            else:
                raise ValueError(f"Unknown option --{name}")

    if sub == "open":
        path = take()
        dataset_key: Optional[str] = None
        if idx < len(tokens):
            maybe_as = tokens[idx]
            if maybe_as == "as":
                idx += 1
                dataset_key = take_optional()
        if not dataset_key:
            dataset_key = Path(path).stem or "dataset"
        parse_options()
        payload = {"path": path, "datasetKey": dataset_key}
        payload.update(options)
        description = f"Open dataset {path} as {dataset_key}"
        return ParsedCommand("view.open_dataset", payload, out_path=None, echo_json=echo_json, description=description)

    if sub == "list_vars":
        dataset_key = take_optional(fallback_dataset)
        if not dataset_key:
            raise ValueError("Dataset key required; open a dataset first")
        parse_options()
        payload = {"datasetKey": dataset_key}
        description = f"List vars for {dataset_key}"
        return ParsedCommand("view.list_vars", payload, out_path=None, echo_json=echo_json, description=description)

    if sub == "preview":
        dataset_key = take()
        subset_key: Optional[str] = None
        if idx < len(tokens) and tokens[idx] == "as":
            idx += 1
            subset_key = take()
        parse_options()
        payload = {"datasetKey": dataset_key}
        if subset_key:
            payload["subsetKey"] = subset_key
        payload.update(options)
        description = f"Preview {dataset_key}"
        return ParsedCommand("view.preview", payload, out_path=None, echo_json=echo_json, description=description)

    if sub == "plot":
        dataset_key = take()
        kind = take()
        parse_options()
        payload = {"datasetKey": dataset_key, "kind": kind}
        payload.update(options)
        description = f"Plot {kind} for {dataset_key}"
        return ParsedCommand("view.plot", payload, out_path=out_path, echo_json=echo_json, description=description)

    if sub == "export":
        dataset_key = take()
        fmt = take()
        if fmt.lower() != "csv":
            raise ValueError("Only CSV export is supported")
        parse_options()
        payload = {"datasetKey": dataset_key, "format": "csv"}
        payload.update(options)
        description = f"Export {dataset_key} to CSV"
        return ParsedCommand("view.export", payload, out_path=out_path, echo_json=echo_json, description=description)

    if sub == "close":
        dataset_key = take_optional(fallback_dataset)
        if not dataset_key:
            raise ValueError("Dataset key required")
        parse_options()
        payload = {"datasetKey": dataset_key}
        description = f"Close dataset {dataset_key}"
        return ParsedCommand("view.close_dataset", payload, out_path=None, echo_json=echo_json, description=description)

    raise ValueError(f"Unsupported /view subcommand '{sub}'")


def exit_code_for_error(code: Optional[str]) -> int:
    if not code:
        return EXIT_PLUGIN
    if code in {"DATASET_OPEN_FAIL", "DATASET_NOT_FOUND", "UNSUPPORTED_ENGINE"}:
        return EXIT_DATASET
    if code in {"FILTER_INVALID", "FILTER_UNSUPPORTED_OP", "COLUMN_UNKNOWN"}:
        return EXIT_FILTER
    if code in {"PLOT_FAIL", "EXPORT_FAIL"}:
        return EXIT_OUTPUT
    if code in {"PLUGIN_NOT_AVAILABLE", "INTERNAL_ERROR"}:
        return EXIT_PLUGIN
    return EXIT_PLUGIN


__all__ = [
    "HELP_ENTRIES",
    "ParsedCommand",
    "exit_code_for_error",
    "parse_slash_command",
    "split_commands",
]
