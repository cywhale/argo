import json
import sys
import traceback
import threading
import io
import math
import hashlib
import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterable, Union
import os
os.environ.setdefault("XARRAY_DISABLE_PLUGIN_AUTOLOADING", "1")
warnings.filterwarnings(
    "ignore",
    message="Engine 'argo' loading failed",
    category=RuntimeWarning,
)
ODBARGO_DEBUG = os.getenv("ODBARGO_DEBUG", "0") not in ("", "0", "false", "False")

import xarray as xr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from matplotlib import ticker as mticker

try:  # pandas is optional but required for preview/export convenience
    import pandas as pd
except ImportError:  # pragma: no cover - runtime guard
    pd = None


PLUGIN_PROTOCOL_VERSION = "1.0"
DEFAULT_PREVIEW_LIMIT = 500
MAX_PREVIEW_ROWS = 1000
MAX_PREVIEW_COLUMNS = 16
MAX_FILTER_LENGTH = 2048

class PluginError(Exception):
    def __init__(self, code: str, message: str, *, hint: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint
        self.details = details or {}


@dataclass
class FilterSpec:
    dsl: Optional[str] = None
    json_spec: Optional[Dict[str, Any]] = None


@dataclass
class SubsetSpec:
    parent_key: str
    columns: Optional[List[str]]
    filters: List[FilterSpec]


@dataclass
class ResolvedSubset:
    columns: Optional[List[str]]
    filters: List[FilterSpec]


class DatasetStore:
    def __init__(self) -> None:
        self._datasets: Dict[str, xr.Dataset] = {}
        self._subsets: Dict[str, SubsetSpec] = {}

    def open_dataset(self, key: str, path: str, engine_preference: Optional[List[str]] = None) -> xr.Dataset:
        if key in self._datasets:
            self._datasets[key].close()
        engines = engine_preference or ["h5netcdf", "netcdf4"]
        last_err: Optional[Exception] = None
        for engine in engines:
            try:
                ds = xr.open_dataset(path, engine=engine)
                self._datasets[key] = ds
                return ds
            except Exception as exc:  # pragma: no cover - engine fallback
                last_err = exc
                continue
        # fallback to default engine
        try:
            ds = xr.open_dataset(path)
            self._datasets[key] = ds
            return ds
        except Exception as exc:
            raise PluginError("DATASET_OPEN_FAIL", f"Failed to open dataset at {path}", hint=str(last_err or exc)) from exc

    def get(self, key: str) -> xr.Dataset:
        if key in self._datasets:
            return self._datasets[key]
        if key in self._subsets:
            subset = self._subsets[key]
            return self.get(subset.parent_key)
        raise PluginError("DATASET_NOT_FOUND", f"Dataset '{key}' not found")

    def root_key(self, key: str) -> str:
        if key in self._datasets:
            return key
        if key in self._subsets:
            return self.root_key(self._subsets[key].parent_key)
        raise PluginError("DATASET_NOT_FOUND", f"Dataset '{key}' not found")

    def resolve(self, key: str) -> Tuple[xr.Dataset, Optional[ResolvedSubset]]:
        if key in self._datasets:
            return self._datasets[key], None
        if key in self._subsets:
            spec = self._subsets[key]
            dataset, parent = self.resolve(spec.parent_key)
            filters: List[FilterSpec] = []
            columns: Optional[List[str]] = None
            if parent:
                filters.extend(parent.filters)
                columns = parent.columns
            filters.extend(spec.filters)
            if spec.columns is not None:
                if columns is not None:
                    columns = [col for col in spec.columns if col in columns]
                else:
                    columns = list(spec.columns)
            return dataset, ResolvedSubset(columns=columns, filters=filters)
        raise PluginError("DATASET_NOT_FOUND", f"Dataset '{key}' not found")

    def register_subset(self, subset_key: str, parent_key: str, columns: Optional[List[str]], filters: List[FilterSpec]) -> None:
        if subset_key in self._datasets or subset_key in self._subsets:
            raise PluginError("DATASET_OPEN_FAIL", f"Dataset key '{subset_key}' already exists")
        filtered_filters = [f for f in filters if f.dsl or f.json_spec]
        if not filtered_filters and columns is None:
            raise PluginError("INTERNAL_ERROR", "Subset requires filters or columns")
        root = self.root_key(parent_key)
        stored_columns = list(columns) if columns is not None else None
        self._subsets[subset_key] = SubsetSpec(parent_key=root, columns=stored_columns, filters=filtered_filters)

    def close(self, key: str, *, async_close: bool = False) -> None:
        ds = self._datasets.pop(key, None)
        # remove any subsets referencing this key directly
        to_remove = [name for name, subset in self._subsets.items() if subset.parent_key == key or name == key]
        for name in to_remove:
            self._subsets.pop(name, None)
        if ds is not None:
            def _close_dataset() -> None:
                try:
                    ds.close()
                except Exception as exc:  # pragma: no cover - defensive
                    if ODBARGO_DEBUG:
                        print(f"[plugin] dataset close error for {key}: {exc}", file=sys.stderr)
            if async_close:
                threading.Thread(target=_close_dataset, name=f"close-ds-{key}", daemon=True).start()
            else:
                _close_dataset()

    def close_all(self) -> None:  # pragma: no cover - safety
        for key in list(self._datasets.keys()):
            self.close(key)


Token = Tuple[str, Any]


class DSLTokenizer:
    def __init__(self, text: str) -> None:
        self.text = text
        self.length = len(text)
        self.pos = 0

    def _peek(self) -> str:
        return self.text[self.pos] if self.pos < self.length else ""

    def _consume(self) -> str:
        ch = self._peek()
        self.pos += 1
        return ch

    def _consume_while(self, condition) -> str:
        start = self.pos
        while self.pos < self.length and condition(self.text[self.pos]):
            self.pos += 1
        return self.text[start:self.pos]

    def tokens(self) -> Iterable[Token]:
        while self.pos < self.length:
            ch = self._peek()
            if ch.isspace():
                self.pos += 1
                continue
            if ch in "(),":
                self.pos += 1
                yield (ch, ch)
                continue
            if ch in "<>!=":
                self.pos += 1
                if self._peek() == "=":
                    op = ch + self._consume()
                else:
                    op = ch
                yield ("OP", op)
                continue
            if ch in "'\"":
                quote = self._consume()
                buf = []
                while self.pos < self.length:
                    c = self._consume()
                    if c == quote:
                        break
                    if c == "\\" and self._peek() == quote:
                        c = self._consume()
                    buf.append(c)
                yield ("STRING", "".join(buf))
                continue
            if ch.isdigit() or (ch == "-" and self.pos + 1 < self.length and self.text[self.pos + 1].isdigit()):
                num = self._consume_while(lambda c: c.isdigit() or c in ".eE+-")
                yield ("NUMBER", num)
                continue
            if ch.isalpha() or ch == "_":
                ident = self._consume_while(lambda c: c.isalnum() or c == "_")
                upper = ident.upper()
                if upper in {"AND", "OR", "BETWEEN", "TRUE", "FALSE", "NULL"}:
                    yield (upper, upper)
                else:
                    yield ("IDENT", ident)
                continue
            raise PluginError("FILTER_INVALID", f"Unexpected character '{ch}' in filter", details={"position": self.pos})
        yield ("EOF", None)


class DSLParser:
    def __init__(self, text: str) -> None:
        if len(text) > MAX_FILTER_LENGTH:
            raise PluginError("FILTER_INVALID", "Filter too long")
        self.tokens = list(DSLTokenizer(text).tokens())
        self.pos = 0
        self.identifiers: List[str] = []

    def _peek(self) -> Token:
        return self.tokens[self.pos]

    def _advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _expect(self, kind: str) -> Token:
        tok = self._advance()
        if tok[0] != kind:
            raise PluginError("FILTER_INVALID", f"Expected {kind}, got {tok[0]}")
        return tok

    def parse(self) -> Tuple[Dict[str, Any], List[str]]:
        expr = self._parse_or()
        if self._peek()[0] != "EOF":
            raise PluginError("FILTER_INVALID", "Unexpected tokens at end of filter")
        return expr, self.identifiers

    def _parse_or(self):
        node = self._parse_and()
        while self._peek()[0] == "OR":
            self._advance()
            right = self._parse_and()
            node = {"op": "or", "children": [node, right]}
        return node

    def _parse_and(self):
        node = self._parse_factor()
        while self._peek()[0] == "AND":
            self._advance()
            right = self._parse_factor()
            node = {"op": "and", "children": [node, right]}
        return node

    def _parse_factor(self):
        tok = self._peek()
        if tok[0] == "(":
            self._advance()
            node = self._parse_or()
            self._expect(")")
            return node
        return self._parse_condition()

    def _parse_condition(self):
        ident = self._expect("IDENT")[1]
        self.identifiers.append(ident)
        tok = self._advance()
        if tok[0] == "BETWEEN":
            low = self._parse_value()
            if self._advance()[0] != "AND":
                raise PluginError("FILTER_INVALID", "Expected AND in BETWEEN expression")
            high = self._parse_value()
            return {"op": "between", "field": ident, "low": low, "high": high}
        if tok[0] != "OP":
            raise PluginError("FILTER_INVALID", f"Expected comparator after identifier '{ident}'")
        comparator = tok[1]
        value = self._parse_value()
        return {"op": "compare", "field": ident, "cmp": comparator, "value": value}

    def _parse_value(self):
        tok = self._advance()
        if tok[0] == "NUMBER":
            try:
                if any(c in tok[1] for c in ".eE"):
                    return float(tok[1])
                return int(tok[1])
            except ValueError:
                return float(tok[1])
        if tok[0] == "STRING":
            return tok[1]
        if tok[0] == "TRUE":
            return True
        if tok[0] == "FALSE":
            return False
        if tok[0] == "NULL":
            return None
        if tok[0] == "IDENT":
            # treat as bare identifier string literal
            return tok[1]
        raise PluginError("FILTER_INVALID", "Invalid literal in filter", details={"token": tok})


def _ensure_pandas() -> None:
    if pd is None:
        raise PluginError("INTERNAL_ERROR", "pandas is required for this operation", hint="Install extras: pip install 'odbargo-view[pandas]'")


def _infer_literal(value: Any, series: pd.Series) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str) and value == "":
        return value
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(value)
    if pd.api.types.is_numeric_dtype(series):
        try:
            return float(value)
        except Exception as exc:  # pragma: no cover - parse fallback
            raise PluginError("FILTER_INVALID", f"Expected numeric literal for {series.name}") from exc
    return value


def _evaluate_json_filter(filter_json: Dict[str, Any], df: pd.DataFrame) -> pd.Series:
    if not filter_json:
        return pd.Series([True] * len(df), index=df.index)

    def eval_node(node) -> pd.Series:
        if "and" in node:
            series_list = [eval_node(child) for child in node["and"]]
            result = series_list[0]
            for s in series_list[1:]:
                result = result & s
            return result
        if "or" in node:
            series_list = [eval_node(child) for child in node["or"]]
            result = series_list[0]
            for s in series_list[1:]:
                result = result | s
            return result
        if "between" in node:
            field, low, high = node["between"]
            series = _get_series(df, field)
            low = _infer_literal(low, series)
            high = _infer_literal(high, series)
            return series.between(low, high)
        # comparator nodes structure: {"op":[field, value]}
        for op, payload in node.items():
            if op in {">", ">=", "<", "<=", "=", "!="}:
                field, value = payload
                series = _get_series(df, field)
                value = _infer_literal(value, series)
                if op == ">":
                    return series > value
                if op == ">=":
                    return series >= value
                if op == "<":
                    return series < value
                if op == "<=":
                    return series <= value
                if op == "=":
                    return series == value
                if op == "!=":
                    return series != value
        raise PluginError("FILTER_INVALID", "Unsupported filter node", details={"node": node})

    return eval_node(filter_json).fillna(False)


def _get_series(df: pd.DataFrame, field: str) -> pd.Series:
    if field not in df.columns:
        raise PluginError("COLUMN_UNKNOWN", f"Unknown column '{field}'")
    return df[field]


def _evaluate_dsl_filter(dsl: str, df: pd.DataFrame) -> Tuple[pd.Series, List[str]]:
    parser = DSLParser(dsl)
    ast, identifiers = parser.parse()

    def eval_node(node) -> pd.Series:
        op = node["op"]
        if op == "and":
            left = eval_node(node["children"][0])
            for child in node["children"][1:]:
                left = left & eval_node(child)
            return left
        if op == "or":
            left = eval_node(node["children"][0])
            for child in node["children"][1:]:
                left = left | eval_node(child)
            return left
        if op == "between":
            series = _get_series(df, node["field"])
            low = _infer_literal(node["low"], series)
            high = _infer_literal(node["high"], series)
            return series.between(low, high)
        if op == "compare":
            series = _get_series(df, node["field"])
            value = _infer_literal(node["value"], series)
            cmp = node["cmp"]
            if cmp == ">":
                return series > value
            if cmp == ">=":
                return series >= value
            if cmp == "<":
                return series < value
            if cmp == "<=":
                return series <= value
            if cmp == "=":
                return series == value
            if cmp == "!=":
                return series != value
            raise PluginError("FILTER_UNSUPPORTED_OP", f"Unsupported comparator '{cmp}'")
        raise PluginError("FILTER_INVALID", f"Invalid filter node '{op}'")

    mask = eval_node(ast)
    return mask.fillna(False), identifiers


def _build_dataframe(ds: xr.Dataset) -> pd.DataFrame:
    _ensure_pandas()
    # Include coordinates as columns for easier filtering
    df = ds.reset_coords().to_dataframe().reset_index()
    return df


def _format_value(value: Any) -> Any:
    if isinstance(value, (np.generic,)):
        value = value.item()
    if pd is not None and isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.datetime64,)):
        return np.datetime_as_string(value, timezone="UTC")
    if isinstance(value, (np.floating, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if value is None:
        return None
    return value


class OdbArgoViewPlugin:
    def __init__(self) -> None:
        self.store = DatasetStore()

    def _debug(self, message: str, msg_id: str | None = None) -> None:
        if not ODBARGO_DEBUG:
            return
        try:
            obj = {"op": "debug", "message": str(message)}
            if msg_id:
                obj["msgId"] = msg_id
            self.send_json(obj)
        except Exception:
            pass

    def send_json(self, payload: Dict[str, Any]) -> None:
        sys.stdout.write(json.dumps(payload) + "\n")
        sys.stdout.flush()

    def send_error(self, msg_id: Optional[str], code: str, message: str, hint: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> None:
        payload = {
            "op": "error",
            "msgId": msg_id,
            "code": code,
            "message": message,
        }
        if hint:
            payload["hint"] = hint
        if details:
            payload["details"] = details
        self.send_json(payload)

    def handle(self, message: Dict[str, Any]) -> None:
        op = message.get("op")
        msg_id = message.get("msgId")
        try:
            if op == "open_dataset":
                self._handle_open_dataset(msg_id, message)
            elif op == "list_vars":
                self._handle_list_vars(msg_id, message)
            elif op == "preview":
                self._handle_preview(msg_id, message)
            elif op == "plot":
                self._handle_plot(msg_id, message)
            elif op == "export":
                self._handle_export(msg_id, message)
            elif op == "close_dataset":
                self._handle_close_dataset(msg_id, message)
            else:
                raise PluginError("INTERNAL_ERROR", f"Unsupported op '{op}'")
        except PluginError as exc:
            self.send_error(msg_id, exc.code, exc.message, exc.hint, exc.details)
        except Exception as exc:  # pragma: no cover - protective catch
            tb = traceback.format_exc()
            self.send_error(msg_id, "INTERNAL_ERROR", str(exc), details={"trace": tb})

    def _handle_open_dataset(self, msg_id: str, message: Dict[str, Any]) -> None:
        path = message.get("path")
        dataset_key = message.get("datasetKey")
        engine_preference = message.get("enginePreference")
        if not path or not dataset_key:
            raise PluginError("DATASET_OPEN_FAIL", "Missing path or datasetKey")
        ds = self.store.open_dataset(dataset_key, path, engine_preference)
        summary = self._build_dataset_summary(ds)
        self.send_json({
            "op": "open_dataset.ok",
            "msgId": msg_id,
            "datasetKey": dataset_key,
            "summary": summary,
        })

    def _build_dataset_summary(self, ds: xr.Dataset) -> Dict[str, Any]:
        dims = {name: int(size) for name, size in ds.sizes.items()}
        coords = []
        for name, coord in ds.coords.items():
            coords.append({
                "name": name,
                "dtype": str(coord.dtype),
                "size": int(coord.size),
                "attrs": {k: self._safe_attr(v) for k, v in coord.attrs.items()},
            })
        vars_meta = []
        for name, var in ds.data_vars.items():
            vars_meta.append({
                "name": name,
                "dtype": str(var.dtype),
                "shape": [int(dim) for dim in var.shape],
                "attrs": {k: self._safe_attr(v) for k, v in var.attrs.items()},
            })
        return {"dims": dims, "coords": coords, "vars": vars_meta}

    def _safe_attr(self, value: Any) -> Any:
        try:
            json.dumps(value)
            return value
        except (TypeError, OverflowError):
            return str(value)

    def _handle_list_vars(self, msg_id: str, message: Dict[str, Any]) -> None:
        dataset_key = message.get("datasetKey")
        ds, subset = self.store.resolve(dataset_key)
        summary = self._build_dataset_summary(ds)
        if subset and subset.columns is not None:
            allowed = set(subset.columns)
            summary["vars"] = [var for var in summary["vars"] if var["name"] in allowed]
            summary["coords"] = [coord for coord in summary["coords"] if coord["name"] in allowed]
        self.send_json({
            "op": "list_vars.ok",
            "msgId": msg_id,
            "datasetKey": dataset_key,
            "coords": summary["coords"],
            "vars": summary["vars"],
        })

    def _mask_from_filter_specs(self, filters: List[FilterSpec], df: pd.DataFrame) -> pd.Series:
        mask = pd.Series([True] * len(df), index=df.index)
        for filter_spec in filters:
            if not (filter_spec.dsl or filter_spec.json_spec):
                continue
            if filter_spec.dsl:
                dsl_mask, _ = _evaluate_dsl_filter(filter_spec.dsl, df)
                mask = mask & dsl_mask
            if filter_spec.json_spec:
                json_mask = _evaluate_json_filter(filter_spec.json_spec, df)
                mask = mask & json_mask
        return mask.fillna(False)

    def _parse_filter(self, message: Dict[str, Any], df: pd.DataFrame, base_filters: Optional[List[FilterSpec]] = None) -> Tuple[pd.Series, Optional[FilterSpec]]:
        filter_obj = message.get("filter") or {}
        mask = pd.Series([True] * len(df), index=df.index)
        if base_filters:
            mask = mask & self._mask_from_filter_specs(base_filters, df)
        json_nodes: List[Dict[str, Any]] = []
        stored_spec: Optional[FilterSpec] = None

        dsl_text = filter_obj.get("dsl") if filter_obj else None
        if dsl_text:
            dsl_mask, _ = _evaluate_dsl_filter(dsl_text, df)
            mask = mask & dsl_mask

        if filter_obj.get("json"):
            json_spec = deepcopy(filter_obj["json"])
            json_mask = _evaluate_json_filter(json_spec, df)
            mask = mask & json_mask
            json_nodes.append(json_spec)

        bbox = message.get("bbox")
        if bbox is None:
            bbox = message.get("box")
        if bbox is not None:
            if isinstance(bbox, str):
                bbox_values = [part.strip() for part in bbox.split(",") if part.strip()]
            else:
                bbox_values = list(bbox)
            if len(bbox_values) != 4:
                raise PluginError("FILTER_INVALID", "bbox requires four numeric values: x0,y0,x1,y1")
            try:
                x0, y0, x1, y1 = [float(value) for value in bbox_values]
            except (TypeError, ValueError) as exc:
                raise PluginError("FILTER_INVALID", "bbox values must be numeric") from exc
            lon_col = self._resolve_column(df, ["LONGITUDE", "longitude", "lon"], "LONGITUDE")
            lat_col = self._resolve_column(df, ["LATITUDE", "latitude", "lat"], "LATITUDE")
            lon_min, lon_max = sorted((x0, x1))
            lat_min, lat_max = sorted((y0, y1))
            bbox_json = {
                "and": [
                    {"between": [lon_col, lon_min, lon_max]},
                    {"between": [lat_col, lat_min, lat_max]},
                ]
            }
            json_nodes.append(bbox_json)
            mask = mask & _evaluate_json_filter(bbox_json, df)

        start = message.get("start")
        end = message.get("end")
        if start or end:
            time_col = self._resolve_column(df, ["TIME", "time"], "TIME")
            time_nodes: List[Dict[str, Any]] = []
            if start:
                try:
                    start_ts = pd.to_datetime(start)
                except Exception as exc:  # pragma: no cover - defensive
                    raise PluginError("FILTER_INVALID", f"Invalid start datetime '{start}'") from exc
                if pd.isna(start_ts):
                    raise PluginError("FILTER_INVALID", f"Invalid start datetime '{start}'")
                time_nodes.append({">=": [time_col, start_ts.isoformat()]})
            if end:
                try:
                    end_ts = pd.to_datetime(end)
                except Exception as exc:  # pragma: no cover - defensive
                    raise PluginError("FILTER_INVALID", f"Invalid end datetime '{end}'") from exc
                if pd.isna(end_ts):
                    raise PluginError("FILTER_INVALID", f"Invalid end datetime '{end}'")
                time_nodes.append({"<=": [time_col, end_ts.isoformat()]})
            if time_nodes:
                if len(time_nodes) == 1:
                    time_json: Dict[str, Any] = time_nodes[0]
                else:
                    time_json = {"and": time_nodes}
                json_nodes.append(time_json)
                mask = mask & _evaluate_json_filter(time_json, df)

        combined_json: Optional[Dict[str, Any]] = None
        if json_nodes:
            if len(json_nodes) == 1:
                combined_json = json_nodes[0]
            else:
                combined_json = {"and": json_nodes}

        if dsl_text or combined_json:
            stored_spec = FilterSpec(dsl=dsl_text, json_spec=combined_json)

        return mask.fillna(False), stored_spec

    def _collect_columns(self, message: Dict[str, Any], subset_filters: Optional[List[FilterSpec]] = None, allowed_columns: Optional[List[str]] = None) -> List[str]:
        columns = message.get("columns") or []
        extra_cols: List[str] = []
        filter_obj = message.get("filter") or {}
        if "json" in filter_obj:
            extra_cols.extend(self._extract_fields_from_json(filter_obj["json"]))
        if "dsl" in filter_obj and filter_obj["dsl"]:
            parser = DSLParser(filter_obj["dsl"])
            _, identifiers = parser.parse()
            extra_cols.extend(identifiers)
        bbox_present = message.get("bbox") if message.get("bbox") is not None else message.get("box")
        if bbox_present:
            extra_cols.extend(["LONGITUDE", "LATITUDE"])
        if message.get("start") or message.get("end"):
            extra_cols.append("TIME")
        if subset_filters:
            for filt in subset_filters:
                if filt.json_spec:
                    extra_cols.extend(self._extract_fields_from_json(filt.json_spec))
                if filt.dsl:
                    parser = DSLParser(filt.dsl)
                    _, identifiers = parser.parse()
                    extra_cols.extend(identifiers)
        if message.get("orderBy"):
            for entry in message["orderBy"]:
                extra_cols.append(entry.get("col"))
        result = []
        for col in list(columns) + extra_cols:
            if col and col not in result:
                result.append(col)
        if allowed_columns is not None:
            result = [col for col in result if col in allowed_columns]
        return result

    def _apply_order(self, df: pd.DataFrame, order_by: List[Dict[str, Any]], allowed_columns: Optional[List[str]] = None) -> pd.DataFrame:
        if not order_by:
            return df
        sort_cols: List[str] = []
        ascending: List[bool] = []
        for entry in order_by:
            col = entry.get("col")
            if not col:
                continue
            if allowed_columns is not None and col not in allowed_columns:
                raise PluginError("COLUMN_UNKNOWN", f"Column '{col}' not in subset")
            if col not in df.columns:
                raise PluginError("COLUMN_UNKNOWN", f"Unknown column '{col}'")
            direction = (entry.get("dir") or "asc").lower()
            sort_cols.append(col)
            ascending.append(direction != "desc")
        if not sort_cols:
            return df
        return df.sort_values(by=sort_cols, ascending=ascending, kind="mergesort")

    def _resolve_column(self, df: pd.DataFrame, candidates: Iterable[str], label: str) -> str:
        columns = list(df.columns)
        for name in candidates:
            if name in columns:
                return name
        lower_map = {col.lower(): col for col in columns}
        for name in candidates:
            match = lower_map.get(name.lower())
            if match:
                return match
        raise PluginError("FILTER_INVALID", f"Dataset missing required {label} column for filter")

    def _required_filter_columns(
        self,
        message: Dict[str, Any],
        df: pd.DataFrame,
        allowed_columns: Optional[List[str]],
    ) -> List[str]:
        required: List[str] = []
        bbox_present = message.get("bbox") if message.get("bbox") is not None else message.get("box")
        if bbox_present:
            lon_col = self._resolve_column(df, ["LONGITUDE", "longitude", "lon"], "LONGITUDE")
            lat_col = self._resolve_column(df, ["LATITUDE", "latitude", "lat"], "LATITUDE")
            required.extend([lon_col, lat_col])
        if message.get("start") or message.get("end"):
            time_col = self._resolve_column(df, ["TIME", "time"], "TIME")
            required.append(time_col)
        if allowed_columns is not None:
            disallowed = [col for col in required if col not in allowed_columns]
            if disallowed:
                raise PluginError("COLUMN_UNKNOWN", f"Column '{disallowed[0]}' not in subset")
        return required

    def _pick_reducer(self, agg: Optional[str]):
        import numpy as _np
        if not agg or agg.lower() == "mean":
            return _np.mean
        a = str(agg).lower()
        if a == "median":
            return _np.median
        if a == "max":
            return _np.max
        if a == "min":
            return _np.min
        if a == "count":
            return "count"  # sentinel handled at call site
        raise PluginError("PLOT_FAIL", f"Unsupported agg '{agg}'")

    def _apply_groupby_bins(self, df: "pd.DataFrame", spec: str) -> Tuple["pd.DataFrame", str, Optional[str]]:
        """
        Parse one groupBy spec and return (df_with_bincol, group_col_name, resample_freq_or_None).
        - "COL"        → group by that column
        - "COL:10.0"   → numeric binning width 10.0 (new column)
        - "TIME:1D"    → request resample on TIME with freq "1D"
        """
        import pandas as pd
        import numpy as np
        if ":" not in spec:
            col = spec.strip()
            if col not in df.columns:
                raise PluginError("COLUMN_UNKNOWN", f"Unknown groupBy column '{col}'")
            return df, col, None
        col, param = (s.strip() for s in spec.split(":", 1))
        if col.upper() == "TIME":
            return df, col, param  # tell caller to resample on time
        if col not in df.columns:
            raise PluginError("COLUMN_UNKNOWN", f"Unknown groupBy column '{col}'")
        try:
            bw = float(param)
            if bw <= 0:
                raise ValueError
        except Exception:
            return df, col, None
        bin_col = f"__bin__{col}"
        df = df.copy()
        df[bin_col] = (np.floor(df[col].astype(float) / bw) * bw).astype(df[col].dtype, copy=False)
        return df, bin_col, None

    def _parse_group_by_spec(self, df: "pd.DataFrame", group_by: Optional[List[str]]):
        """
        Parse groupBy list. Supports:
        - bare column names:   ["WMO", "CYCLE_NUMBER"]
        - time resample:       ["TIME:1D"] uses pd.Grouper(key="TIME", freq="1D")
        - numeric binning:     ["LONGITUDE:1.0"] → floor to 1.0 degree bins
        Returns a tuple (group_keys, is_resample), where:
        - group_keys is a list of valid groupers (column names, pd.Grouper, or derived bin columns)
        - is_resample indicates if any time resampling was requested (affects plotting order).
        """

        if not group_by:
            return None, False

        import pandas as pd
        import numpy as np

        group_keys: List[Any] = []
        is_resample = False

        for item in group_by:
            if not item:
                continue
            spec = str(item)
            if ":" not in spec:
                # plain column
                col = spec
                if col not in df.columns:
                    raise PluginError("COLUMN_UNKNOWN", f"Unknown groupBy column '{col}'")
                group_keys.append(col)
                continue

            col, param = spec.split(":", 1)
            col = col.strip()
            param = param.strip()
            if col not in df.columns:
                raise PluginError("COLUMN_UNKNOWN", f"Unknown groupBy column '{col}'")

            # TIME:<freq> → pandas time resample
            if col.upper() == "TIME":
                is_resample = True
                try:
                    # ensure datetime
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                except Exception:
                    pass
                group_keys.append(pd.Grouper(key=col, freq=param))
                continue

            # NUMERIC:<binwidth> → discretize with floor(binwidth)
            try:
                bw = float(param)
                if bw <= 0:
                    raise ValueError
                bin_col = f"__bin__{col}"
                df[bin_col] = (np.floor(df[col].astype(float) / bw) * bw).astype(df[col].dtype, copy=False)
                group_keys.append(bin_col)
            except Exception:
                # fallback: treat as plain column value if parse failed
                group_keys.append(col)

        return group_keys, is_resample

    def _resolve_map_columns(
        self,
        df: pd.DataFrame,
        message: Dict[str, Any],
        allowed_columns: Optional[List[str]],
    ) -> Tuple[str, str, Optional[str]]:
        style = message.get("style") or {}
        lon_col = message.get("x") or style.get("lon") or "LONGITUDE"
        lat_col = style.get("lat") or "LATITUDE"
        value_col = message.get("z") or style.get("field") or style.get("fields")
        y_token = message.get("y")
        if not value_col and y_token and y_token not in {lon_col, lat_col}:
            value_col = y_token
        required = [lon_col, lat_col]
        if value_col:
            required.append(value_col)
        if allowed_columns is not None:
            missing = [col for col in required if col and col not in allowed_columns]
            if missing:
                raise PluginError("COLUMN_UNKNOWN", f"Column '{missing[0]}' not in subset")
        return lon_col, lat_col, value_col

    def _build_map_grid(
        self,
        df: pd.DataFrame,
        lon_col: str,
        lat_col: str,
        value_col: Optional[str],
        bins: Optional[Dict[str, Any]] = None,
        agg: Optional[str] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if not value_col or value_col not in df.columns or not isinstance(bins, dict):
            return None

        try:
            lon_step = float(bins.get("lon", 0.0))
            lat_step = float(bins.get("lat", 0.0))
        except Exception:
            return None
        if lon_step <= 0.0 or lat_step <= 0.0:
            return None

        b = df[[lon_col, lat_col, value_col]].dropna()
        if b.empty:
            return None

        import numpy as _np
        # floor-binning is stable for negative longitudes/latitudes
        b["__lon_bin__"] = (_np.floor(b[lon_col].to_numpy(dtype=float) / lon_step) * lon_step)
        b["__lat_bin__"] = (_np.floor(b[lat_col].to_numpy(dtype=float) / lat_step) * lat_step)

        agg_value = str(agg or bins.get("agg", "mean")).lower()
        if   agg_value in {"mean", "avg"}: reducer = "mean"
        elif agg_value in {"median", "med"}: reducer = "median"
        elif agg_value in {"min", "max", "sum", "count"}: reducer = agg_value
        else: reducer = "mean"

        pivot = b.pivot_table(
            index="__lat_bin__",   # Y
            columns="__lon_bin__", # X
            values=value_col,
            aggfunc=reducer,
        )
        if pivot.empty or pivot.shape[0] < 2 or pivot.shape[1] < 2:
            return None

        pivot = pivot.sort_index().sort_index(axis=1)
        try:
            lon_vals = pd.to_numeric(pivot.columns)
        except Exception:
            lon_vals = pivot.columns.to_numpy()
        try:
            lat_vals = pd.to_numeric(pivot.index)
        except Exception:
            lat_vals = pivot.index.to_numpy()

        lon_grid, lat_grid = _np.meshgrid(lon_vals, lat_vals)
        return lon_grid, lat_grid, pivot.to_numpy()

    def _normalize_plot_message(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        params = raw.get("params")
        if not isinstance(params, dict):
            return dict(raw)
        merged: Dict[str, Any] = dict(raw)
        merged.update(params)
        style: Dict[str, Any] = {}
        if isinstance(params.get("style"), dict):
            style.update(params["style"])
        if isinstance(raw.get("style"), dict):
            style.update(raw["style"])
        if style:
            merged["style"] = style
        merged.pop("params", None)
        group_by = merged.get("groupBy")
        if isinstance(group_by, str):
            merged["groupBy"] = [item.strip() for item in group_by.split(",") if item.strip()]
        return merged

    def _build_color_cycle(self, cmap_name: Optional[str], target_size: int) -> Optional[List[Tuple[float, ...]]]:
        if not cmap_name:
            return None
        try:
            cmap = plt.get_cmap(cmap_name)
        except Exception as exc:
            self._debug(f"color_cycle: invalid cmap '{cmap_name}' ({exc})", "-")
            return None
        try:
            total = max(1, int(target_size))
        except Exception:
            total = 1
        total = min(total, 64)
        if total == 1:
            positions = np.asarray([0.5])
        else:
            positions = np.linspace(0.02, 0.98, total)
        colors: List[Tuple[float, ...]] = []
        for pos in positions:
            rgba = tuple(np.asarray(cmap(pos)).flatten())
            colors.append(rgba)
        return colors if colors else None

    def _adjust_legend_margins(
        self,
        fig: plt.Figure,
        loc_key: str,
        orientation: str,
        rows: int,
    ) -> None:
        if orientation != "horizontal" or rows <= 0:
            return

        per_row_margin = 0.045
        extra = per_row_margin * rows
        params = fig.subplotpars

        if loc_key == "bottom":
            base = max(0.1, params.bottom)
            desired = min(0.5, base + extra)
            if desired < params.top:
                fig.subplots_adjust(bottom=desired)
        elif loc_key == "top":
            base_margin = max(0.06, 1.0 - params.top)
            desired_margin = min(0.5, base_margin + extra)
            new_top = max(params.bottom + 0.1, 1.0 - desired_margin)
            new_top = min(params.top, new_top)
            if new_top > params.bottom:
                fig.subplots_adjust(top=new_top)

    def _legend_params(
        self,
        style: Dict[str, Any],
        msg_id: Optional[str],
        series_count: int,
    ) -> Tuple[str, Dict[str, Any], str, str]:
        loc_raw = style.get("legend_loc")
        if not loc_raw:
            return "best", {"frameon": False}, "inside", "inside"
        norm = str(loc_raw).strip().lower().replace('_', '-').replace(' ', '-')
        outside = {
            "top": ("upper center", {"bbox_to_anchor": (0.5, 1.12), "borderaxespad": 0.3}, "horizontal"),
            "bottom": ("lower center", {"bbox_to_anchor": (0.5, -0.3), "borderaxespad": 0.3}, "horizontal"),
            "right": ("center left", {"bbox_to_anchor": (1.02, 0.5), "borderaxespad": 0.3}, "vertical"),
            "left": ("center right", {"bbox_to_anchor": (-0.02, 0.5), "borderaxespad": 0.3}, "vertical"),
        }
        if norm in outside:
            loc, extra, orientation = outside[norm]
            extra = dict(extra)
            extra.setdefault("frameon", False)
            return loc, extra, orientation, norm
        builtin = {
            "best": "best",
            "upper-right": "upper right",
            "upper-left": "upper left",
            "lower-left": "lower left",
            "lower-right": "lower right",
            "upper-center": "upper center",
            "lower-center": "lower center",
            "center-left": "center left",
            "center-right": "center right",
            "center": "center",
        }
        if norm in builtin:
            return builtin[norm], {"frameon": False}, "inside", norm
        self._debug(f"legend: unknown loc '{loc_raw}', using 'best'", msg_id or "-")
        return "best", {"frameon": False}, "inside", norm

    def _legend_layout(
        self,
        orientation: str,
        item_count: int,
    ) -> Tuple[Optional[int], int]:
        if item_count <= 0:
            return None, 0

        if orientation == "horizontal":
            max_per_row = 6
            per_row = max(1, min(item_count, max_per_row))
            rows = int(math.ceil(item_count / per_row))
            return per_row, rows

        max_per_column = 12
        if orientation == "vertical":
            columns = max(1, int(math.ceil(item_count / max_per_column)))
            rows = int(math.ceil(item_count / columns))
            return columns, rows

        if item_count > max_per_column:
            columns = max(1, int(math.ceil(item_count / max_per_column)))
            rows = int(math.ceil(item_count / columns))
            return columns, rows
        return None, 1

    def _sample_cmap(self, cmap_name: Optional[str], n: int):
        """
        Return a list of RGBA colors sampled from a named colormap.
        - Used for grouped timeseries/profile lines (categorical-ish palette).
        - Keeps imports local to avoid heavy global imports during module load.
        """
        import numpy as np
        import matplotlib.cm as cm

        if n <= 0:
            return []
        name = cmap_name or "tab20"  # pleasant categorical default
        try:
            cmap = cm.get_cmap(name)
        except Exception:
            cmap = cm.get_cmap("tab20")

        if n == 1:
            return [cmap(0.0)]
        xs = np.linspace(0.0, 1.0, num=n)
        return [cmap(x) for x in xs]


    def _format_time_axis(self, ax: plt.Axes) -> None:
        # Only runs for datetime axes; keep labels horizontal and compact.
        if pd is None:
            return

        samples: List[pd.Series] = []
        for line in ax.get_lines():
            data = line.get_xdata(orig=False)
            if data is None or len(data) == 0:
                continue
            arr = np.asarray(data)
            try:
                if np.issubdtype(arr.dtype, np.datetime64):
                    series = pd.to_datetime(arr, errors="coerce")
                else:
                    series = pd.to_datetime(mdates.num2date(arr), errors="coerce")
            except Exception:
                try:
                    series = pd.to_datetime(arr, errors="coerce")
                except Exception:
                    continue
            series = series.dropna()
            if not series.empty:
                samples.append(series if isinstance(series, pd.Series) else series.to_series(index=np.arange(len(series))))

        if not samples:
            return

        dt = pd.concat(samples)
        if dt.empty:
            return

        nums = mdates.date2num(dt.to_numpy())
        span_days = float(np.nanmax(nums) - np.nanmin(nums))
        if span_days <= 0:
            return

        ax.xaxis.get_offset_text().set_visible(False)
        ax.tick_params(axis="x", which="major", rotation=0)

        # Aim for ~8–10 ticks
        TARGET = 9

        if span_days <= 31:
            # show mm-dd, put the year on the first tick or when a month flips
            step = max(1, int(round(span_days / TARGET)))  # ~daily
            locator = mdates.DayLocator(interval=step)

            def _day_fmt(x, pos):
                d = mdates.num2date(x)
                lab = d.strftime("%m-%d")
                if pos == 0 or d.day == 1:
                    lab += f"\n{d.year}"
                return lab

            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(_day_fmt))
            return

        # Up to ~13 months → month ticks
        months = max(1, int(round(span_days / 30.4375)))
        if months <= 13:
            step = max(1, int(round(months / TARGET)))  # month interval
            locator = mdates.MonthLocator(interval=step)

            def _month_fmt(x, pos):
                d = mdates.num2date(x)
                lab = f"{d.month:02d}"
                if pos == 0 or d.month == 1:
                    lab += f"\n{d.year}"
                return lab

            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(_month_fmt))
            return

        # Multi-year → yearly ticks
        years = span_days / 365.25
        step = max(1, int(round(years / TARGET)))
        ax.xaxis.set_major_locator(mdates.YearLocator(base=step))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    def _valid_bins(self, bins: Any) -> bool:
        if not isinstance(bins, dict):
            return False
        try:
            lon_step = float(bins.get("lon", 0.0))
            lat_step = float(bins.get("lat", 0.0))
        except Exception:
            return False
        return lon_step > 0 and lat_step > 0

    def _extract_fields_from_json(self, node: Dict[str, Any]) -> List[str]:
        fields: List[str] = []
        if not node:
            return fields
        if "and" in node:
            for child in node["and"]:
                fields.extend(self._extract_fields_from_json(child))
            return fields
        if "or" in node:
            for child in node["or"]:
                fields.extend(self._extract_fields_from_json(child))
            return fields
        if "between" in node:
            field = node["between"][0]
            fields.append(field)
            return fields
        for key, value in node.items():
            if key in {">", ">=", "<", "<=", "=", "!="} and isinstance(value, (list, tuple)) and value:
                fields.append(value[0])
        return fields

    def _handle_preview(self, msg_id: str, message: Dict[str, Any]) -> None:
        """
        Preview a bounded table. Keeps coordinate/dimension variables by default
        when columns are narrowed, unless trimDimensions=True. Also preserves the
        last displayed column set across pagination (cursor) for the *effective*
        dataset key (subset or base).
        """
        try:
            dataset_key = message.get("datasetKey")
            subset_key = message.get("subsetKey")
            base_ds, subset_spec = self.store.resolve(dataset_key)
            allowed_columns = subset_spec.columns if (subset_spec and subset_spec.columns is not None) else None

            df = _build_dataframe(base_ds)

            # Collect requested columns using your existing helper
            explicit_columns = bool(message.get("columns"))
            columns = self._collect_columns(
                message,
                subset_filters=subset_spec.filters if subset_spec else None,
                allowed_columns=allowed_columns,
            )
            # normalize to list
            if columns is None:
                columns = []
            else:
                columns = list(columns)

            # Effective key for remembering pagination columns:
            # when "as <subsetKey>" is used, the user will continue paging with datasetKey=subsetKey.
            effective_key = subset_key or dataset_key  # <<< key fix

            # Pagination: if cursor is provided and user didn't re-specify columns, reuse last shown
            cursor = message.get("cursor")
            if cursor and not explicit_columns:
                prev_cols = getattr(self, "_preview_state", {}).get(effective_key)
                if prev_cols:
                    columns = list(prev_cols)

            # Keep coord/dimension vars when user narrows columns, unless explicitly trimmed
            trim_dims = bool(message.get("trimDimensions", False))
            if explicit_columns and not trim_dims:
                for cname in list(base_ds.coords.keys()):
                    if cname in df.columns and cname not in columns:
                        columns.append(cname)

            # Ensure any columns required by filter/order are present
            for extra_col in self._required_filter_columns(message, df, allowed_columns):
                if extra_col not in columns:
                    columns.append(extra_col)

            # Default columns when nothing was chosen yet
            if not columns:
                if allowed_columns is not None:
                    columns = list(allowed_columns)
                else:
                    columns = list(df.columns[:MAX_PREVIEW_COLUMNS])

            # Permit coords even if not in subset projection; still block truly unknown columns
            if allowed_columns is not None:
                coord_names = set(base_ds.coords.keys())
                disallowed = [col for col in columns if (col not in allowed_columns and col not in coord_names)]
                if disallowed:
                    raise PluginError("COLUMN_UNKNOWN", "Requested column not in subset")

            # Parse and clamp limit defensively (handle None/str)
            raw_limit = message.get("limit", DEFAULT_PREVIEW_LIMIT)
            try:
                limit = int(raw_limit) if raw_limit is not None else int(DEFAULT_PREVIEW_LIMIT)
            except Exception:
                limit = int(DEFAULT_PREVIEW_LIMIT)

            if limit > MAX_PREVIEW_ROWS:
                limit = MAX_PREVIEW_ROWS
            if limit > 10000:
                raise PluginError("ROW_LIMIT_EXCEEDED", "Preview limit too large")

            # Column count guard
            if columns and len(columns) > MAX_PREVIEW_COLUMNS:
                if explicit_columns:
                    raise PluginError("PREVIEW_TOO_LARGE", "Too many columns requested")
                columns = columns[:MAX_PREVIEW_COLUMNS]

            # Existence check against the actual dataframe
            available_columns = list(df.columns)
            for col in columns:
                if col not in available_columns:
                    raise PluginError("COLUMN_UNKNOWN", f"Unknown column '{col}'")

            # Build mask/order with subset filters merged
            order_by = message.get("orderBy") or []
            filter_mask, requested_spec = self._parse_filter(
                message, df, base_filters=subset_spec.filters if subset_spec else None
            )
            df = df[filter_mask]
            df = self._apply_order(df, order_by, allowed_columns)

            # Pagination window
            start = 0
            if cursor is not None:
                try:
                    start = max(0, int(cursor))
                except Exception:
                    start = 0
            end = start + min(limit, MAX_PREVIEW_ROWS)

            total_rows = len(df)
            limited_df = df.iloc[start:end]
            limit_hit = end < total_rows
            next_cursor = str(end) if limit_hit else None

            # Materialize rows
            rows = []
            for _, row in limited_df.iterrows():
                rows.append([_format_value(row.get(col)) for col in columns])

            # Remember columns we actually displayed — keyed by the *effective* dataset
            if not hasattr(self, "_preview_state"):
                self._preview_state = {}
            self._preview_state[effective_key] = list(columns)

            # Persist subset if requested
            if subset_key:
                filter_specs: List[FilterSpec] = []
                if subset_spec and subset_spec.filters:
                    filter_specs.extend(subset_spec.filters)
                if requested_spec and (requested_spec.dsl or requested_spec.json_spec):
                    filter_specs.append(requested_spec)

                # Persist subset projection:
                # - If user explicitly chose columns AND did NOT request trimDimensions,
                #   store user's columns + coords (so TIME/LON/LAT survive).
                # - If user explicitly chose columns AND trimDimensions=True, store exactly those.
                # - Else inherit existing subset columns or None.
                trim_dims = bool(message.get("trimDimensions", False))
                if explicit_columns:
                    user_cols = list(message.get("columns") or [])
                    if not trim_dims:
                        coord_names = [c for c in base_ds.coords.keys() if c in df.columns]
                        # keep order: user columns first, then coords not already included
                        seen = set(user_cols)
                        for c in coord_names:
                            if c not in seen:
                                user_cols.append(c)
                                seen.add(c)
                    stored_columns = user_cols if user_cols else None
                    if stored_columns and allowed_columns is not None:
                        # if narrowing an existing subset, intersect
                        stored_columns = [c for c in stored_columns if c in allowed_columns or c in base_ds.coords]
                else:
                    if subset_spec and subset_spec.columns is not None:
                        stored_columns = list(subset_spec.columns)
                    else:
                        stored_columns = None

                self.store.register_subset(subset_key, dataset_key, stored_columns, filter_specs)

            # Reply
            self.send_json({
                "op": "preview.ok",
                "msgId": msg_id,
                "datasetKey": dataset_key,
                "columns": columns,
                "rows": rows,
                "limitHit": limit_hit,
                "nextCursor": next_cursor,
                "subsetKey": subset_key,
            })

        except PluginError:
            # Re-raise known plugin errors unchanged
            raise
        except Exception as e:
            # Defensive: report concrete cause rather than generic INTERNAL_ERROR black-boxing it
            raise PluginError("INTERNAL_ERROR", f"Preview failed: {type(e).__name__}: {e}")

    def _render_plot(self, raw_message: Dict[str, Any], msg_id: Optional[str] = None) -> bytes:
        message = self._normalize_plot_message(raw_message)
        dataset_key = message.get("datasetKey")
        if not dataset_key:
            raise PluginError("DATASET_NOT_FOUND", "Missing datasetKey")
        kind = message.get("kind")
        if not kind:
            raise PluginError("PLOT_FAIL", "Missing plot kind")
        x_col = message.get("x")
        y_col = message.get("y")
        style = dict(message.get("style") or {})
        bins_param: Dict[str, float] = {}
        style_bins = style.get("bins")
        if isinstance(style_bins, dict):
            for key, value in style_bins.items():
                k = str(key).strip().lower()
                try:
                    bins_param[k] = float(value)
                except Exception:
                    raise PluginError("BAD_BINS", f"bins.{k} must be numeric")
        message_bins = message.get("bins")
        if isinstance(message_bins, dict):
            for key, value in message_bins.items():
                k = str(key).strip().lower()
                try:
                    bins_param[k] = float(value)
                except Exception:
                    raise PluginError("BAD_BINS", f"bins.{k} must be numeric")
        elif message_bins is not None and not isinstance(message_bins, dict):
            raise PluginError("BAD_BINS", "bins must be an object with numeric values")
        def _bool_opt(value, default=False):
            if value is None:
                return default
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return value != 0
            return str(value).lower() not in {"false", "0", "no", "off"}

        group_by = message.get("groupBy") or []
        agg = message.get("agg") or "mean"
        self._debug(f"plot recv: kind={kind} x={x_col} y={y_col} groupBy={group_by} agg={agg}", msg_id or "-")
        reducer = self._pick_reducer(agg)

        base_ds, subset_spec = self.store.resolve(dataset_key)
        df = _build_dataframe(base_ds)
        allowed_columns = subset_spec.columns if (subset_spec and subset_spec.columns is not None) else None

        filter_mask, _ = self._parse_filter(message, df, base_filters=subset_spec.filters if subset_spec else None)
        df = df[filter_mask]
        df = self._apply_order(df, message.get("orderBy") or [], allowed_columns=None)

        try:
            limit = int(message.get("limit") or 0)
        except Exception:
            limit = 0
        if limit > 0:
            df = df.head(limit)
        if df.empty:
            raise PluginError("PLOT_FAIL", "Filter returned no rows")
        self._debug(f"plot df: rows={len(df)} cols={list(df.columns)[:8]}...", msg_id or "-")

        width = style.get("width", 800)
        height = style.get("height", 600)
        dpi = style.get("dpi", 120)
        figsize = (width / dpi, height / dpi)

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        try:

            def _exists(col: Optional[str]) -> bool:
                return bool(col) and col in df.columns

            def plot_line(x_vals, y_vals, marker=None, label=None):
                kwargs = {
                    "marker": marker if marker is not None else style.get("marker", ""),
                    "linestyle": style.get("line", "-"),
                    "alpha": float(style.get("alpha", 1.0)),
                }
                if label is not None:
                    kwargs["label"] = str(label)
                ax.plot(x_vals, y_vals, **kwargs)

            legend_enabled = _bool_opt(style.get('legend'), True)
            legend_always = _bool_opt(style.get('legend_always'), False)
            try:
                max_series_allowed = max(1, int(style.get('max_series', 24)))
            except Exception:
                max_series_allowed = 24

            color_cycle = self._build_color_cycle(style.get("cmap"), max_series_allowed)

            if kind == "timeseries":
                if not (_exists(x_col) and _exists(y_col)):
                    raise PluginError("COLUMN_UNKNOWN", "timeseries plot requires existing x and y columns")
                try:
                    df[x_col] = pd.to_datetime(df[x_col], errors="coerce")
                except Exception:
                    pass
                df = df.sort_values(by=x_col)

                if color_cycle:
                    ax.set_prop_cycle(color=color_cycle)

                work = df
                resample_freq: Optional[str] = None
                group_cols: List[str] = []
                for spec in group_by:
                    work, gcol, freq = self._apply_groupby_bins(work, str(spec))
                    if freq:
                        resample_freq = freq
                    elif gcol:
                        group_cols.append(gcol)

                if resample_freq:
                    try:
                        work[x_col] = pd.to_datetime(work[x_col], errors="coerce")
                    except Exception:
                        pass
                    if group_cols:
                        plotted = 0
                        for keys, g in work.groupby(group_cols, dropna=False):
                            rs = g.set_index(x_col).resample(resample_freq)[y_col]
                            g2 = (rs.count().reset_index() if reducer == "count" else rs.apply(reducer).reset_index())
                            label = keys if isinstance(keys, (tuple, list)) else (keys,)
                            plot_line(g2[x_col], g2[y_col], label=label)
                            plotted += 1
                            if plotted >= max_series_allowed:
                                break
                        self._debug(f"timeseries grouped (resample={resample_freq}) series plotted={plotted}", msg_id or "-")
                    else:
                        rs = work.set_index(x_col).resample(resample_freq)[y_col]
                        g2 = (rs.count().reset_index() if reducer == "count" else rs.apply(reducer).reset_index())
                        plot_line(g2[x_col], g2[y_col])
                        self._debug("timeseries resample-only → single aggregated series", msg_id or "-")
                elif group_cols:
                    plotted = 0
                    for keys, g in work.groupby(group_cols, dropna=False):
                        if reducer == "count":
                            g2 = g.groupby(x_col, dropna=False)[y_col].count().reset_index()
                        else:
                            g2 = g.groupby(x_col, dropna=False)[y_col].apply(reducer).reset_index()
                        label = keys if isinstance(keys, (tuple, list)) else (keys,)
                        plot_line(g2[x_col], g2[y_col], label=label)
                        plotted += 1
                        if plotted >= max_series_allowed:
                            break
                    self._debug(f"timeseries grouped (discrete/bin) series plotted={plotted}", msg_id or "-")
                else:
                    plot_line(df[x_col], df[y_col])
                    self._debug("timeseries: grouping disabled → single series", msg_id or "-")

                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                self._format_time_axis(ax)

            elif kind == "profile":
                y_col = y_col or "PRES"
                if not x_col:
                    raise PluginError("PLOT_FAIL", "profile plot requires --x <var> (X vs PRES)")
                if not (_exists(x_col) and _exists(y_col)):
                    raise PluginError("COLUMN_UNKNOWN", "profile plot requires existing x and y (default PRES)")

                bins_y: Optional[float] = None
                if "y" in bins_param:
                    try:
                        bins_y = float(bins_param["y"])
                    except Exception:
                        raise PluginError("BAD_BINS", "bins.y must be a positive float")
                    if not np.isfinite(bins_y) or bins_y <= 0:
                        raise PluginError("BAD_BINS", "bins.y must be a positive float")

                # reducer from --agg
                # gather group-by tokens and resolve into real columns
                gb_tokens = message.get("groupBy") or []
                gb_tokens = [t for t in gb_tokens if t]

                df_gb = df  # we may add derived columns here
                gb_cols: List[str] = []          # actual DataFrame columns used to group
                gb_meta: List[Tuple[str, Optional[float]]] = []  # (base_name, bin_step or None)

                if color_cycle:
                    ax.set_prop_cycle(color=color_cycle)

                for tok in gb_tokens:
                    if ":" in tok:
                        base, step_str = tok.split(":", 1)
                        base = base.strip()
                        if base not in df_gb.columns:
                            raise PluginError("COLUMN_UNKNOWN", f"Unknown group-by column '{base}'")
                        try:
                            step = float(step_str)
                        except Exception:
                            raise PluginError("BAD_GROUPBY", f"Non-numeric bin step in '{tok}'")
                        if step <= 0:
                            raise PluginError("BAD_GROUPBY", f"Bin step must be > 0 in '{tok}'")
                        new_col = f"__gb_{base}_bin__"
                        vals = pd.to_numeric(df_gb[base], errors="coerce").to_numpy(dtype=float)
                        df_gb = df_gb.assign(**{new_col: np.floor(vals / step) * step})
                        gb_cols.append(new_col)
                        gb_meta.append((base, step))
                    else:
                        base = tok.strip()
                        if base not in df_gb.columns:
                            raise PluginError("COLUMN_UNKNOWN", f"Unknown group-by column '{base}'")
                        gb_cols.append(base)
                        gb_meta.append((base, None))

                # pretty legend label from group key
                def _label_from_key(key) -> str:
                    if not isinstance(key, tuple):
                        key = (key,)
                    parts = []
                    for (base, step), val in zip(gb_meta, key):
                        if step is not None:
                            parts.append(f"{base}:{val:.0f}")
                        else:
                            parts.append(f"{base}={val}")
                    return ", ".join(parts)

                # optional sort by depth
                try:
                    df_gb = df_gb.sort_values(by=y_col)
                except Exception:
                    pass

                # optional thin/markers/alpha (defaults conservative)
                lw = float(style.get("linewidth", 1.2))
                mk = style.get("marker", "")  # default no markers
                alpha = float(style.get("alpha", 0.95))

                # palette
                n_groups = len(df_gb.groupby(gb_cols)) if gb_cols else 1
                colors = self._sample_cmap(style.get("cmap"), n_groups)
                color_iter = iter(colors) if colors else None

                # collapse duplicates (or bins.y) at the same depth → one X per Y per group
                def _collapse(g: pd.DataFrame) -> pd.DataFrame:
                    frame = g
                    depth_field = y_col
                    bin_col = "__pres_bin__"
                    if bins_y is not None:
                        numeric = pd.to_numeric(frame[y_col], errors="coerce")
                        valid_mask = numeric.notna()
                        if not valid_mask.any():
                            return pd.DataFrame(columns=[y_col, x_col])
                        frame = frame.loc[valid_mask].copy()
                        numeric = numeric.loc[valid_mask]
                        frame[bin_col] = np.floor(numeric.to_numpy() / bins_y) * bins_y
                        depth_field = bin_col
                    if frame.empty:
                        return pd.DataFrame(columns=[y_col, x_col])
                    if reducer == "count":
                        grouped = frame.groupby(depth_field, dropna=False)[x_col].count()
                    else:
                        grouped = frame.groupby(depth_field, dropna=False)[x_col].apply(reducer)
                    result = grouped.reset_index()
                    if depth_field != y_col:
                        result = result.rename(columns={depth_field: y_col})
                    return result.sort_values(y_col)

                if gb_cols:
                    for i, (gk, gdf) in enumerate(df_gb.groupby(gb_cols)):
                        adf = _collapse(gdf)
                        if adf.empty:
                            continue
                        color = next(color_iter, None) if color_iter is not None else None
                        ax.plot(
                            adf[x_col].to_numpy(),
                            adf[y_col].to_numpy(),
                            color=color,
                            linewidth=lw,
                            marker=mk,
                            alpha=alpha,
                            label=_label_from_key(gk),
                        )
                    # legend placement you already handle elsewhere (bottom/outside)
                else:
                    adf = _collapse(df_gb)
                    if not adf.empty:
                        color = next(color_iter, None) if color_iter is not None else None
                        ax.plot(adf[x_col].to_numpy(), adf[y_col].to_numpy(),
                                linewidth=lw, marker=mk, alpha=alpha, color=color)

                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                if style.get("invert_y", True):
                    ax.invert_yaxis()
                if style.get("grid"):
                    ax.grid(True)

            elif kind == "map":
                lon_col, lat_col, value_col = self._resolve_map_columns(df, message, None)
                if lon_col not in df.columns or lat_col not in df.columns:
                    raise PluginError("PLOT_FAIL", "map plot requires longitude and latitude columns")
                cmap = style.get("cmap", "viridis")
                agg_param = message.get("agg") or (bins_param.get("agg") if isinstance(bins_param, dict) else None)
                use_grid = self._valid_bins(bins_param) and bool(value_col)
                grid = self._build_map_grid(df, lon_col, lat_col, value_col, bins_param, agg_param) if use_grid else None
                if use_grid and grid is not None:
                    lon_grid, lat_grid, value_grid = grid
                    self._debug(f"map: gridded shape={value_grid.shape}", msg_id or "-")
                    pcm = ax.pcolormesh(lon_grid, lat_grid, value_grid, shading="auto", cmap=cmap)
                    cbar = fig.colorbar(pcm, ax=ax)
                    if value_col:
                        cbar.set_label(value_col)
                else:
                    reason = "no/invalid bins" if not use_grid else "grid=None"
                    self._debug(f"map: fallback scatter ({reason})", msg_id or "-")
                    color_values = df[value_col] if value_col and value_col in df.columns else None
                    point_size = (
                        style.get("pointSize")
                        or message.get("pointSize")
                        or style.get("scatterSize")
                        or 36
                    )
                    scatter_kwargs = {
                        "x": df[lon_col],
                        "y": df[lat_col],
                        "s": float(point_size),
                        "alpha": style.get("alpha", 0.75),
                        "edgecolors": style.get("edgecolor", "none"),
                        "marker": style.get("marker", "o"),
                    }
                    if color_values is not None:
                        scatter_kwargs["c"] = color_values
                        scatter_kwargs["cmap"] = cmap
                    sc = ax.scatter(**scatter_kwargs)
                    if color_values is not None:
                        cbar = fig.colorbar(sc, ax=ax)
                        cbar.set_label(value_col)
                ax.set_xlabel(lon_col)
                ax.set_ylabel(lat_col)
                if style.get("grid"):
                    ax.set_aspect("equal", adjustable="datalim")
            else:
                raise PluginError("PLOT_FAIL", f"Unsupported plot kind '{kind}'")

            handles, labels = ax.get_legend_handles_labels()
            legend_pairs = [
                (handle, label)
                for handle, label in zip(handles, labels)
                if label and not str(label).startswith("_")
            ]
            series_count = len(legend_pairs)
            show_legend = (
                legend_enabled
                and kind in {"timeseries", "profile"}
                and series_count > 0
                and (series_count > 1 or legend_always)
            )
            if show_legend:
                loc, legend_extra, legend_orientation, legend_loc_key = self._legend_params(style, msg_id, series_count)
                legend_kwargs = dict(legend_extra)
                legend_size = style.get("legend_fontsize", "small")
                try:
                    legend_size = float(legend_size)
                except Exception:
                    pass

                legend_ncol, legend_rows = self._legend_layout(legend_orientation, series_count)
                if legend_ncol is not None:
                    legend_kwargs["ncol"] = legend_ncol
                else:
                    legend_rows = max(legend_rows, 1)

                if legend_orientation == "horizontal" and legend_rows > 1:
                    anchor = legend_kwargs.get("bbox_to_anchor")
                    if isinstance(anchor, tuple) and len(anchor) == 2:
                        x_anchor, y_anchor = anchor
                        offset = 0.08 * (legend_rows - 1)
                        if legend_loc_key == "top":
                            y_anchor += offset
                        elif legend_loc_key == "bottom":
                            y_anchor -= offset
                        legend_kwargs["bbox_to_anchor"] = (x_anchor, y_anchor)

                if legend_loc_key in {"top", "bottom"}:
                    self._adjust_legend_margins(fig, legend_loc_key, legend_orientation, legend_rows)

                legend_handles = [handle for handle, _ in legend_pairs]
                legend_labels = [str(label) for _, label in legend_pairs]

                try:
                    ax.legend(
                        legend_handles,
                        legend_labels,
                        loc=loc,
                        fontsize=legend_size,
                        **legend_kwargs,
                    )
                except Exception as exc:
                    self._debug(f"legend: fallback to 'best' ({exc})", msg_id or "-")
                    try:
                        ax.legend(
                            legend_handles,
                            legend_labels,
                            loc="best",
                            fontsize=legend_size,
                        )
                    except Exception:
                        pass

            if style.get("title"):
                ax.set_title(style["title"])
            if kind != "map" and style.get("grid"):
                ax.grid(True)
            if kind == "map" and style.get("grid"):
                ax.grid(True, linestyle=style.get("grid_linestyle", "--"), alpha=style.get("grid_alpha", 0.3))

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
            png_bytes = buf.getvalue()
        finally:
            plt.close(fig)

        return png_bytes

    def _handle_plot(self, msg_id: str, message: Dict[str, Any]) -> None:
        png_bytes = self._render_plot(message, msg_id)
        self.send_json({
            "op": "plot_blob",
            "msgId": msg_id,
            "contentType": "image/png",
            "size": len(png_bytes),
        })
        sys.stdout.flush()
        sys.stdout.buffer.write(png_bytes)
        sys.stdout.flush()
    def _handle_export(self, msg_id: str, message: Dict[str, Any]) -> None:
        dataset_key = message.get("datasetKey")
        export_format = message.get("format", "csv")
        if export_format != "csv":
            raise PluginError("EXPORT_FAIL", f"Unsupported export format '{export_format}'")

        columns = message.get("columns") or []
        base_ds, subset_spec = self.store.resolve(dataset_key)
        df = _build_dataframe(base_ds)
        allowed_columns = subset_spec.columns if subset_spec and subset_spec.columns is not None else None

        if allowed_columns is not None and not columns:
            columns = list(allowed_columns)
        if allowed_columns is not None and any(col not in allowed_columns for col in columns):
            raise PluginError("COLUMN_UNKNOWN", "Requested column not in subset")

        if columns:
            for col in columns:
                if col not in df.columns:
                    raise PluginError("COLUMN_UNKNOWN", f"Unknown column '{col}'")

        filter_mask, _ = self._parse_filter(message, df, base_filters=subset_spec.filters if subset_spec else None)
        df = df[filter_mask]

        df = self._apply_order(df, message.get("orderBy") or [], allowed_columns)

        # --- Enforce limit for export, after filter+order (consistent with preview/plot) ---
        limit = int(message.get("limit") or 0)            # <<< CHANGED
        if limit and limit > 0:                           # <<< CHANGED
            df = df.head(limit)                           # <<< CHANGED

        if columns:
            df = df[columns]

        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        data = buffer.getvalue().encode("utf-8")
        sha = hashlib.sha256(data).hexdigest()
        filename = message.get("filename") or f"{dataset_key}_export.csv"

        self.send_json({
            "op": "file_start",
            "msgId": msg_id,
            "contentType": "text/csv",
            "filename": filename,
        })
        sys.stdout.flush()
        chunk_size = 256 * 1024
        for start in range(0, len(data), chunk_size):
            chunk = data[start:start + chunk_size]
            self.send_json({
                "op": "file_chunk",
                "msgId": msg_id,
                "size": len(chunk),
            })
            sys.stdout.flush()
            sys.stdout.buffer.write(chunk)
            sys.stdout.flush()
        self.send_json({
            "op": "file_end",
            "msgId": msg_id,
            "sha256": sha,
            "size": len(data),
        })

    def _handle_close_dataset(self, msg_id: str, message: Dict[str, Any]) -> None:
        dataset_key = message.get("datasetKey")
        if not dataset_key:
            raise PluginError("DATASET_NOT_FOUND", "Missing datasetKey")
        self.store.close(dataset_key, async_close=True)
        self.send_json({
            "op": "close_dataset.ok",
            "msgId": msg_id,
            "datasetKey": dataset_key,
        })

def _compute_plot_bytes(plugin: OdbArgoViewPlugin, message: Dict[str, Any]) -> bytes:
    return plugin._render_plot(message, message.get("msgId"))

def _compute_export_bytes(plugin: OdbArgoViewPlugin, message: Dict[str, Any]) -> Tuple[str, bytes, str]:
    dataset_key = message.get("datasetKey")
    columns = message.get("columns") or []
    base_ds, subset_spec = plugin.store.resolve(dataset_key)
    df = _build_dataframe(base_ds)
    allowed_columns = subset_spec.columns if subset_spec and subset_spec.columns is not None else None
    if allowed_columns is not None and not columns:
        columns = list(allowed_columns)
    if allowed_columns is not None and any(col not in allowed_columns for col in columns):
        raise PluginError("COLUMN_UNKNOWN", "Requested column not in subset")
    if columns:
        for col in columns:
            if col not in df.columns:
                raise PluginError("COLUMN_UNKNOWN", f"Unknown column '{col}'")
    filter_mask, _ = plugin._parse_filter(message, df, base_filters=subset_spec.filters if subset_spec else None)
    df = df[filter_mask]
    df = plugin._apply_order(df, message.get("orderBy") or [], allowed_columns)
    if columns:
        df = df[columns]
    buffer = io.StringIO(); df.to_csv(buffer, index=False); data = buffer.getvalue().encode("utf-8")
    sha = hashlib.sha256(data).hexdigest()
    filename = message.get("filename") or f"{dataset_key}_export.csv"
    return filename, data, sha


async def run_ws_mode(ws_url: str, token: str) -> None:
    import asyncio, websockets
    plugin = OdbArgoViewPlugin()

    async with websockets.connect(
        ws_url,
        additional_headers={"X-Plugin": "odbargo-view", "Authorization": f"Bearer {token}"} if token else None,
        ping_interval=20,
        ping_timeout=20,
        close_timeout=5,
        max_size=None,
    ) as ws:
        # Register to CLI
        await ws.send(json.dumps({
            "type": "plugin.register",
            "pluginProtocolVersion": PLUGIN_PROTOCOL_VERSION,
            "capabilities": {
                "open_dataset": True,
                "list_vars": True,
                "preview": True,
                "plot": True,
                "export": True,
                "subset": True,
            },
            "token": token if token else "",
        }))
        # Optionally wait for plugin.register_ok:
        # ack = json.loads(await ws.recv())

        # Process messages forever
        while True:
            msg = await ws.recv()
            if isinstance(msg, (bytes, bytearray)):
                # In this direction we rarely receive binary frames; ignore or extend if needed.
                continue

            try:
                obj = json.loads(msg)
            except json.JSONDecodeError:
                # ignore garbage
                continue

            # Expect CLI to forward plugin ops as {"op": "...", "msgId": "...", ...}
            op = obj.get("op")
            if not op:
                continue

            # Intercept plot/export responses to send binary frames over WS
            # We accomplish this by monkey-patching send_json + writing binary directly.

            # Temporarily override the plugin's send_json for this request cycle
            def ws_send_json(payload: Dict[str, Any]) -> None:
                asyncio.get_event_loop().create_task(ws.send(json.dumps(payload)))

            plugin.send_json = ws_send_json  # type: ignore

            # For binary: when plugin would write to stdout.buffer, we redirect to ws.send(...)
            # Minimal approach: change plugin methods to call a helper we can override; since your
            # implementation already writes via stdout.buffer only in _handle_plot/_handle_export,
            # just duplicate those write paths here:

            # Handle the op with original logic; it will call send_json(...) and create png/csv bytes
            # but instead of stdout we'll capture by intercepting at the end:
            # Easiest: call plugin.handle(obj) and rely on its internal code to call send_json,
            # then immediately after, send the binary if we can get it. To avoid heavy refactor,
            # we replicate the two places that write binary:
            if op == "plot":
                # Run the existing logic to compute png bytes but send via ws
                msg_id = obj.get("msgId")
                # Copy of _handle_plot except the final two writes:
                try:
                    # Reuse private method by calling it and replacing the two writes:
                    # Quick approach: factor out the common part would be ideal; for now,
                    # call a small wrapper:
                    png_bytes = _compute_plot_bytes(plugin, obj)  # <-- see helper below
                    await ws.send(json.dumps({"op": "plot_blob", "msgId": msg_id, "contentType": "image/png", "size": len(png_bytes)}))
                    await ws.send(png_bytes)  # binary
                except PluginError as exc:
                    ws_send_json({"op": "error", "msgId": msg_id, "code": exc.code, "message": exc.message})
                continue

            if op == "export":
                msg_id = obj.get("msgId")
                try:
                    # Produce CSV bytes using existing flow
                    filename, data, sha = _compute_export_bytes(plugin, obj)  # <-- see helper below
                    await ws.send(json.dumps({"op": "file_start", "msgId": msg_id, "contentType": "text/csv", "filename": filename}))
                    CHUNK = 256 * 1024
                    for i in range(0, len(data), CHUNK):
                        await ws.send(json.dumps({"op": "file_chunk", "msgId": msg_id, "size": min(CHUNK, len(data)-i)}))
                        await ws.send(data[i:i+CHUNK])  # binary
                    await ws.send(json.dumps({"op": "file_end", "msgId": msg_id, "sha256": sha, "size": len(data)}))
                except PluginError as exc:
                    ws_send_json({"op": "error", "msgId": msg_id, "code": exc.code, "message": exc.message})
                continue

            # All other ops (open_dataset, list_vars, preview, close_dataset) can run as-is
            plugin.handle(obj)


def main() -> None:
    ws_url = os.environ.get("ODBARGO_CLI_WS", "ws://127.0.0.1:8765").strip()  # e.g. ws://127.0.0.1:8765
    token  = os.environ.get("ODBARGO_PLUGIN_TOKEN", "").strip()

    if ws_url:
        # --- WS transport mode (self-register to CLI) ---
        import asyncio, websockets  # lightweight dep; if you prefer no extra dep, vendor a tiny client
        asyncio.run(run_ws_mode(ws_url, token))
        return

    # --- existing stdio/NDJSON mode ---
    plugin = OdbArgoViewPlugin()
    plugin.send_json({
        "type": "plugin.hello_ok",
        "pluginProtocolVersion": PLUGIN_PROTOCOL_VERSION,
        "capabilities": {
            "open_dataset": True,
            "list_vars": True,
            "preview": True,
            "plot": True,
            "export": True,
            "subset": True,
        },
    })
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                plugin.send_error(None, "INTERNAL_ERROR", "Invalid JSON received")
                continue
            plugin.handle(message)
    except KeyboardInterrupt:
        plugin.store.close_all()
