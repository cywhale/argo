import json
import sys
import traceback
import io
import math
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterable, Union

import xarray as xr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

    def close(self, key: str) -> None:
        ds = self._datasets.pop(key, None)
        if ds is not None:
            ds.close()
        # remove any subsets referencing this key directly
        to_remove = [name for name, subset in self._subsets.items() if subset.parent_key == key or name == key]
        for name in to_remove:
            self._subsets.pop(name, None)

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
        stored_spec: Optional[FilterSpec] = None
        if not filter_obj:
            return mask.fillna(False), stored_spec
        if "dsl" in filter_obj and filter_obj["dsl"]:
            dsl_mask, _ = _evaluate_dsl_filter(filter_obj["dsl"], df)
            mask = mask & dsl_mask
        if "json" in filter_obj and filter_obj["json"]:
            json_mask = _evaluate_json_filter(filter_obj["json"], df)
            mask = mask & json_mask
        stored_spec = FilterSpec(dsl=filter_obj.get("dsl"), json_spec=filter_obj.get("json"))
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
        dataset_key = message.get("datasetKey")
        subset_key = message.get("subsetKey")
        base_ds, subset_spec = self.store.resolve(dataset_key)
        allowed_columns = subset_spec.columns if subset_spec and subset_spec.columns is not None else None
        df = _build_dataframe(base_ds)
        explicit_columns = bool(message.get("columns"))
        columns = self._collect_columns(
            message,
            subset_filters=subset_spec.filters if subset_spec else None,
            allowed_columns=allowed_columns,
        )
        if not columns:
            if allowed_columns is not None:
                columns = list(allowed_columns)
            else:
                columns = list(df.columns[:MAX_PREVIEW_COLUMNS])
        if allowed_columns is not None and any(col not in allowed_columns for col in columns):
            raise PluginError("COLUMN_UNKNOWN", "Requested column not in subset")
        limit = message.get("limit", DEFAULT_PREVIEW_LIMIT)
        cursor = message.get("cursor")
        order_by = message.get("orderBy") or []
        if limit > MAX_PREVIEW_ROWS:
            limit = MAX_PREVIEW_ROWS
        if limit > 10000:
            raise PluginError("ROW_LIMIT_EXCEEDED", "Preview limit too large")
        if columns and len(columns) > MAX_PREVIEW_COLUMNS:
            if explicit_columns:
                raise PluginError("PREVIEW_TOO_LARGE", "Too many columns requested")
            columns = columns[:MAX_PREVIEW_COLUMNS]
        available_columns = list(df.columns)
        for col in columns:
            if col not in available_columns:
                raise PluginError("COLUMN_UNKNOWN", f"Unknown column '{col}'")
        filter_mask, requested_spec = self._parse_filter(message, df, base_filters=subset_spec.filters if subset_spec else None)
        df = df[filter_mask]
        if order_by:
            sort_cols = []
            ascending = []
            for entry in order_by:
                col = entry.get("col")
                direction = entry.get("dir", "asc").lower()
                if col not in df.columns:
                    raise PluginError("COLUMN_UNKNOWN", f"Unknown column '{col}'")
                sort_cols.append(col)
                ascending.append(direction != "desc")
            df = df.sort_values(by=sort_cols, ascending=ascending, kind="mergesort")
        total_rows = len(df)
        start = int(cursor) if cursor else 0
        if start < 0:
            start = 0
        end = start + min(limit, MAX_PREVIEW_ROWS)
        limited_df = df.iloc[start:end]
        limit_hit = end < total_rows
        next_cursor = str(end) if limit_hit else None
        rows = []
        for _, row in limited_df.iterrows():
            rows.append([_format_value(row.get(col)) for col in columns])
        if subset_key:
            filter_specs: List[FilterSpec] = []
            if subset_spec and subset_spec.filters:
                filter_specs.extend(subset_spec.filters)
            if requested_spec and (requested_spec.dsl or requested_spec.json_spec):
                filter_specs.append(requested_spec)
            if explicit_columns:
                stored_columns = list(message.get("columns") or [])
                if stored_columns and allowed_columns is not None:
                    stored_columns = [col for col in stored_columns if col in allowed_columns]
                stored_columns = stored_columns if stored_columns else None
            else:
                if subset_spec and subset_spec.columns is not None:
                    stored_columns = list(subset_spec.columns)
                else:
                    stored_columns = None
            self.store.register_subset(subset_key, dataset_key, stored_columns, filter_specs)
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

    def _handle_plot(self, msg_id: str, message: Dict[str, Any]) -> None:
        dataset_key = message.get("datasetKey")
        kind = message.get("kind")
        x_col = message.get("x")
        y_col = message.get("y")
        style = message.get("style") or {}
        base_ds, subset_spec = self.store.resolve(dataset_key)
        df = _build_dataframe(base_ds)
        allowed_columns = subset_spec.columns if subset_spec and subset_spec.columns is not None else None
        if allowed_columns is not None:
            missing = [col for col in [x_col, y_col, message.get("z")] if col and col not in allowed_columns]
            if missing:
                raise PluginError("COLUMN_UNKNOWN", f"Column '{missing[0]}' not in subset")
        filter_mask, _ = self._parse_filter(message, df, base_filters=subset_spec.filters if subset_spec else None)
        df = df[filter_mask]
        if df.empty:
            raise PluginError("PLOT_FAIL", "Filter returned no rows")
        width = style.get("width", 800)
        height = style.get("height", 600)
        dpi = style.get("dpi", 120)
        figsize = (width / dpi, height / dpi)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        try:
            if kind == "timeseries":
                if not x_col or not y_col:
                    raise PluginError("PLOT_FAIL", "timeseries plot requires x and y columns")
                ax.plot(df[x_col], df[y_col], marker=style.get("marker", ""), linestyle=style.get("line", "-"), alpha=style.get("alpha", 1.0))
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
            elif kind == "profile":
                if not x_col:
                    raise PluginError("PLOT_FAIL", "profile plot requires x column")
                y_column = y_col or "PRES"
                ax.plot(df[x_col], df[y_column], marker=style.get("marker", "."), linestyle=style.get("line", "-"), alpha=style.get("alpha", 1.0))
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_column)
                if style.get("invert_y", True):
                    ax.invert_yaxis()
            elif kind == "map":
                lon_col = style.get("lon", x_col or "LONGITUDE")
                lat_col = style.get("lat", y_col or "LATITUDE")
                if lon_col not in df.columns or lat_col not in df.columns:
                    raise PluginError("PLOT_FAIL", "map plot requires longitude and latitude columns")
                ax.scatter(df[lon_col], df[lat_col], c=df.get(message.get("z")) if message.get("z") in df.columns else None, s=style.get("size", 20), alpha=style.get("alpha", 0.8))
                ax.set_xlabel(lon_col)
                ax.set_ylabel(lat_col)
            else:
                raise PluginError("PLOT_FAIL", f"Unsupported plot kind '{kind}'")
            if style.get("title"):
                ax.set_title(style["title"])
            if style.get("grid"):
                ax.grid(True)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
            png_bytes = buf.getvalue()
        finally:
            plt.close(fig)
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
        self.store.close(dataset_key)
        self.send_json({
            "op": "close_dataset.ok",
            "msgId": msg_id,
            "datasetKey": dataset_key,
        })


def main() -> None:
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
    except KeyboardInterrupt:  # pragma: no cover - graceful exit
        plugin.store.close_all()
    except Exception:  # pragma: no cover - final safeguard
        plugin.store.close_all()
        raise
