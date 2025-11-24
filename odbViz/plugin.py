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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable, Union
import os
os.environ.setdefault("XARRAY_DISABLE_PLUGIN_AUTOLOADING", "1")
warnings.filterwarnings(
    "ignore",
    message="Engine 'argo' loading failed",
    category=RuntimeWarning,
)
_DEBUG_FLAG = os.getenv("ODBVIZ_DEBUG")
if _DEBUG_FLAG is None:
    _DEBUG_FLAG = os.getenv("ODBARGO_DEBUG", "0")
ODBVIZ_DEBUG = _DEBUG_FLAG not in ("", "0", "false", "False")

import xarray as xr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from matplotlib import ticker as mticker
from matplotlib import colors as mcolors
import matplotlib.patches as mpatches

try:  # pandas is optional but required for preview/export convenience
    import pandas as pd
except ImportError:  # pragma: no cover - runtime guard
    pd = None


PLUGIN_PROTOCOL_VERSION = "1.0"
DEFAULT_PREVIEW_LIMIT = 500
MAX_PREVIEW_ROWS = 1000
MAX_PREVIEW_COLUMNS = 16
MAX_FILTER_LENGTH = 2048

# Reference MHW palette (not enforced; used when user supplies equivalent colors)
MHW_STANDARD_COLORS = ["#f5c268", "#ec6b1a", "#cb3827", "#7f1416"]

# Geo helpers (shared with legacy map_plot)
def _lon180(arr: np.ndarray) -> np.ndarray:
    return ((arr + 180.0) % 360.0) - 180.0

def _lon360(arr: np.ndarray) -> np.ndarray:
    return np.mod(arr, 360.0)

def _am_blocks_from_lon180(lon180: np.ndarray) -> List[slice]:
    x = lon180[0, :]
    x360 = np.mod(x, 360.0)
    d = np.diff(x360)
    seam = np.where(d < -180.0)[0]
    if seam.size:
        j = int(seam[0] + 1)
        return [slice(0, j), slice(j, None)]
    return [slice(None)]


def _smart_geo_ticks(lo: float, hi: float, coord_type: str = "lon") -> np.ndarray:
    span = hi - lo
    if span <= 0 or not np.isfinite(span):
        return np.array([lo])
    preferred = [5, 10, 15, 20, 30, 45, 60, 90, 180] if coord_type == "lon" else [5, 10, 15, 20, 30, 45, 90]
    for interval in preferred:
        n = span / interval
        if 3 <= n <= 8:
            start = math.ceil(lo / interval) * interval
            end = math.floor(hi / interval) * interval
            ticks = np.arange(start, end + 0.5 * interval, interval)
            if len(ticks) >= 3:
                return ticks
    return np.linspace(lo, hi, 5)

def _choose_colorbar_layout(ax, fig, engine: str = "cartopy", legend_loc: Optional[str] = None):
    """
    從 map_plot.py 移植的 Colorbar 佈局邏輯。
    決定是水平還是垂直放置，並回傳對應的參數 (pad, fraction, aspect)。
    """
    fig_w, fig_h = fig.get_size_inches()
    aspect_ratio = fig_w / max(fig_h, 1e-9)
    bottom_space = 0.2
    right_space = 0.2
    ax_height = 0.7
    if ax is not None:
        try:
            pos = ax.get_position()
            bottom_space = float(pos.y0)
            right_space = float(1 - pos.x1)
            ax_height = float(pos.height)
        except Exception:
            pass

    # Defaults tuned to be close across engines (from map_plot.py)
    horiz = {"pad": 0.12, "fraction": 0.065, "aspect": 28, "location": "bottom"}
    vert = {"pad": 0.08, "fraction": 0.06, "aspect": 20, "location": "right", "shrink": 0.6}

    if engine == "basemap":
        horiz = {"pad": 0.12, "fraction": 0.07, "aspect": 30, "location": "bottom"}
    elif engine == "plain":
        horiz = {"pad": 0.12, "fraction": 0.07, "aspect": 28, "location": "bottom"}

    norm_loc = (legend_loc or "").strip().lower()
    if norm_loc in {"top", "bottom", "left", "right"}:
        if norm_loc in {"top", "bottom"}:
            cfg = dict(horiz)
            cfg["location"] = norm_loc
            return "horizontal", cfg
        cfg = dict(vert)
        cfg["location"] = norm_loc
        # shrink relative to axis height to avoid oversized colorbar
        cfg["shrink"] = min(0.9, max(0.3, ax_height * 0.9))
        cfg["fraction"] = min(cfg.get("fraction", 0.06), max(0.025, ax_height * 0.5))
        return "vertical", cfg

    # 簡單的啟發式規則
    if bottom_space > 0.15 and aspect_ratio > 1.5:
        return "horizontal", horiz
    if right_space > 0.12:
        cfg = dict(vert)
        cfg["shrink"] = min(0.9, max(0.3, ax_height * 0.9))
        cfg["fraction"] = min(cfg.get("fraction", 0.06), max(0.025, ax_height * 0.5))
        return "vertical", cfg
    # Fallback
    return "horizontal", {"pad": 0.15, "fraction": horiz.get("fraction", 0.065), "aspect": horiz.get("aspect", 28), "location": "bottom"}

def _sort_cols_by_lon(
    L: np.ndarray,
    Z: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sort 2D longitude grid from west→east, and apply the same column order to Z.

    - L, Z: shape (nlat, nlon)
    - 回傳排序後的 (L_sorted, Z_sorted)
    """
    if L.ndim != 2 or Z.ndim != 2:
        return L, Z

    # 以第一列當代表 (每一欄的 lon)
    x = L[0, :]

    # nan 放到最後；剩下照大小排
    order = np.argsort(np.where(np.isfinite(x), x, np.inf))

    return L[:, order], Z[:, order]


def _recenter_lon_for_plain(lon_arr: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
    """
    Recenter longitudes so the data arc is centered at 0 (min gap -> max continuity).
    Returns (recentered_lons, xmin, xmax, center_deg).
    """
    arr = np.asarray(lon_arr)
    if arr.ndim == 1:
        x = arr
    else:
        x = arr[0, :]
    x360 = np.sort(np.unique(np.mod(x, 360.0)))
    if x360.size < 2:
        center = float((np.nanmean(x360) if x360.size else 0.0))
        data_len = float(np.nanmax(arr) - np.nanmin(arr)) if arr.size else 360.0
        rec = ((arr - center + 180.0) % 360.0) - 180.0
        return rec, -data_len / 2.0, data_len / 2.0, center
    diffs = np.diff(x360)
    wrap_gap = (x360[0] + 360.0) - x360[-1]
    gaps = np.concatenate([diffs, [wrap_gap]])
    k = int(np.argmax(gaps))
    data_len = 360.0 - float(gaps[k])
    start = x360[(k + 1) % x360.size]
    center = float((start + data_len / 2.0) % 360.0)
    rec = ((arr - center + 180.0) % 360.0) - 180.0
    return rec, -data_len / 2.0, data_len / 2.0, center


def _sort_cols_by_lon(x2d: np.ndarray, z2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    將 2D 網格 x2d (lon) 依欄位的第一列值排序，並以相同順序重排 z2d。
    備註：只做欄排序，不動列（lat）。
    """
    x2d = np.asarray(x2d)
    z2d = np.asarray(z2d)
    if x2d.ndim != 2 or z2d.ndim != 2:
        raise ValueError("x2d and z2d must be 2D arrays.")
    if x2d.shape != z2d.shape:
        raise ValueError("x2d and z2d must have the same shape.")

    # 以第一列作為每一欄的代表經度來排序（確保單調遞增）
    col_lon = x2d[0, :]
    order = np.argsort(col_lon)
    return x2d[:, order], z2d[:, order]


def _roll_for_plain(lon_grid: np.ndarray, val_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    針對 plain backend 的 0..360 經度格點，把資料沿最小間隙對側『翻卷』到 [-180,180]；
    回傳 (Lrec, Vrec, xmin, xmax) 給 pcolormesh 使用，避免 180° 斷裂造成的遠端帶狀錯繪。
    """
    L = np.asarray(lon_grid)
    V = np.asarray(val_grid)
    if L.ndim == 1:
        x = L
    else:
        x = L[0, :]

    x360 = np.mod(x, 360.0)
    # 找到最大跳躍（=資料弧外的缺口），在缺口「對面」置中
    diffs = np.diff(np.r_[x360, x360[0] + 360.0])
    k = int(np.argmax(diffs))  # 缺口結束的位置
    # 以缺口後的第一點當作新的起點，把序列 roll 到從那點開始遞增
    shift = (k + 1) % x360.size

    if L.ndim == 2:
        Lr = np.roll(L, -shift, axis=1)
        Vr = np.roll(V, -shift, axis=1)
    else:
        Lr = np.roll(L, -shift, axis=0)
        Vr = np.roll(V, -shift, axis=0)

    # 平移到 [-180,180]
    Lrec = ((Lr + 180.0) % 360.0) - 180.0
    xmin = float(np.nanmin(Lrec))
    xmax = float(np.nanmax(Lrec))
    return Lrec, Vr, xmin, xmax


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
        self._case_flags: Dict[str, bool] = {}

    def open_dataset(
        self,
        key: str,
        path: str,
        engine_preference: Optional[List[str]] = None,
        *,
        case_insensitive: bool = False,
    ) -> xr.Dataset:
        if key in self._datasets:
            self._datasets[key].close()
        engines = engine_preference or ["h5netcdf", "netcdf4"]
        last_err: Optional[Exception] = None
        for engine in engines:
            try:
                ds = xr.open_dataset(path, engine=engine)
                ds = self._prepare_dataset(ds, case_insensitive)
                return self.register_dataset(key, ds, case_insensitive=case_insensitive)
            except Exception as exc:  # pragma: no cover - engine fallback
                last_err = exc
                continue
        # fallback to default engine
        try:
            ds = xr.open_dataset(path)
            ds = self._prepare_dataset(ds, case_insensitive)
            return self.register_dataset(key, ds, case_insensitive=case_insensitive)
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

    def ensure_case_insensitive(self, key: str) -> None:
        root = self.root_key(key)
        if self._case_flags.get(root):
            return
        ds = self._datasets.get(root)
        if ds is None:
            raise PluginError("DATASET_NOT_FOUND", f"Dataset '{root}' not found")
        ds = self._prepare_dataset(ds, True)
        self._datasets[root] = ds
        self._case_flags[root] = True

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
        if columns is not None:
            stored_columns = list(columns)
            if self.is_case_insensitive(root):
                stored_columns = [col.lower() for col in stored_columns]
        else:
            stored_columns = None
        self._subsets[subset_key] = SubsetSpec(parent_key=root, columns=stored_columns, filters=filtered_filters)

    def close(self, key: str, *, async_close: bool = False) -> None:
        ds = self._datasets.pop(key, None)
        self._case_flags.pop(key, None)
        # remove any subsets referencing this key directly
        to_remove = [name for name, subset in self._subsets.items() if subset.parent_key == key or name == key]
        for name in to_remove:
            self._subsets.pop(name, None)
        if ds is not None:
            def _close_dataset() -> None:
                try:
                    ds.close()
                except Exception as exc:  # pragma: no cover - defensive
                    if ODBVIZ_DEBUG:
                        print(f"[plugin] dataset close error for {key}: {exc}", file=sys.stderr)
            if async_close:
                threading.Thread(target=_close_dataset, name=f"close-ds-{key}", daemon=True).start()
            else:
                _close_dataset()

    def close_all(self) -> None:  # pragma: no cover - safety
        for key in list(self._datasets.keys()):
            self.close(key)

    def is_case_insensitive(self, key: str) -> bool:
        if key in self._datasets:
            return self._case_flags.get(key, False)
        if key in self._subsets:
            return self.is_case_insensitive(self._subsets[key].parent_key)
        raise PluginError("DATASET_NOT_FOUND", f"Dataset '{key}' not found")

    def _prepare_dataset(self, ds: xr.Dataset, case_insensitive: bool) -> xr.Dataset:
        if case_insensitive:
            ds = self._lowercase_dataset(ds)
            ds = self._ensure_default_coords(ds)
        return ds

    def _lowercase_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        dim_rename: Dict[str, str] = {}
        for dim in ds.dims:
            lower = dim.lower()
            if lower != dim:
                if lower in ds.dims and lower != dim:
                    raise PluginError("DATASET_OPEN_FAIL", f"Cannot normalise dimension '{dim}' to '{lower}'")
                dim_rename[dim] = lower
        if dim_rename:
            ds = ds.rename_dims(dim_rename)

        rename_map: Dict[str, str] = {}
        seen: Dict[str, str] = {}
        for name in list(ds.coords.keys()) + list(ds.data_vars.keys()):
            lower = name.lower()
            if lower != name:
                if lower in seen and seen[lower] != name:
                    raise PluginError("DATASET_OPEN_FAIL", f"Multiple variables collapse to '{lower}' during case normalisation")
                if (lower in ds.coords or lower in ds.data_vars) and lower not in rename_map.values():
                    raise PluginError("DATASET_OPEN_FAIL", f"Variable '{name}' conflicts with existing '{lower}'")
                rename_map[name] = lower
                seen[lower] = name
        if rename_map:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"rename .* does not create an index anymore",
                    category=UserWarning,
                )
                ds = ds.rename(rename_map)
        return ds

    def _ensure_default_coords(self, ds: xr.Dataset) -> xr.Dataset:
        coord_candidates = ["longitude", "latitude", "time"]
        if any(name in ds.coords for name in coord_candidates):
            return ds
        available = set(ds.data_vars.keys()) | set(ds.coords.keys())
        missing = [name for name in coord_candidates if name not in available]
        if missing:
            raise PluginError("DATASET_OPEN_FAIL", f"Dataset missing required columns {missing} for coordinate fallback")
        promote = [name for name in coord_candidates if name in ds.data_vars]
        if promote:
            ds = ds.set_coords(promote)
        return ds

    def register_dataset(self, key: str, ds: xr.Dataset, *, case_insensitive: bool) -> xr.Dataset:
        if key in self._datasets:
            try:
                self._datasets[key].close()
            except Exception:
                pass
        self._datasets[key] = ds
        self._case_flags[key] = case_insensitive
        return ds


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


def _evaluate_json_filter(
    filter_json: Dict[str, Any],
    df: pd.DataFrame,
    column_map: Optional[Dict[str, str]] = None,
) -> pd.Series:
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
            series = _get_series(df, field, column_map)
            low = _infer_literal(low, series)
            high = _infer_literal(high, series)
            return series.between(low, high)
        # comparator nodes structure: {"op":[field, value]}
        for op, payload in node.items():
            if op in {">", ">=", "<", "<=", "=", "!="}:
                field, value = payload
                series = _get_series(df, field, column_map)
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


def _get_series(df: pd.DataFrame, field: str, column_map: Optional[Dict[str, str]] = None) -> pd.Series:
    if field in df.columns:
        return df[field]
    if column_map is not None:
        lookup = column_map.get(field.lower())
        if lookup and lookup in df.columns:
            return df[lookup]
    raise PluginError("COLUMN_UNKNOWN", f"Unknown column '{field}'")


def _evaluate_dsl_filter(
    dsl: str,
    df: pd.DataFrame,
    column_map: Optional[Dict[str, str]] = None,
) -> Tuple[pd.Series, List[str]]:
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
            series = _get_series(df, node["field"], column_map)
            low = _infer_literal(node["low"], series)
            high = _infer_literal(node["high"], series)
            return series.between(low, high)
        if op == "compare":
            series = _get_series(df, node["field"], column_map)
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


class OdbVizPlugin:
    def __init__(self) -> None:
        self.store = DatasetStore()
        self.dataset_meta: Dict[str, Dict[str, Any]] = {}
        self.available_engines = self._detect_engines()

    def _set_dataset_meta(self, dataset_key: str, meta: Dict[str, Any]) -> None:
        self.dataset_meta[dataset_key] = meta

    def _get_dataset_meta(self, dataset_key: Optional[str]) -> Dict[str, Any]:
        if not dataset_key:
            return {}
        try:
            root = self.store.root_key(dataset_key)
        except PluginError:
            return {}
        return self.dataset_meta.get(root, {})

    def _forget_dataset_meta(self, dataset_key: str) -> None:
        self.dataset_meta.pop(dataset_key, None)

    def _debug(self, message: str, msg_id: str | None = None) -> None:
        if not ODBVIZ_DEBUG:
            return
        try:
            obj = {"op": "debug", "message": str(message)}
            if msg_id:
                obj["msgId"] = msg_id
            self.send_json(obj)
        except Exception:
            pass

    def _dataset_case_flag(self, dataset_key: Optional[str], message: Dict[str, Any]) -> bool:
        requested = message.get("caseInsensitive")
        if requested is not None:
            flag = bool(requested)
            if flag and dataset_key:
                try:
                    self.store.ensure_case_insensitive(dataset_key)
                except PluginError:
                    pass
            return flag
        if dataset_key:
            try:
                return self.store.is_case_insensitive(dataset_key)
            except PluginError:
                return False
        return False

    def _normalise_case_payload(self, message: Dict[str, Any], case_insensitive: bool) -> Dict[str, Any]:
        if not case_insensitive:
            return message
        data = deepcopy(message)

        for key in ("x", "y", "z"):
            value = data.get(key)
            if isinstance(value, str):
                data[key] = value.lower()

        if isinstance(data.get("columns"), list):
            data["columns"] = [col.lower() if isinstance(col, str) else col for col in data["columns"]]

        if isinstance(data.get("groupBy"), list):
            normalised: List[str] = []
            for item in data["groupBy"]:
                if not isinstance(item, str):
                    continue
                if ":" in item:
                    base, rest = item.split(":", 1)
                    normalised.append(f"{base.lower()}:{rest}")
                else:
                    normalised.append(item.lower())
            data["groupBy"] = normalised

        if isinstance(data.get("orderBy"), list):
            normalised_order: List[Dict[str, Any]] = []
            for entry in data["orderBy"]:
                if not isinstance(entry, dict):
                    continue
                entry_copy = dict(entry)
                col_name = entry_copy.get("col")
                if isinstance(col_name, str):
                    entry_copy["col"] = col_name.lower()
                normalised_order.append(entry_copy)
            data["orderBy"] = normalised_order

        bins = data.get("bins")
        if isinstance(bins, dict):
            transformed: Dict[str, Any] = {}
            for key, value in bins.items():
                transformed[str(key).lower()] = value
            data["bins"] = transformed

        style = data.get("style")
        if isinstance(style, dict):
            style_bins = style.get("bins")
            if isinstance(style_bins, dict):
                style["bins"] = {str(k).lower(): v for k, v in style_bins.items()}

        filter_obj = data.get("filter")
        if isinstance(filter_obj, dict):
            data["filter"] = self._normalise_filter_case(filter_obj)

        return data

    def _normalise_filter_case(self, filter_obj: Dict[str, Any]) -> Dict[str, Any]:
        filt = deepcopy(filter_obj)
        json_spec = filt.get("json")
        if json_spec:
            filt["json"] = self._lowercase_filter_json(json_spec)
        return filt

    def _lowercase_filter_json(self, node: Any) -> Any:
        if isinstance(node, dict):
            if "between" in node and isinstance(node["between"], list) and node["between"]:
                field = node["between"][0]
                if isinstance(field, str):
                    node["between"][0] = field.lower()
            for key, value in list(node.items()):
                if key in {"and", "or"} and isinstance(value, list):
                    node[key] = [self._lowercase_filter_json(child) for child in value]
                elif key in {">", ">=", "<", "<=", "=", "!="} and isinstance(value, list) and value:
                    field = value[0]
                    if isinstance(field, str):
                        node[key][0] = field.lower()
            return node
        if isinstance(node, list):
            return [self._lowercase_filter_json(item) for item in node]
        return node

    def _records_dataframe_from_records(self, records: Union[List[Dict[str, Any]], Dict[str, Any]]) -> "pd.DataFrame":
        _ensure_pandas()
        if isinstance(records, list):
            return pd.DataFrame.from_records(records)
        if isinstance(records, dict) and records.get("data"):
            return pd.DataFrame.from_records(records["data"])
        if isinstance(records, dict):
            return pd.DataFrame.from_records([records])
        raise PluginError("DATASET_OPEN_FAIL", "records must be a list or dict")

    def _detect_engines(self) -> List[str]:
        engines = ["plain"]
        try:
            import cartopy  # noqa: F401
            engines.append("cartopy")
        except Exception:
            pass
        try:
            from mpl_toolkits import basemap  # noqa: F401
            engines.append("basemap")
        except Exception:
            pass
        return engines

    def _infer_time_resolution(self, df: "pd.DataFrame", field: Optional[str], current: Optional[str]) -> Optional[str]:
        if current and current != "auto":
            return current
        if not field or field not in df.columns:
            return current or "auto"
        series = pd.to_datetime(df[field], errors="coerce").sort_values()
        diffs = series.diff().dropna()
        if diffs.empty:
            return current or "auto"
        median = diffs.median()
        if pd.isna(median):
            return current or "auto"
        seconds = abs(median.total_seconds())
        if seconds >= 25 * 24 * 3600:
            return "monthly"
        if seconds >= 20 * 3600:
            return "daily"
        if seconds >= 60:
            return "hourly"
        return "auto"

    def _infer_schema_from_df(self, df: "pd.DataFrame", schema: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        schema = dict(schema or {})
        columns = list(df.columns)

        def pick(defaults: List[str], existing: Optional[str]) -> Optional[str]:
            if existing and existing in columns:
                return existing
            if existing:
                lower = existing.lower()
                for col in columns:
                    if col.lower() == lower:
                        return col
            for name in defaults:
                for col in columns:
                    if col == name:
                        return col
                    if col.lower() == name.lower():
                        return col
            return existing

        schema["timeField"] = pick(["time", "date", "timestamp"], schema.get("timeField"))
        schema["lonField"] = pick(["lon", "longitude"], schema.get("lonField"))
        schema["latField"] = pick(["lat", "latitude"], schema.get("latField"))
        schema["timeResolution"] = self._infer_time_resolution(df, schema.get("timeField"), schema.get("timeResolution"))
        return schema

    def _register_records_dataframe(
        self,
        dataset_key: str,
        df: "pd.DataFrame",
        schema: Optional[Dict[str, Any]],
        source: Optional[str],
        *,
        case_insensitive: bool,
    ) -> Dict[str, Any]:
        df = df.copy()
        df.columns = [str(col) for col in df.columns]
        inferred_schema = self._infer_schema_from_df(df, schema)
        # add canonical coordinate columns so downstream defaults work
        lon_field = inferred_schema.get("lonField")
        lat_field = inferred_schema.get("latField")
        time_field = inferred_schema.get("timeField")
        try:
            if lon_field and "longitude" not in df.columns and lon_field in df.columns:
                df["longitude"] = df[lon_field]
            if lat_field and "latitude" not in df.columns and lat_field in df.columns:
                df["latitude"] = df[lat_field]
            if time_field and "time" not in df.columns and time_field in df.columns:
                df["time"] = pd.to_datetime(df[time_field], errors="coerce")
        except Exception:
            # if coercion fails, _prepare_dataset will raise a clearer error later
            pass
        ds = xr.Dataset.from_dataframe(df.reset_index(drop=True))
        ds = self.store._prepare_dataset(ds, case_insensitive)
        self.store.register_dataset(dataset_key, ds, case_insensitive=case_insensitive)
        self._forget_dataset_meta(dataset_key)
        meta = {
            "source": source,
            "schema": inferred_schema,
            "timeResolution": inferred_schema.get("timeResolution"),
            "nRows": int(len(df)),
            "nColumns": int(len(df.columns)),
        }
        self._set_dataset_meta(dataset_key, meta)
        return {
            "nRows": meta["nRows"],
            "nColumns": meta["nColumns"],
            "timeField": inferred_schema.get("timeField"),
            "lonField": inferred_schema.get("lonField"),
            "latField": inferred_schema.get("latField"),
            "source": source,
            "timeResolution": inferred_schema.get("timeResolution"),
        }

    def _load_tabular_file(self, path: str, fmt: str) -> "pd.DataFrame":
        _ensure_pandas()
        fmt = fmt.lower()
        if fmt in {".csv"}:
            return pd.read_csv(path)
        if fmt in {".json", ".jsonl", ".ndjson"}:
            try:
                return pd.read_json(path, lines=fmt in {".jsonl", ".ndjson"})
            except ValueError:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict) and "data" in data:
                    data = data["data"]
                if isinstance(data, list):
                    return pd.DataFrame.from_records(data)
                if isinstance(data, dict):
                    return pd.DataFrame.from_records([data])
        raise PluginError("DATASET_OPEN_FAIL", f"Unsupported tabular format '{fmt}'")

    def _tabular_format_for_path(self, path: str, fmt: Optional[str]) -> Optional[str]:
        if fmt:
            fmt = fmt.lower()
        else:
            fmt = Path(path).suffix.lower()
        if fmt in {".csv", ".json", ".jsonl", ".ndjson"}:
            return fmt
        return None

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
            elif op == "open_records":
                self._handle_open_records(msg_id, message)
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
        ci_flag = message.get("caseInsensitive")
        if ci_flag is None:
            case_insensitive = True
        else:
            case_insensitive = bool(ci_flag)
        if not path or not dataset_key:
            raise PluginError("DATASET_OPEN_FAIL", "Missing path or datasetKey")
        fmt = self._tabular_format_for_path(path, message.get("format"))
        if fmt:
            df = self._load_tabular_file(path, fmt)
            self._register_records_dataframe(
                dataset_key,
                df,
                schema=message.get("schema"),
                source=message.get("source"),
                case_insensitive=case_insensitive,
            )
            ds = self.store.get(dataset_key)
            summary = self._build_dataset_summary(ds)
        else:
            ds = self.store.open_dataset(dataset_key, path, engine_preference, case_insensitive=case_insensitive)
            schema = message.get("schema")
            source = message.get("source")
            self._forget_dataset_meta(dataset_key)
            if schema or source:
                self._set_dataset_meta(dataset_key, {"schema": schema or {}, "source": source})
            summary = self._build_dataset_summary(ds)
        self.send_json({
            "op": "open_dataset.ok",
            "msgId": msg_id,
            "datasetKey": dataset_key,
            "summary": summary,
        })

    def _handle_open_records(self, msg_id: str, message: Dict[str, Any]) -> None:
        dataset_key = message.get("datasetKey")
        if not dataset_key:
            raise PluginError("DATASET_OPEN_FAIL", "Missing datasetKey")
        records = message.get("records")
        if records is None:
            raise PluginError("DATASET_OPEN_FAIL", "Missing records payload")
        source = message.get("source")
        if not source:
            raise PluginError("DATASET_OPEN_FAIL", "Missing source")
        schema = message.get("schema")
        case_flag = message.get("caseInsensitive")
        case_insensitive = True if case_flag is None else bool(case_flag)
        df = self._records_dataframe_from_records(records)
        summary = self._register_records_dataframe(
            dataset_key,
            df,
            schema=schema,
            source=source,
            case_insensitive=case_insensitive,
        )
        self.send_json({
            "op": "open_records.ok",
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

    def _mask_from_filter_specs(
        self,
        filters: List[FilterSpec],
        df: pd.DataFrame,
        column_map: Optional[Dict[str, str]] = None,
    ) -> pd.Series:
        mask = pd.Series([True] * len(df), index=df.index)
        for filter_spec in filters:
            if not (filter_spec.dsl or filter_spec.json_spec):
                continue
            if filter_spec.dsl:
                dsl_mask, _ = _evaluate_dsl_filter(filter_spec.dsl, df, column_map)
                mask = mask & dsl_mask
            if filter_spec.json_spec:
                json_mask = _evaluate_json_filter(filter_spec.json_spec, df, column_map)
                mask = mask & json_mask
        return mask.fillna(False)

    def _parse_filter(
        self,
        message: Dict[str, Any],
        df: pd.DataFrame,
        base_filters: Optional[List[FilterSpec]] = None,
        *,
        case_insensitive: bool = False,
    ) -> Tuple[pd.Series, Optional[FilterSpec]]:
        column_map = {col.lower(): col for col in df.columns} if case_insensitive else None
        filter_obj = message.get("filter") or {}
        mask = pd.Series([True] * len(df), index=df.index)
        if base_filters:
            mask = mask & self._mask_from_filter_specs(base_filters, df, column_map)
        json_nodes: List[Dict[str, Any]] = []
        stored_spec: Optional[FilterSpec] = None

        dsl_text = filter_obj.get("dsl") if filter_obj else None
        if dsl_text:
            dsl_mask, _ = _evaluate_dsl_filter(dsl_text, df, column_map)
            mask = mask & dsl_mask

        if filter_obj.get("json"):
            json_spec = deepcopy(filter_obj["json"])
            json_mask = _evaluate_json_filter(json_spec, df, column_map)
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
            mask = mask & _evaluate_json_filter(bbox_json, df, column_map)

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
                mask = mask & _evaluate_json_filter(time_json, df, column_map)

        combined_json: Optional[Dict[str, Any]] = None
        if json_nodes:
            if len(json_nodes) == 1:
                combined_json = json_nodes[0]
            else:
                combined_json = {"and": json_nodes}

        if dsl_text or combined_json:
            stored_spec = FilterSpec(dsl=dsl_text, json_spec=combined_json)

        return mask.fillna(False), stored_spec

    def _collect_columns(
        self,
        message: Dict[str, Any],
        subset_filters: Optional[List[FilterSpec]] = None,
        allowed_columns: Optional[List[str]] = None,
        *,
        case_insensitive: bool = False,
    ) -> List[str]:
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
        combined = []
        seen: set[str] = set()
        for col in list(columns) + extra_cols:
            if not col:
                continue
            if case_insensitive and isinstance(col, str):
                col_value = col.lower()
            else:
                col_value = col
            if col_value in seen:
                continue
            seen.add(col_value)
            combined.append(col_value)
        if allowed_columns is not None:
            allowed_set = set(allowed_columns)
            combined = [col for col in combined if col in allowed_set]
        return combined

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

    def _find_column(self, df: "pd.DataFrame", name: str) -> str:
        if name in df.columns:
            return name
        lower = name.lower()
        for candidate in df.columns:
            if candidate.lower() == lower:
                return candidate
        raise PluginError("COLUMN_UNKNOWN", f"Unknown groupBy column '{name}'")

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
            resolved = self._find_column(df, col)
            return df, resolved, None
        col_raw, param = (s.strip() for s in spec.split(":", 1))
        if col_raw.upper() == "TIME":
            resolved = self._find_column(df, col_raw)
            return df, resolved, param  # tell caller to resample on time
        resolved = self._find_column(df, col_raw)
        try:
            bw = float(param)
            if bw <= 0:
                raise ValueError
        except Exception:
            return df, resolved, None
        bin_col = f"__bin__{resolved}"
        df = df.copy()
        df[bin_col] = (np.floor(df[resolved].astype(float) / bw) * bw).astype(df[resolved].dtype, copy=False)
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
        lon_candidates = [message.get("x"), style.get("lon"), "LONGITUDE", "longitude", "lon"]
        lon_candidates = [c for c in lon_candidates if c]
        lon_col = self._resolve_column(df, lon_candidates or ["longitude"], "LONGITUDE")

        lat_candidates = [style.get("lat"), message.get("lat"), "LATITUDE", "latitude", "lat"]
        lat_candidates = [c for c in lat_candidates if c]
        lat_col = self._resolve_column(df, lat_candidates or ["latitude"], "LATITUDE")

        value_candidates: List[str] = []
        for key in ("z", "field", "fields"):
            candidate = style.get(key) if key != "z" else message.get("z")
            if isinstance(candidate, str):
                value_candidates.append(candidate)
        y_token = message.get("y")
        if isinstance(y_token, str):
            value_candidates.append(y_token)
        value_col: Optional[str] = None
        for cand in value_candidates:
            try:
                value_col = self._resolve_column(df, [cand], cand)
                if value_col:
                    break
            except PluginError:
                continue

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
            lon_mode: str = "native",  # <--- 新增參數
        ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
            if not value_col or value_col not in df.columns:
                return None

            use_bins = False
            lon_step = None
            lat_step = None
            if isinstance(bins, dict):
                try:
                    lon_step = float(bins.get("lon", 0.0))
                    lat_step = float(bins.get("lat", 0.0))
                    use_bins = lon_step > 0.0 and lat_step > 0.0
                except Exception:
                    use_bins = False

            # 複製資料以免影響原始 DataFrame
            b = df[[lon_col, lat_col, value_col]].dropna().copy()
            if b.empty:
                return None

            # 關鍵修正：在轉成 Grid 之前，先依照模式統一經度
            # 這樣如果是 antimeridian (如 -179...179)，轉成 360 後 (181...179) 就會變成連續的數值
            vals = b[lon_col].to_numpy(dtype=float)
            if lon_mode == "360":
                b[lon_col] = _lon360(vals)
            elif lon_mode == "180":
                b[lon_col] = _lon180(vals)

            if not use_bins:
                # 嘗試偵測既有的規則網格
                try:
                    unique_lon = np.sort(b[lon_col].unique())
                    unique_lat = np.sort(b[lat_col].unique())
                    # 寬鬆檢查網格完整性
                    if len(unique_lon) >= 2 and len(unique_lat) >= 2:
                        # 使用 pivot_table 取 mean 以處理經度轉換後可能重疊的點
                        pivot = b.pivot_table(index=lat_col, columns=lon_col, values=value_col, aggfunc='mean').sort_index().sort_index(axis=1)
                        lon_grid, lat_grid = np.meshgrid(pivot.columns.to_numpy(dtype=float), pivot.index.to_numpy(dtype=float))
                        return lon_grid, lat_grid, pivot.to_numpy()
                except Exception:
                    pass
                return None

            # Bin-based gridding (自訂網格)
            b["__lon_bin__"] = (np.floor(b[lon_col].to_numpy(dtype=float) / lon_step) * lon_step)
            b["__lat_bin__"] = (np.floor(b[lat_col].to_numpy(dtype=float) / lat_step) * lat_step)

            agg_value = str(agg or bins.get("agg", "mean")).lower() if bins else "mean"
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

            lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)
            return lon_grid, lat_grid, pivot.to_numpy()

    # ------------------------------------------------------------------
    # Map helpers (categorical cmap, engine setup, gridded rendering)
    # ------------------------------------------------------------------
    def _parse_categorical_cmap(self, cmap_spec: Any, n_categories: Optional[int] = None) -> Tuple[mcolors.Colormap, Optional[mcolors.Normalize]]:
        if cmap_spec is None:
            return plt.get_cmap("tab20"), None
        colors_list: Optional[List[str]] = None
        if isinstance(cmap_spec, str):
            text = cmap_spec.strip()
            if "," in text or text.startswith("["):
                try:
                    colors_list = json.loads(text) if text.startswith("[") else None
                except Exception:
                    colors_list = None
                if colors_list is None and "," in text:
                    colors_list = [c.strip() for c in text.split(",") if c.strip()]
        elif isinstance(cmap_spec, (list, tuple)):
            colors_list = list(cmap_spec)
        if colors_list:
            try:
                discrete_cmap = mcolors.ListedColormap(colors_list)
                if n_categories is not None and n_categories > 0:
                    bounds = [i - 0.5 for i in range(n_categories + 1)]
                    norm = mcolors.BoundaryNorm(bounds, discrete_cmap.N)
                    return discrete_cmap, norm
                return discrete_cmap, None
            except Exception as exc:
                self._debug(f"categorical cmap parse failed: {exc}", "-")
        try:
            return self._resolve_cmap(cmap_spec), None
        except Exception:
            return plt.get_cmap("tab20"), None
        
    def _setup_map_engine(
            self,
            engine: str,
            lon_lo: float,
            lon_hi: float,
            lat_lo: float,
            lat_hi: float,
            bbox_mode: str,
            figsize: Tuple[float, float],
            dpi: int,
        ) -> Tuple[plt.Figure, plt.Axes, Any]:
            """
            建立繪圖引擎與座標系。
            修正: Cartopy 改用 set_xticks/yticks 搭配 Formatter，以獲得與 Basemap 一致的精細外觀。
            """

            # ---------- CARTOPY ----------
            if engine == "cartopy":
                try:
                    import cartopy.crs as ccrs
                    import cartopy.feature as cfeature
                    from cartopy.io import DownloadWarning
                    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
                    warnings.simplefilter("ignore", DownloadWarning)

                    # 投影設定
                    if bbox_mode == "antimeridian":
                        proj = ccrs.PlateCarree(central_longitude=180.0)
                    else:
                        proj = ccrs.PlateCarree()
                    
                    data_crs = ccrs.PlateCarree()

                    fig = plt.figure(figsize=figsize, dpi=dpi)
                    ax = plt.axes(projection=proj)

                    ax._odb_cartopy_data_crs = data_crs
                    ax.set_extent([lon_lo, lon_hi, lat_lo, lat_hi], crs=data_crs)

                    try:
                        ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#d3d3d3", edgecolor="none", zorder=3)
                    except Exception:
                        pass

                    # Style Fix: 改用傳統 ticks 而非 gridliner labels，以匹配 Basemap 風格
                    try:
                        # 1. 畫內網格線 (不畫標籤)
                        gl = ax.gridlines(draw_labels=False, linewidth=0.3, color="#666666", alpha=0.5, linestyle="--")
                        
                        # 2. 設定 Ticks
                        xt = _smart_geo_ticks(lon_lo, lon_hi, coord_type="lon")
                        yt = _smart_geo_ticks(lat_lo, lat_hi, coord_type="lat")
                        
                        # 注意: Cartopy 的 set_xticks 需要指定 crs
                        ax.set_xticks(xt, crs=data_crs)
                        ax.set_yticks(yt, crs=data_crs)
                        
                        # 3. 設定 Formatter (自動處理 °E/°W)
                        ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
                        ax.yaxis.set_major_formatter(LatitudeFormatter())
                        
                        # 4. 調整字體大小與樣式 (匹配 Basemap 的 fontsize=8)
                        ax.tick_params(axis='both', which='major', labelsize=8, width=0.5, length=3)
                        
                    except Exception:
                        # Fallback 如果 formatter 失敗
                        pass

                    return fig, ax, None

                except Exception as exc:
                    print(f"[odbViz] Cartopy init failed: {exc}", file=sys.stderr)

            # ---------- BASEMAP ----------
            if engine == "basemap":
                try:
                    from mpl_toolkits.basemap import Basemap

                    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

                    m = Basemap(
                        projection="cyl",
                        llcrnrlon=lon_lo,
                        urcrnrlon=lon_hi,
                        llcrnrlat=lat_lo,
                        urcrnrlat=lat_hi,
                        resolution="l",
                        ax=ax,
                    )
                    
                    ax._odb_basemap = m

                    m.drawcoastlines(linewidth=0.7)
                    try:
                        m.fillcontinents(color="#d3d3d3", zorder=0)
                    except Exception:
                        pass

                    try:
                        xt = _smart_geo_ticks(lon_lo, lon_hi, coord_type="lon")
                        yt = _smart_geo_ticks(lat_lo, lat_hi, coord_type="lat")
                        # Basemap style: fontsize=8, linewidth=0.3
                        m.drawmeridians(xt, labels=[0, 0, 0, 1], linewidth=0.3, color="#666666", fontsize=8, dashes=[1, 0])
                        m.drawparallels(yt, labels=[1, 0, 0, 0], linewidth=0.3, color="#666666", fontsize=8, dashes=[1, 0])
                    except Exception:
                        pass

                    return fig, ax, m

                except Exception as exc:
                    print(f"[odbViz] Basemap init failed: {exc}", file=sys.stderr)

            # ---------- PLAIN (Fallback) ----------
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            ax.set_xlim(lon_lo, lon_hi)
            ax.set_ylim(lat_lo, lat_hi)
            ax.set_aspect("equal", adjustable="box")
            
            xt = _smart_geo_ticks(lon_lo, lon_hi, coord_type="lon")
            yt = _smart_geo_ticks(lat_lo, lat_hi, coord_type="lat")
            ax.set_xticks(xt)
            ax.set_yticks(yt)
            ax.tick_params(axis='both', which='major', labelsize=8, width=0.5, length=3)
            
            return fig, ax, None


    def _render_gridded_map(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        lon_grid: np.ndarray,
        lat_grid: np.ndarray,
        value_grid: np.ndarray,
        value_col: str,
        cmap: mcolors.Colormap,
        norm: Optional[mcolors.Normalize],
        vmin: Optional[float],
        vmax: Optional[float],
        is_categorical: bool,
        engine: str,
        bbox_mode: str,
    ) -> None:
        """
        在既有 map engine 上畫出 gridded data。
        修正: Colormap extend 邏輯與 map_plot.py 完全一致 (預設 both)。
        """
        vkw: Dict[str, Any] = {}
        if vmin is not None:
            vkw["vmin"] = vmin
        if vmax is not None:
            vkw["vmax"] = vmax
        
        # Extend logic from map_plot.py
        extend = "both"

        def _finalize_color(mappable=None):
            if is_categorical:
                vals = [v for v in np.unique(value_grid) if np.isfinite(v)]
                if not vals: return
                patches = []
                tn = norm; tc = cmap
                for i, val in enumerate(sorted(vals)):
                    color = tc(i / max(len(vals) - 1, 1)) if tn is None else tc(tn(val))
                    patches.append(mpatches.Patch(color=color, label=str(val)))
                ax.legend(handles=patches, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=min(4, len(patches)), frameon=False)
            elif mappable:
                try:
                    orientation, cbar_params = _choose_colorbar_layout(ax, fig, engine, style.get("legend_loc"))
                    cbar = fig.colorbar(mappable, ax=ax, orientation=orientation, extend=extend, **cbar_params)
                    
                    label = value_col
                    if orientation == "horizontal":
                        cbar.set_label(label, labelpad=10)
                    else:
                        cbar.set_label(label, rotation=90, labelpad=15)
                    cbar.ax.tick_params(labelsize=8)
                except Exception:
                    pass

        # ---------- CARTOPY ----------
        if engine == "cartopy":
            try:
                import cartopy.crs as ccrs
            except Exception:
                engine = "plain"
            else:
                data_crs = getattr(ax, "_odb_cartopy_data_crs", ccrs.PlateCarree())
                im = None
                
                if bbox_mode == "antimeridian":
                    # Splitting strategy for Cartopy
                    L180 = _lon180(lon_grid)
                    drew = False
                    for s in _am_blocks_from_lon180(L180):
                        Xs = lon_grid[:, s]
                        Zs = value_grid[:, s]
                        Xs, Zs = _sort_cols_by_lon(Xs, Zs)
                        im = ax.pcolormesh(
                            Xs, lat_grid[:, s], Zs,
                            transform=data_crs,
                            cmap=cmap, norm=norm, shading="auto", zorder=1, **vkw
                        )
                        drew = True
                    if not drew:
                        X, Z = _sort_cols_by_lon(lon_grid, value_grid)
                        im = ax.pcolormesh(X, lat_grid, Z, transform=data_crs, cmap=cmap, norm=norm, shading="auto", **vkw)
                else:
                    X, Z = _sort_cols_by_lon(lon_grid, value_grid)
                    im = ax.pcolormesh(
                        X, lat_grid, Z,
                        transform=data_crs,
                        cmap=cmap, norm=norm, shading="auto", zorder=1, **vkw
                    )
                
                _finalize_color(im)
                return

        # ---------- BASEMAP ----------
        if engine == "basemap":
            m = getattr(ax, "_odb_basemap", None)
            if m is None:
                engine = "plain"
            else:
                im = None
                if bbox_mode == "antimeridian":
                    L180 = _lon180(lon_grid)
                    for s in _am_blocks_from_lon180(L180):
                        Xs = lon_grid[:, s]
                        Zs = value_grid[:, s]
                        Xs, Zs = _sort_cols_by_lon(Xs, Zs)
                        if vmin is not None and vmax is not None:
                             levels = np.linspace(vmin, vmax, 21)
                             im = m.contourf(Xs, lat_grid[:, s], Zs, levels=levels, cmap=cmap, norm=norm, extend=extend, latlon=True)
                        else:
                             im = m.contourf(Xs, lat_grid[:, s], Zs, cmap=cmap, norm=norm, extend=extend, latlon=True)
                else:
                    X, Z = _sort_cols_by_lon(lon_grid, value_grid)
                    if vmin is not None and vmax is not None:
                        levels = np.linspace(vmin, vmax, 21)
                        im = m.contourf(X, lat_grid, Z, levels=levels, cmap=cmap, norm=norm, extend=extend, latlon=True)
                    else:
                        im = m.contourf(X, lat_grid, Z, cmap=cmap, norm=norm, extend=extend, latlon=True)

                _finalize_color(im)
                return

        # ---------- PLAIN ----------
        im = None
        if bbox_mode == "antimeridian":
            Lrec, xmin, xmax, _center = _recenter_lon_for_plain(lon_grid)
            X, Z = _sort_cols_by_lon(Lrec, value_grid)
            im = ax.pcolormesh(X, lat_grid, Z, cmap=cmap, norm=norm, shading="auto", **vkw)
            ax.set_xlim(xmin, xmax)
        elif bbox_mode == "crossing-zero":
            X = _lon180(lon_grid)
            X, Z = _sort_cols_by_lon(X, value_grid)
            im = ax.pcolormesh(X, lat_grid, Z, cmap=cmap, norm=norm, shading="auto", **vkw)
        else:
            X, Z = _sort_cols_by_lon(lon_grid, value_grid)
            im = ax.pcolormesh(X, lat_grid, Z, cmap=cmap, norm=norm, shading="auto", **vkw)

        ax.set_aspect("equal", adjustable="box")
        xt = _smart_geo_ticks(*ax.get_xlim(), coord_type="lon")
        yt = _smart_geo_ticks(*ax.get_ylim(), coord_type="lat")
        ax.set_xticks(xt)
        ax.set_yticks(yt)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(self._format_deg()))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(self._format_lat()))

        _finalize_color(im)
        return


    def _render_plot_map_section(
        self,
        message: Dict[str, Any],
        df: pd.DataFrame,
        style: Dict[str, Any],
        bins_param: Dict[str, float],
        cmap_resolved: Optional[mcolors.Colormap],
        vmin: Optional[float],
        vmax: Optional[float],
        figsize: Tuple[float, float],
        dpi: int,
    ) -> bytes:
        lon_col, lat_col, value_col = self._resolve_map_columns(df, message, None)
        if lon_col not in df.columns or lat_col not in df.columns:
            raise PluginError("PLOT_FAIL", "map plot requires longitude and latitude columns")

        raw_lons = df[lon_col].to_numpy(dtype=float)
        bbox_mode = self._auto_bbox_mode(raw_lons, message.get("bboxMode") or message.get("bbox_mode"))

        lon_mode = "native"
        if bbox_mode == "crossing-zero":
            lon_mode = "180"
        elif bbox_mode == "antimeridian":
             lon_mode = "360"

        engine_requested = message.get("engine") or style.get("engine")
        engine = "plain"
        if isinstance(engine_requested, str):
            cand = engine_requested.strip().lower()
            if cand in [e.lower() for e in self.available_engines]:
                engine = cand

        is_categorical = bool(message.get("categorical"))
        cmap_spec = style.get("cmap") or message.get("cmap")
        if is_categorical:
            n_categories = None
            if value_col and value_col in df.columns:
                try:
                    n_categories = len([v for v in pd.unique(df[value_col]) if pd.notna(v)])
                except Exception:
                    n_categories = None
            cmap_to_use, cmap_norm = self._parse_categorical_cmap(cmap_spec, n_categories)
        else:
            cmap_to_use = cmap_resolved or self._resolve_cmap(cmap_spec) or plt.get_cmap("viridis")
            cmap_norm = None

        use_gridded = bool(message.get("gridded"))
        agg_param = message.get("agg") or (bins_param.get("agg") if isinstance(bins_param, dict) else None)
        grid = None
        
        if use_gridded and value_col:
            grid = self._build_map_grid(df, lon_col, lat_col, value_col, bins_param, agg_param, lon_mode=lon_mode)

        if grid is not None:
            lon_vals = grid[0].flatten()
            lat_vals = grid[1].flatten()
        else:
            plot_df = df.copy()
            if lon_mode == "360":
                plot_df[lon_col] = _lon360(plot_df[lon_col].to_numpy(dtype=float))
            elif lon_mode == "180":
                plot_df[lon_col] = _lon180(plot_df[lon_col].to_numpy(dtype=float))
            lon_vals = plot_df[lon_col].to_numpy(dtype=float)
            lat_vals = plot_df[lat_col].to_numpy(dtype=float)

        lon_lo = float(np.nanmin(lon_vals)) if lon_vals.size else -180.0
        lon_hi = float(np.nanmax(lon_vals)) if lon_vals.size else 180.0
        lat_lo = float(np.nanmin(lat_vals)) if lat_vals.size else -90.0
        lat_hi = float(np.nanmax(lat_vals)) if lat_vals.size else 90.0

        if engine == "plain" and bbox_mode == "antimeridian":
            if grid is not None:
                lon_grid, lat_grid, value_grid = grid
                lon_grid, lon_lo, lon_hi, center = _recenter_lon_for_plain(lon_grid)
                grid = (lon_grid, lat_grid, value_grid)
                lon_vals = lon_grid.flatten()
            else:
                lon_vals_rec, lon_lo, lon_hi, center = _recenter_lon_for_plain(lon_vals)
                plot_df[lon_col] = lon_vals_rec
                lon_vals = lon_vals_rec

        fig, ax, engine_obj = self._setup_map_engine(engine, lon_lo, lon_hi, lat_lo, lat_hi, bbox_mode, figsize, dpi)

        try:
            orientation = None # Initialize
            
            if use_gridded and grid is not None:
                lon_grid, lat_grid, value_grid = grid
                self._render_gridded_map(
                    fig, ax, lon_grid, lat_grid, value_grid, value_col or "value",
                    cmap_to_use, cmap_norm, vmin, vmax, is_categorical, engine, bbox_mode
                )
            else:
                # Scatter Plot
                point_size = style.get("pointSize") or message.get("pointSize") or 36
                scatter_kwargs = {
                    "s": float(point_size),
                    "alpha": style.get("alpha", 0.75),
                    "edgecolors": style.get("edgecolor", "none"),
                    "marker": style.get("marker", "o"),
                }
                
                color_values = plot_df[value_col] if value_col and value_col in plot_df.columns else None
                
                if engine == "cartopy":
                    import cartopy.crs as ccrs
                    scatter_kwargs["transform"] = getattr(ax, "_odb_cartopy_data_crs", ccrs.PlateCarree())

                def _do_scatter(x, y, **kw):
                    if engine == "basemap" and engine_obj is not None:
                        engine_obj.scatter(x, y, latlon=True, **kw)
                    else:
                        ax.scatter(x, y, **kw)

                if is_categorical and color_values is not None:
                    unique_vals = sorted([v for v in pd.unique(color_values) if pd.notna(v)])
                    n = max(len(unique_vals), 1)
                    for i, val in enumerate(unique_vals):
                        mask = color_values == val
                        color = cmap_to_use(cmap_norm(val)) if cmap_norm else cmap_to_use(i / max(n - 1, 1))
                        _do_scatter(plot_df.loc[mask, lon_col], plot_df.loc[mask, lat_col], color=[color], label=str(val), **scatter_kwargs)
                    if len(unique_vals) <= 20:
                        ax.legend(fontsize=8, loc="best", frameon=False)
                elif color_values is not None:
                    if engine == "basemap" and engine_obj is not None:
                        scatter = engine_obj.scatter(plot_df[lon_col], plot_df[lat_col], c=color_values, cmap=cmap_to_use, vmin=vmin, vmax=vmax, latlon=True, **scatter_kwargs)
                    else:
                        scatter = ax.scatter(plot_df[lon_col], plot_df[lat_col], c=color_values, cmap=cmap_to_use, vmin=vmin, vmax=vmax, **scatter_kwargs)
                    extend = "both"

                    orientation, cbar_params = _choose_colorbar_layout(ax, fig, engine, style.get("legend_loc"))
                    cbar = fig.colorbar(scatter, ax=ax, orientation=orientation, extend=extend, **cbar_params)
                    
                    if orientation == "horizontal":
                        cbar.set_label(value_col, labelpad=10)
                    else:
                        cbar.set_label(value_col, rotation=90, labelpad=15)
                    cbar.ax.tick_params(labelsize=8)
                else:
                    _do_scatter(plot_df[lon_col], plot_df[lat_col], **scatter_kwargs)

            if not style.get("title"):
                ax.set_title("")

            if orientation is None:
                orientation, _ = _choose_colorbar_layout(ax, fig, engine, style.get("legend_loc"))

            if is_categorical and use_gridded:
                plt.subplots_adjust(bottom=0.22)
            else:
                if orientation == "horizontal":
                    plt.subplots_adjust(bottom=0.25)
                else:
                    plt.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
            return buf.getvalue()
        finally:
            plt.close(fig)

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

    def _normalize_lons(self, arr: np.ndarray) -> np.ndarray:
        return ((arr + 180.0) % 360.0) - 180.0

    def _lon_extent(self, lons: np.ndarray) -> Tuple[float, float]:
        if lons.size == 0:
            return (-180.0, 180.0)
        norm = self._normalize_lons(lons)
        lo = float(np.nanmin(norm))
        hi = float(np.nanmax(norm))
        return lo, hi

    def _auto_bbox_mode(self, lons: np.ndarray, explicit: Optional[str] = None) -> str:
        if explicit:
            return explicit
        if lons.size == 0:
            return "none"
        lo = float(np.nanmin(lons))
        hi = float(np.nanmax(lons))
        span = hi - lo
        if span > 180.0:
            return "antimeridian"
        if lo < 0.0 < hi:
            return "crossing-zero"
        return "none"

    def _format_deg(self):
        def _fmt(val, pos=None):
            if not np.isfinite(val):
                return ""
            hemi = "E"
            if val < 0:
                hemi = "W"
            return f"{abs(val):.0f}°{hemi}"
        return _fmt

    def _format_lat(self):
        def _fmt(val, pos=None):
            if not np.isfinite(val):
                return ""
            hemi = "N"
            if val < 0:
                hemi = "S"
            return f"{abs(val):.0f}°{hemi}"
        return _fmt


    def _apply_dataset_hints(self, message: Dict[str, Any]) -> Dict[str, Any]:
        dataset_key = message.get("datasetKey")
        meta = self._get_dataset_meta(dataset_key)
        if not meta:
            return message
        schema = meta.get("schema") or {}
        style = message.setdefault("style", {}) if isinstance(message.get("style"), dict) else message.setdefault("style", {})

        if schema.get("timeField") and not message.get("timeField"):
            message["timeField"] = schema["timeField"]
        if schema.get("lonField") and not style.get("lon"):
            style["lon"] = schema["lonField"]
        if schema.get("latField") and not style.get("lat"):
            style["lat"] = schema["latField"]
        if schema.get("timeResolution") and not message.get("timeResolution"):
            message["timeResolution"] = schema["timeResolution"]
        if meta.get("source") and not message.get("source"):
            message["source"] = meta.get("source")
        kind = message.get("kind")
        if kind in {"timeseries", "climatology"} and not message.get("x"):
            message["x"] = message.get("timeField") or schema.get("timeField")
        return message

    def _resolve_cmap(self, spec: Any) -> Optional[mcolors.Colormap]:
        if spec is None:
            return None
        orig = spec
        # If user passed a JSON-ish string or comma-separated list, parse into colors first
        if isinstance(spec, str):
            text = spec.strip()
            parsed: Any = None
            if text.startswith("[") or "," in text:
                try:
                    parsed = json.loads(text)
                except Exception:
                    parsed = [tok.strip() for tok in text.split(",") if tok.strip()]
            if parsed is not None:
                spec = parsed
        if isinstance(spec, (list, tuple)):
            try:
                return mcolors.ListedColormap(list(spec))
            except Exception as exc:
                raise PluginError("PLOT_FAIL", "Invalid cmap list", hint=str(exc)) from exc
        try:
            return plt.get_cmap(spec)  # type: ignore[arg-type]
        except Exception as exc:
            raise PluginError("PLOT_FAIL", f"Invalid cmap '{orig}'", hint=str(exc)) from exc

    def _build_color_cycle(self, cmap_name: Optional[str], target_size: int) -> Optional[List[Tuple[float, ...]]]:
        cmap = None
        try:
            cmap = self._resolve_cmap(cmap_name)
        except PluginError as exc:
            self._debug(f"color_cycle: {exc.message}", "-")
            return None
        if cmap is None:
            return None
        try:
            total = max(1, int(target_size))
        except Exception:
            total = 1
        total = min(total, 64)
        positions = np.asarray([0.5]) if total == 1 else np.linspace(0.02, 0.98, total)
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
            base_ds, subset_spec = self.store.resolve(dataset_key)
            case_insensitive = self._dataset_case_flag(dataset_key, message)
            message = self._normalise_case_payload(message, case_insensitive)
            subset_key = message.get("subsetKey")
            allowed_columns = subset_spec.columns if (subset_spec and subset_spec.columns is not None) else None

            df = _build_dataframe(base_ds)

            # Collect requested columns using your existing helper
            explicit_columns = bool(message.get("columns"))
            columns = self._collect_columns(
                message,
                subset_filters=subset_spec.filters if subset_spec else None,
                allowed_columns=allowed_columns,
                case_insensitive=case_insensitive,
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
                message,
                df,
                base_filters=subset_spec.filters if subset_spec else None,
                case_insensitive=case_insensitive,
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
        message = self._apply_dataset_hints(message)
        dataset_key = message.get("datasetKey")
        if not dataset_key:
            raise PluginError("DATASET_NOT_FOUND", "Missing datasetKey")
        kind = message.get("kind")
        if not kind:
            raise PluginError("PLOT_FAIL", "Missing plot kind")
        case_insensitive = self._dataset_case_flag(message.get("datasetKey"), message)
        message = self._normalise_case_payload(message, case_insensitive)
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

        filter_mask, _ = self._parse_filter(
            message,
            df,
            base_filters=subset_spec.filters if subset_spec else None,
            case_insensitive=case_insensitive,
        )
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

        cmap_resolved: Optional[mcolors.Colormap] = None
        try:
            cmap_resolved = self._resolve_cmap(style.get("cmap"))
        except PluginError as exc:
            # propagate as plot error before touching matplotlib
            raise

        def _float_opt(val: Any) -> Optional[float]:
            if val is None:
                return None
            try:
                f = float(val)
            except Exception:
                return None
            return f if np.isfinite(f) else None

        vmin = _float_opt(style.get("vmin"))
        vmax = _float_opt(style.get("vmax"))

        if kind == "map":
            return self._render_plot_map_section(message, df, style, bins_param, cmap_resolved, vmin, vmax, figsize, dpi)

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

            elif kind == "climatology":
                time_col = message.get("timeField") or x_col
                if not (_exists(time_col) and _exists(y_col)):
                    raise PluginError("COLUMN_UNKNOWN", "climatology plot requires timeField and y column")
                work = df[[time_col, y_col] + [col for col in group_by if col in df.columns]].copy()
                try:
                    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
                except Exception:
                    pass
                work["__month__"] = work[time_col].dt.month
                work = work.dropna(subset=["__month__", y_col])
                if work.empty:
                    raise PluginError("PLOT_FAIL", "No rows available for climatology plot")

                gb_cols: List[str] = []
                for token in group_by:
                    spec = str(token).strip()
                    if not spec:
                        continue
                    if ":" in spec:
                        raise PluginError("PLOT_FAIL", "climatology groupBy does not support binning/resample syntax")
                    if spec not in work.columns:
                        raise PluginError("COLUMN_UNKNOWN", f"Unknown group-by column '{spec}'")
                    gb_cols.append(spec)

                agg_keys = gb_cols + ["__month__"]
                grouped = work.groupby(agg_keys, dropna=False)[y_col]
                if reducer == "count":
                    agg_df = grouped.count().reset_index(name="__value__")
                else:
                    agg_df = grouped.aggregate(reducer).reset_index(name="__value__")

                if color_cycle:
                    ax.set_prop_cycle(color=color_cycle)
                    color_iter = iter(color_cycle)
                else:
                    color_iter = None

                month_ticks = np.arange(1, 13)
                month_labels = style.get("monthLabels")
                if not isinstance(month_labels, list) or len(month_labels) != 12:
                    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

                def _label_from_group(key) -> Optional[str]:
                    if not gb_cols:
                        return None
                    if not isinstance(key, tuple):
                        key = (key,)
                    parts = []
                    for col, value in zip(gb_cols, key):
                        parts.append(f"{col}={value}")
                    return ", ".join(parts)

                def _series_for_group(rows: "pd.DataFrame") -> List[float]:
                    vals = [np.nan] * 12
                    for _, row in rows.iterrows():
                        try:
                            month_idx = int(row["__month__"]) - 1
                        except Exception:
                            continue
                        if 0 <= month_idx < 12:
                            vals[month_idx] = row["__value__"]
                    return vals

                def _group_iterator():
                    if gb_cols:
                        for key, sub in agg_df.groupby(gb_cols, dropna=False):
                            yield key, sub
                    else:
                        yield None, agg_df

                for key, rows in _group_iterator():
                    values = _series_for_group(rows)
                    label = _label_from_group(key)
                    kwargs = {
                        "linestyle": style.get("line", "-"),
                        "alpha": float(style.get("alpha", 1.0)),
                    }
                    marker = style.get("marker")
                    if marker:
                        kwargs["marker"] = marker
                    if label:
                        kwargs["label"] = label
                    if color_iter is not None:
                        try:
                            kwargs["color"] = next(color_iter)
                        except StopIteration:
                            pass
                    ax.plot(month_ticks, values, **kwargs)

                ax.set_xticks(month_ticks)
                ax.set_xticklabels(month_labels)
                ax.set_xlabel(style.get("labelX") or "Month")
                ax.set_ylabel(style.get("labelY") or y_col)
                if style.get("grid", True):
                    ax.grid(True, alpha=0.3)

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
                return self._render_plot_map_section(message, df, style, bins_param, cmap_resolved, vmin, vmax, figsize, dpi)
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

        case_insensitive = self._dataset_case_flag(dataset_key, message)
        message = self._normalise_case_payload(message, case_insensitive)

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

        filter_mask, _ = self._parse_filter(
            message,
            df,
            base_filters=subset_spec.filters if subset_spec else None,
            case_insensitive=case_insensitive,
        )
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
        try:
            root_key = self.store.root_key(dataset_key)
        except PluginError:
            root_key = dataset_key
        self.store.close(dataset_key, async_close=True)
        self._forget_dataset_meta(root_key)
        self.send_json({
            "op": "close_dataset.ok",
            "msgId": msg_id,
            "datasetKey": dataset_key,
        })

def _compute_plot_bytes(plugin: OdbVizPlugin, message: Dict[str, Any]) -> bytes:
    return plugin._render_plot(message, message.get("msgId"))

def _compute_export_bytes(plugin: OdbVizPlugin, message: Dict[str, Any]) -> Tuple[str, bytes, str]:
    dataset_key = message.get("datasetKey")
    case_insensitive = plugin._dataset_case_flag(dataset_key, message)
    message = plugin._normalise_case_payload(message, case_insensitive)
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
    filter_mask, _ = plugin._parse_filter(
        message,
        df,
        base_filters=subset_spec.filters if subset_spec else None,
        case_insensitive=case_insensitive,
    )
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
    plugin = OdbVizPlugin()

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
            "open_records": True,
            "list_vars": True,
            "preview": True,
            "plot": True,
            "export": True,
            "subset": True,
            "engine": plugin.available_engines,
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
                except Exception as exc:
                    ws_send_json({"op": "error", "msgId": msg_id, "code": "PLOT_FAIL", "message": str(exc)})
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
                except Exception as exc:
                    ws_send_json({"op": "error", "msgId": msg_id, "code": "EXPORT_FAIL", "message": str(exc)})
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
    plugin = OdbVizPlugin()
    plugin.send_json({
        "type": "plugin.hello_ok",
        "pluginProtocolVersion": PLUGIN_PROTOCOL_VERSION,
        "capabilities": {
            "open_dataset": True,
            "open_records": True,
            "list_vars": True,
            "preview": True,
            "plot": True,
            "export": True,
            "subset": True,
            "engine": plugin.available_engines,
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
# Reference MHW palette (not enforced; used when user supplies equivalent colors)
MHW_STANDARD_COLORS = ["#f5c268", "#ec6b1a", "#cb3827", "#7f1416"]
