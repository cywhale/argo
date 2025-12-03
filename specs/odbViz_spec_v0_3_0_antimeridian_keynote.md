# Technical Keynotes: Handling Antimeridian Crossing in Geospatial Plotting

Canonical reference for odbViz (mirrors the proven `map_plot.py` behavior). Use this to keep Pacific‑centric views (e.g., 135°E → 60°W) clean and artifact‑free.

## 1) End‑to‑End Flow (fetch → viewer)
- Preserve the **original user bbox** when deriving `bboxMode`; do **not** recompute from split API chunks.
- Pass that original bbox to odbViz so it can choose the correct antimeridian projection.
- Do **not** limit rows for map plots before bbox detection; full extent is required.

## 2) Data Preparation (Gridding)
- For antimeridian cases, normalize longitudes to **0–360** when building the grid so the mesh is continuous in memory (e.g., −179 → 181).
- Rendering still occurs on a split **−180..180** view; 0–360 is only for gridding continuity.
- If a bbox override is provided for antimeridian, keep it monotonic in 0–360 to avoid world‑wrap extents.

## 3) Basemap Engine (matches map_plot.py)
- Projection: `projection="cyl"`, always pass `latlon=True`.
- Seam handling: convert grid to −180..180, detect the 180° seam, **split into slices**, and call `contourf`/`pcolormesh` per slice. This prevents streaks/ghost bands.
- Ticks: drawmeridians/drawparallels; Basemap coastlines + fillcontinents as before.

## 4) Cartopy Engine
- Axes projection: `ccrs.PlateCarree(central_longitude=180.0)` (Pacific‑centered). Data transform remains `ccrs.PlateCarree()`.
- Seam handling: convert grid to −180..180, find seam (`diff(lon) < -180`), **split into continuous blocks**, plot each block with `pcolormesh(..., transform=PlateCarree())`.
- Ticks: manual `set_xticks/set_yticks` + `LongitudeFormatter/LatitudeFormatter`; avoid gridliner labels to match Basemap style.

## 5) Plain Engine (fallback)
- For antimeridian: recenter to −180..180 with `_recenter_lon_for_plain`, set x‑limits to the recentered span, and split if needed.

## Summary Table

| Feature            | Basemap Strategy                                        | Cartopy Strategy                                             |
| :----------------- | :------------------------------------------------------ | :----------------------------------------------------------- |
| Gridding           | Build on 0–360; render on −180..180                    | Build on 0–360; render split −180..180 blocks                |
| View Projection    | `cyl` + `latlon=True`                                  | `PlateCarree(central_longitude=180)`                         |
| Data Transform     | `latlon=True` per slice                                | `transform=PlateCarree()`                                    |
| Seam Handling      | Split slices at 180° seam before contourf/pcolormesh   | Split slices at seam before pcolormesh                       |
| Ticks              | drawmeridians/drawparallels                            | set_xticks + LongitudeFormatter                              |
| Extent override    | Use monotonic 0–360 when bbox is antimeridian          | Same                                                         |
