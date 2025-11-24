# Repository Guidelines

## Project Structure & Module Organization
Core CLI sources live in `cli/` (entry points `__main__.py`, `odbargo_cli.py`, and slash-command helpers) while the optional viewer plugin sits in `odbargo_view/`. The FastAPI bridge that powers Ocean APIverse stays under `argo_app/`, with shared defaults in `argo_app/src/config.py`. PyInstaller specs (`cli.spec`, `view.spec`) and entry helpers (`cli_entry.py`, `view_entry.py`) remain at the repo root; outputs land in `dist/`. Supporting artifacts live in `scripts/` (build automation), `conf/` (sample configs), `specs/` (UX references), and `dev/` (notebooks). Tests target CLI behaviors inside `tests/` (`test_slash_commands.py`, `test_plugin_client.py`, `test_wmo_parser.py`).

## Build, Test, and Development Commands
```
python -m venv .venv && source .venv/bin/activate
pip install -e .[view]          # editable CLI plus viewer extras
python cli_entry.py             # run websocket CLI from source
python view_entry.py            # launch viewer shell
uvicorn argo_app.app:app --reload --port 8090   # FastAPI service
python -m pytest                # run suite
python -m build                 # build wheel/tarball
```
Refresh PyInstaller binaries via `scripts/build_odbargo-cli.sh` (Linux) or the Windows helper under `scripts/win_installer/`.

## Coding Style & Naming Conventions
Stick to PEP 8 with 4-space indentation, snake_case functions, and PascalCase classes. Extend CLI options with long flags that mirror existing names (`--port`, `--case-insensitive-vars`) plus env overrides like `ODBARGO_CASE_INSENSITIVE_VARS`. Keep FastAPI route handlers thin by pushing shared logic into helpers and prefer `logging` over `print`. Add type hints/docstrings when public APIs change and update `pyproject.toml` whenever scripts or extras shift.

## Testing Guidelines
Tests rely on `pytest`; name modules `test_<feature>.py` for auto-discovery. When a scenario hits argopy or ERDDAP, stub `DataFetcher` or feed temp NetCDF fixtures via `tmp_path` to keep the suite deterministic. After automated tests, hit `curl http://localhost:8090/argo/api/test` for the API heartbeat and run one CLI download (`odbargo-cli 5903377`) to exercise the websocket path. Capture unusual manual steps in PR notes.

## Commit & Pull Request Guidelines
Commits use short, imperative subjects (“add viewer spec regression tests”). Group related edits, exclude generated binaries/notebooks, and keep version bumps (`version*.txt`, `dist/manifest`) synchronized with CLI changes. PRs should outline user impact, link issues, and attach CLI/API logs or viewer screenshots whenever behavior shifts. Update the relevant README/spec if commands, defaults, or ports change.

## Security & Configuration Tips
Never commit credentials. Keep ERDDAP or API tokens in local `.env` files loaded before invoking `uvicorn`. Adjust cache/output paths through `argo_app/src/config.py` or environment overrides, and document any port updates so downstream clients expecting `ws://localhost:8765` or the 8090 diagnostics endpoint stay aligned.
