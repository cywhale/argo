# Repository Guidelines

## Project Structure & Module Organization
Source lives under `argo_app/`: `app.py` runs the FastAPI service and WebSocket handlers, while `src/config.py` centralises cache, output, and port defaults. The standalone downloader lives in `odbargo-cli.py`, with console scripts declared in `setup.py`. Packaging artefacts sit in `dist/`, helper scripts in `scripts/`, configuration samples in `conf/`, and exploratory notebooks in `dev/`.

## Build, Test, and Development Commands
Create an isolated environment before contributing:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .  # installs CLI and FastAPI app locally
```
Install dependencies quickly with `pip install -r requirements.txt`. Run the websocket CLI via `python odbargo-cli.py --port 8765`; the binaries in `dist/` behave the same. Launch the API server with `uvicorn argo_app.app:app --reload --port 8090`. Release builds for the bundled executables come from `scripts/build_odbargo-cli.sh` (Windows) and `scripts/build_argopy_cli.sh` (Linux).

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation and snake_case symbols. Keep routers, Pydantic models, and background workers in focused helpers, and centralise configurable constants in `argo_app/src/config.py`. Prefer structured logging over prints when extending the service, and align CLI arguments with existing long options (`--port`, `--insecure`).

## Testing Guidelines
There is no automated suite today; add `pytest`-style tests under `tests/test_*.py` so they are CI-ready. Mock `DataFetcher` responses or patch network calls to avoid ERDDAP traffic. After edits, hit `curl http://localhost:8090/argo/api/test` and trigger a small CLI download (e.g., `python odbargo-cli.py 5903377`). Note any manual verification in your PR until automation arrives.

## Commit & Pull Request Guidelines
Commits use short, imperative summaries (e.g., `add --insecure flag retry logic`). Group related changes and avoid checking in build artefacts. Pull requests should call out the user-facing impact, link issues, and include logs or screenshots showing both CLI and API downloads still succeed. Document reproduction steps and any configuration tweaks (`config.outputPath`, ports) for reviewers.

## Release & Configuration Notes
When changing cache or output behaviour, update `argo_app/src/config.py` and mirror instructions in `README.md`. Keep version bumps in `setup.py` and `version.txt` aligned, and refresh installers under `scripts/win_installer/` whenever CLI flags change. Never commit credentials; load secrets from untracked `.env` files before launching the server.
