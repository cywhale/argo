#### ver 0.0.1 create argo_app.py
#### ver 0.0.2 endpoint: /api/floats/download, try websocket notify (not yet)
#### ver 0.0.3 move to argo_app and setup.py for package/Docker (work for net=host)/PyInstaller(not yet)
    -- create window installer by Inno Setup
	-- Make window installer can use Conda
	-- Installer/uninstall works for Conda/Python

#### ver 0.0.4 More customized functions/n1: adjustable running port
    -- modify Inno Setup to check_port feed %PORT% into odbargo_app
    -- add odbargo_app-0.0.4.installer.zip (.exe) in dist/ for downloading test

#### ver 0.0.5 Handle weekly update logic to POSTGIS db/n1
    -- convert numpy types to python natvie types when writing into db
    -- change build workflow, using pyproject.toml/drop some dummy columns in argofloats db
    -- update window installer (Inno Setup), reduce dummy file size greatly(not yet)/n1
    -- pyproject.toml build workflow trials (not yet)/n2
    -- drop pyproject.toml, use setup.py (python -m build) to fix no entry: No module named argo_app/n3
    -- need uvicorn[standard] to include Websocket after v0.30.6
    -- cancel argo_app app.py zipfile (equal size with .nc file)
    -- Filter duplicated WMS/add websocket onclose and error handling
    -- add TIMEOUT for powershell download python installer
    -- Change to BITS download python and fallback to web client method/n1, correctly detect result by enabledelayedexpansion/n2
    -- package upgrade but argopy1.0.0 still have bug (#412, wait official release)/n1
    -- Fix window installer for case without python installed
    -- Fixed argopy==0.1.17 (wait argopy1.0.0 bug fixation official release)/n2 
    -- try to use fixed requirements.txt in odbargo_app istallation (wait argopy1.0.0 bugfix releas)/n3
	-- add requirement.txt into window installer/n4
	-- add Installation guide in README.md/html
	-- Fix profiler all Unknown problem by using R08 reference table in argopy
	-- Research example of WMO690084 (Black Sea, O2 profile)

#### ver 0.0.6 move to argopy 1.1.0 before adding mcp
        -- mcp server trial/n1/deprecated: 20250610

#### ver 0.1.0 odbargo-cli: lightweight CLI deprecate FastAPI, argopy/milestone/n1
        -- try a windows executive
		-- prepare documentation in markdown/html

#### ver 0.1.1 add --insecure more to disable SSL cert if SSL error and retry downloading
#### ver 0.2.0 add odbargo-view plugin + subset workflows
        -- spawn spec-driven `odbargo_view` subprocess for dataset preview/plot/export
        -- extend slash CLI + WS bridge with `/view` help, alias subsets, and binary streaming fixes
        -- cap preview payloads, surface subset keys, and document usage in README_odbargo-view.md
#### ver 0.2.1 spatial-temporal filters + plot window fallback
        -- add `--bbox/--box` and `--start/--end` slash options layered with existing filters across preview/plot/export
        -- compose bbox/time predicates inside plugin filter pipeline and persist them for subset reuse
        -- open non-blocking matplotlib preview window when slash plots omit `--out`
        -- default map plots to LONGITUDE/LATITUDE axes, reshape values onto a lon/lat mesh, and support `--y` (field colour) plus `--cmap`
        -- allow `--order` sorting across preview/plot/export, run CLI exports in the background, enable readline history/navigation, and refresh README_odbargo-view.md with new usage examples
#### ver 0.2.2 optional viewer + packaging updates
        -- add PyInstaller one-file build commands for CLI and view plugin, keeping heavy deps in the companion binary
        -- make view plugin optional (`--plugin auto|view|none`), support explicit binaries, and emit clear messaging when disabled or missing
#### ver 0.2.3 fix matplotlib plot window cannot open (main CLI-VIEW devision and connection/n1)
#### ver 0.2.4 viewer grouping + stability + debug flag
        -- add `--group-by` with discrete keys, numeric binning (`PRES:<binwidth>`), and time resampling (`TIME:<freq>`);
           support `--agg mean|median|max|min|count`, capped legend series via `style.max_series`
        -- allow plotting against coordinate/dimension columns even when subset projection is narrowed; persist coords by default unless `--trim-dims`
        -- enforce `--limit` consistently across preview/plot/export; preview pagination reuses last displayed columns across `--cursor`
        -- single-port WS bridge supports viewer self-registration; robust handover to a single plugin reader (no concurrent recv)
        -- plot/export binary relay hardened; CSV streaming via `file_start` → binary chunks → `file_end` with SHA-256
        -- add env `ODBARGO_DEBUG=1` to enable CLI⇄plugin debug breadcrumbs (suppressed by default)
        -- README_odbargo-view.md refreshed with new options and examples
#### ver 0.2.5 Breaking completely deprecate old FastAPI structure and CLI packages        
        -- ensure_viewer(), and self‑registration stabilized; robust binary forwarding
        -- one-file pyinstaller packaging (.spec) and pip install (.toml)