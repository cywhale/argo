"""Lightweight Argo downloader – exposes ws://localhost:<port>
   Messages:
   → {"type":"ping"} → ← {"type":"pong"}
   → {"type":"start_job", "jobId": str, "wmoList": [int,…]}  # start download
   ← {"type":"job_status", "jobId":…, "status":"running"}
   ← {"type":"job_status", "jobId":…, "status":"success", "resultPath": …}
   ← {"type":"job_status", "jobId":…, "status":"failed", "message": …}
"""

import asyncio, json, argparse, sys, tempfile, urllib.request, urllib.parse, time, ssl
from pathlib import Path
import websockets

frontend_connected = asyncio.Event()
test_counter = 0

def download_argo_data(wmo_list, output_path, insecure=False):
    base_url = "https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats-synthetic-BGC.nc"
    if len(wmo_list) == 1:
        constraint = f'&platform_number=%22{wmo_list[0]}%22'
    else:
        wmo_pattern = "%7C".join(str(wmo) for wmo in wmo_list)
        constraint = f'&platform_number=~%22{wmo_pattern}%22'
    full_url = f"{base_url}?{constraint}"
    print(f"Downloading from: {full_url} {'[INSECURE]' if insecure else ''}")
    try:
        request = urllib.request.Request(full_url)
        request.add_header('User-Agent', 'odbargo-cli/1.0')
        request.add_header('Connection', 'keep-alive')
        request.add_header('Keep-Alive', 'timeout=120, max=1000')
        ssl_ctx = ssl._create_unverified_context() if insecure else ssl.create_default_context()
        with urllib.request.urlopen(request, timeout=300, context=ssl_ctx) as response:
            if response.getcode() != 200:
                raise Exception(f"HTTP {response.getcode()}: {response.reason}")
            content_length = response.headers.get('Content-Length')
            if content_length and int(content_length) < 100:
                error_content = response.read().decode('utf-8')
                raise Exception(f"Server returned error: {error_content}")
            with open(output_path, 'wb') as f:
                f.write(response.read())
        if Path(output_path).stat().st_size < 100:
            raise Exception("Downloaded file is too small, likely an error")
        return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise Exception(f"No data found for WMO {wmo_list}. Check if WMO IDs are valid and have BGC data.")
        else:
            raise Exception(f"HTTP Error {e.code}: {e.reason}")
    except urllib.error.URLError as e:
        raise Exception(f"Network error: {e.reason}")
    except ssl.SSLError as e:
        raise Exception(f"SSL error: {e}")
    except Exception as e:
        raise Exception(f"Download failed: {str(e)}")

def download_with_retry(wmo_list, output_path, retries=3, delay=10, force_insecure=False):
    insecure_next_try = False
    for attempt in range(retries):
        try:
            return download_argo_data(wmo_list, output_path, insecure=(force_insecure or insecure_next_try))
        except Exception as e:
            print(f"⚠️ Attempt {attempt + 1} failed: {e}")
            if "SSL error" in str(e) and attempt == 0:
                insecure_next_try = True

            if attempt < retries - 1:
                print(f"⏳ Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise

async def process_job(ws, job_id, wmo_list, job_number=None, force_insecure=False):
    try:
        print(f"Task {job_number}: {job_id} now downloading {wmo_list}")
        if len(wmo_list) >= 3:
            await ws.send(json.dumps({
                "type": "job_status", "jobId": job_id, "jobNumber": job_number,
                "status": "running", 
                "message": f"⚠️ Warning: Fetching many WMOs may fail on slow networks. You'll be notified when the file is ready. If it failed, consider fewer WMOs."
            }))

        out_dir = tempfile.mkdtemp(prefix="argo_")
        out = Path(out_dir) / f"argo_{'_'.join(map(str,wmo_list))}.nc"
        success = await asyncio.to_thread(download_with_retry, wmo_list, str(out), force_insecure=force_insecure)
        if success:
            print(f"Download OK: {out}")
            await ws.send(json.dumps({
                "type": "job_status", "jobId": job_id, "jobNumber": job_number,
                "status": "success", "resultPath": str(out),
                "message": f"✅ Task {job_number} success. File: {out}"
            }))
        else:
            raise Exception("Download failed")
    except Exception as e:
        await ws.send(json.dumps({
            "type": "job_status", "jobId": job_id, "jobNumber": job_number,
            "status": "failed", "message": f"❌ Task {job_number} failed. Error: {str(e)}"
        }))

async def handler(websocket):
    async for msg in websocket:
        print("Received websocket message:", msg)
        try:
            data = json.loads(msg)
        except json.JSONDecodeError:
            continue
        if data.get("type") == "ping":
            await websocket.send(json.dumps({"type": "pong"}))
            frontend_connected.set()
        elif data.get("type") == "start_job":
            asyncio.create_task(process_job(
                websocket, data["jobId"], data["wmoList"], data.get("jobNumber"), force_insecure=args.insecure
            ))

async def cli_interactive_mode():
    global test_counter
    print("\n🟡 CLI interactive mode. Type WMO list (comma separated) or 'exit':")
    while True:
        query = await asyncio.to_thread(input, ">>> ")
        if query.strip().lower() == "exit":
            print("Goodbye.")
            break
        query_wmo = ''.join([c if c.isdigit() or c == ',' else '' for c in query])
        wmo_list = [int(x) for x in query_wmo.split(',') if x.strip().isdigit()]
        if wmo_list:
            test_counter += 1
            job_id = f"test-{test_counter}"
            await process_job(FakeSocket(), job_id, wmo_list, job_number=test_counter, force_insecure=args.insecure)
        else:
            print("⚠️  No valid WMO IDs found. Try again.")

class FakeSocket:
    async def send(self, message):
        msg = json.loads(message)
        print(f"[CLI] {msg['status'].upper()} {msg.get('message','')}")
        if "resultPath" in msg:
            print(f"Saved to: {msg['resultPath']}")

async def run_server_and_cli(port):
    server = await websockets.serve(handler, "localhost", port)
    print(f"[Argo-CLI] WebSocket listening on ws://localhost:{port}")
    asyncio.create_task(cli_interactive_mode())
    await server.wait_closed()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765, help="Port to serve WebSocket")
    parser.add_argument("--insecure", action='store_true', help="Force disable SSL cert verification (not recommended)")
    args = parser.parse_args()
    try:
        asyncio.run(run_server_and_cli(args.port))
    except KeyboardInterrupt:
        sys.exit(0)

