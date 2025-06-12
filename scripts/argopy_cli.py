"""Lightweight Argo downloader – exposes ws://localhost:<port>
   Messages:
   → {"type":"ping"} → ← {"type":"pong"}
   → {"type":"start_job", "jobId": str, "wmoList": [int,…]}  # start download
   ← {"type":"job_status", "jobId":…, "status":"running"}
   ← {"type":"job_status", "jobId":…, "status":"success", "resultPath": …}
   ← {"type":"job_status", "jobId":…, "status":"failed", "message": …}
"""
import asyncio, json, argparse, sys, tempfile
from pathlib import Path
from argopy import DataFetcher
import websockets

frontend_connected = asyncio.Event()
test_counter = 0

async def process_job(ws, job_id, wmo_list, job_number=None):
    # await ws.send(json.dumps({"type":"job_status","jobId":job_id,"status":"running"}))
    try:
        print(f"Task {job_number}: {job_id} now downloading {wmo_list}")
        ds = DataFetcher(ds="bgc", mode="expert", src="erddap", params="all", parallel=True) \
            .float(wmo_list).to_xarray()
        out_dir = tempfile.mkdtemp(prefix="argo_")
        out = Path(out_dir) / f"argo_{'_'.join(map(str,wmo_list))}.nc"
        ds.to_netcdf(out)
        print(f"Download OK: {out}")
        await ws.send(json.dumps({
            "type": "job_status", "jobId": job_id, "jobNumber": job_number,
            "status": "success", "resultPath": str(out),
            "message": f"✅ Task {job_number} success. File: {out}"
        }))
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
                websocket, data["jobId"], data["wmoList"], data.get("jobNumber")
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
            job_id = f"test-{test_counter}" #str(uuid.uuid4())
            await process_job(FakeSocket(), job_id, wmo_list, job_number=test_counter)
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
    args = parser.parse_args()
    try:
        asyncio.run(run_server_and_cli(args.port))
    except KeyboardInterrupt:
        sys.exit(0)
