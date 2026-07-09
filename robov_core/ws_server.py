import asyncio
import json
import threading

import websockets

from .terminal_shell import TerminalShell


async def _writer_task(websocket, shell):
    try:
        while shell.running:
            data = shell.read()
            if data:
                try:
                    await websocket.send(json.dumps({"type": "output", "data": data}))
                except websockets.exceptions.ConnectionClosed:
                    break
            await asyncio.sleep(0.03)
    except asyncio.CancelledError:
        pass


async def handler(websocket):
    shell = TerminalShell()
    writer = None

    try:
        async for raw in websocket:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            t = msg.get("type")

            if t == "connect":
                mode = msg.get("mode", "local")
                if mode == "local":
                    shell.start_local()
                elif mode == "ssh":
                    shell.start_ssh(
                        host=msg["host"],
                        port=int(msg.get("port", 22)),
                        user=msg["user"],
                        password=msg["password"],
                    )
                await websocket.send(json.dumps({"type": "connected", "mode": mode}))
                writer = asyncio.create_task(_writer_task(websocket, shell))

            elif t == "input":
                if shell.running:
                    shell.write(msg["data"])

            elif t == "resize":
                if shell.running:
                    shell.resize(int(msg["cols"]), int(msg["rows"]))

            elif t == "disconnect":
                break

    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception:
        pass
    finally:
        if writer:
            writer.cancel()
        shell.close()


async def _serve_forever(host, port):
    async with websockets.serve(handler, host, port):
        await asyncio.Future()


def start(host="0.0.0.0", port=8765):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_serve_forever(host, port))


def start_thread(host="0.0.0.0", port=8765):
    t = threading.Thread(target=start, args=(host, port), daemon=True)
    t.start()
    return t
