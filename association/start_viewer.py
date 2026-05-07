from __future__ import annotations

import argparse
import json
import os
import socket
import threading
import webbrowser
from datetime import datetime
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
VIEWER_URL_PATH = "viewer/index.html"
INFO_FILE = ROOT_DIR / "viewer" / "server_info.json"


def pick_port(preferred: int) -> int:
    if preferred > 0:
        return preferred

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def save_server_info(url: str, port: int) -> None:
    INFO_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "url": url,
        "port": port,
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cwd": str(ROOT_DIR),
    }
    INFO_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="启动轨迹前端页面（自动分配端口）")
    parser.add_argument("--port", type=int, default=0, help="端口号，默认 0 为自动分配")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="监听地址")
    parser.add_argument("--no-browser", action="store_true", help="仅启动服务，不自动打开浏览器")
    args = parser.parse_args()

    port = pick_port(args.port)

    os.chdir(ROOT_DIR)
    server = ThreadingHTTPServer((args.host, port), SimpleHTTPRequestHandler)

    url = f"http://{args.host}:{port}{VIEWER_URL_PATH}"
    save_server_info(url, port)

    print(f"前端已启动: {url}")
    print(f"端口信息文件: {INFO_FILE}")

    if not args.no_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止服务")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
