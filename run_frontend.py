import os
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def _ensure_utf8_output() -> None:
    if os.name == "nt":
        os.system("chcp 65001 > nul")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


if __name__ == "__main__":
    _ensure_utf8_output()
    web_dir = Path(__file__).resolve().parent / "web"
    handler = partial(SimpleHTTPRequestHandler, directory=str(web_dir))
    server = ThreadingHTTPServer(("0.0.0.0", 3000), handler)
    print("=" * 60)
    print("  AI 数据查询助手 - 前端静态服务")
    print("=" * 60)
    print()
    print("  页面地址: http://localhost:3000")
    print()
    print("  按 Ctrl+C 停止服务")
    print("=" * 60)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
