import os
import sys

import uvicorn


def _ensure_utf8_output() -> None:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")
    if os.name == "nt":
        os.system("chcp 65001 > nul")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


if __name__ == "__main__":
    _ensure_utf8_output()
    print("=" * 60)
    print("  AI 数据查询助手 - 后端 API 服务")
    print("=" * 60)
    print()
    print("  API 地址: http://localhost:8080/api")
    print()
    print("  按 Ctrl+C 停止服务")
    print("=" * 60)
    uvicorn.run("web.app:app", host="0.0.0.0", port=8080, reload=True)
