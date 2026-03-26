from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


if __name__ == "__main__":
    web_dir = Path(__file__).resolve().parent / "web"
    handler = partial(SimpleHTTPRequestHandler, directory=str(web_dir))
    server = ThreadingHTTPServer(("0.0.0.0", 3000), handler)
    print("=" * 60)
    print("  AI ?????? - ??????")
    print("=" * 60)
    print()
    print("  ????: http://localhost:3000")
    print()
    print("  ? Ctrl+C ????")
    print("=" * 60)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
