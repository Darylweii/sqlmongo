import uvicorn


if __name__ == "__main__":
    print("=" * 60)
    print("  AI ?????? - ?? API ??")
    print("=" * 60)
    print()
    print("  API ??: http://localhost:8080/api")
    print()
    print("  ? Ctrl+C ????")
    print("=" * 60)
    uvicorn.run("web.app:app", host="0.0.0.0", port=8080, reload=True)
