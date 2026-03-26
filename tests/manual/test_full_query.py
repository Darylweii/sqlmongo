"""
兼容入口：旧版完整查询脚本已迁移到 `tests/manual/legacy/test_full_query.py`。
"""
from pathlib import Path
import runpy


if __name__ == "__main__":
    legacy_path = Path(__file__).with_name("legacy") / "test_full_query.py"
    runpy.run_path(str(legacy_path), run_name="__main__")
