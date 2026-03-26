# 手工测试目录说明

- `30道测试题目.md`：通用问答样例。
- `accuracy_priority_top10.md`：准确性高优先级测试清单。
- `test_comprehensive.py`、`test_20_questions.py`、`test_with_ground_truth.py`、`test_with_validation.py`、`test_pagination_backend.py`：当前仍在使用的手工验证脚本。
- `legacy/`：历史脚本归档区，用于存放一次性或旧版链路验证脚本。
- `reports/`：压测和验证报告输出目录。

## 约定

1. 新增手工测试脚本优先放在 `tests/manual/` 根目录。
2. 已废弃或历史脚本移入 `tests/manual/legacy/`。
3. 生成的 JSON/Markdown 报告统一放在 `tests/manual/reports/`。
4. 如需从旧路径运行脚本，优先使用迁移后的兼容入口或直接指向 `legacy/` 中的实际脚本。
