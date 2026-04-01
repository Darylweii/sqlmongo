from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List


APP_VERSION = "0.2.0"
RELEASE_DATE = "2026-04-01"
RELEASE_NAME = "Hybrid Resolution & User Memory"

CHANGELOG_ENTRIES: List[Dict[str, Any]] = [
    {
        "version": APP_VERSION,
        "release_date": RELEASE_DATE,
        "title": RELEASE_NAME,
        "summary": "设备混合解析、聊天式长期记忆、前端诊断信息与作用域性能保护完成一轮产品化整合。",
        "highlights": [
            "新增词法优先、语义补充的混合设备解析主路径，并统一推荐后确认交互。",
            "新增 SQLite 长期记忆能力，支持聊天里添加、查看、删除常用叫法。",
            "补齐记忆改写与归一化模块，支持规则优先、可选 LLM 兜底改写后再写入记忆库。",
            "前端新增当前用户标识，修复管理页与聊天页 user_id 不一致导致的长期记忆不生效问题。",
            "作用域卡片增加大结果集抑制逻辑，超过阈值时只保留摘要，避免超大渲染开销。",
            "补充语义映射管理页、回归文档、系统流程展示页与多项回归测试。",
        ],
    },
    {
        "version": "0.1.0",
        "release_date": "2026-03-27",
        "title": "Initial Internal Release",
        "summary": "提供基础的设备查询、时序数据检索、表格展示与智能分析能力。",
        "highlights": [
            "支持设备、电流、电压、电量等常见自然语言查询。",
            "支持 MongoDB 查询、分页、统计、智能分析与图表展示。",
            "支持候选设备澄清、SSE 流式回答与前端执行过程面板。",
        ],
    },
]


def build_version_payload() -> Dict[str, Any]:
    latest = CHANGELOG_ENTRIES[0]
    return {
        "version": APP_VERSION,
        "release_date": RELEASE_DATE,
        "release_name": RELEASE_NAME,
        "summary": latest.get("summary", ""),
        "highlights": list(latest.get("highlights") or []),
        "history": deepcopy(CHANGELOG_ENTRIES),
    }
