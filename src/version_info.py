from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List


APP_VERSION = "0.3.0"
RELEASE_DATE = "2026-04-03"
RELEASE_NAME = "Chart Follow-up MVP & UI Cleanup"

CHANGELOG_ENTRIES: List[Dict[str, Any]] = [
    {
        "version": APP_VERSION,
        "release_date": RELEASE_DATE,
        "title": RELEASE_NAME,
        "summary": "完成图表续问 MVP、历史缓存恢复、首页测试题库分组，以及前端快捷查询下线与历史对话清理能力。",
        "highlights": [
            "新增图表规划器 + 注册表 + line/bar/scatter/boxplot/heatmap 五种 ECharts builder 主路径。",
            "补齐“帮我画图/切换图种”续问链路，支持 session 缓存与聊天历史恢复后继续出图。",
            "首页右侧改为分类测试题库，便于按基础、对比、趋势、图表等场景一键回归。",
            "移除前端快捷查询面板与独立结果区，统一回到聊天式查询入口，减少重复交互。",
            "新增清空历史对话接口与前端按钮，可一键清理当前用户聊天记录而不影响常用叫法记忆。",
            "补充缓存恢复、历史续图与接口回归测试，修复 SSE 追问场景的稳定性问题。",
        ],
    },
    {
        "version": "0.2.0",
        "release_date": "2026-04-01",
        "title": "Hybrid Resolution & User Memory",
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
