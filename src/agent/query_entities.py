"""Query entity parser.

Local fallback parser for QueryPlan extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Optional, Tuple

CURRENT_QUESTION_MARKER = "\u5f53\u524d\u95ee\u9898:"
DEVICE_CODE_PATTERN = re.compile(r"(?<![A-Za-z0-9_])[a-zA-Z]\d*_[a-zA-Z0-9_]+(?![A-Za-z0-9_])")
LEADING_VERB_PATTERN = re.compile(
    r"^(?:\u8bf7|\u5e2e\u6211|\u9ebb\u70e6|\u8bf7\u5e2e\u6211|\u9ebb\u70e6\u5e2e\u6211)?\s*(?:\u67e5\u8be2|\u67e5\u4e00\u4e0b|\u67e5\u4e0b|\u770b\u770b|\u770b\u4e0b|\u6bd4\u8f83|\u5bf9\u6bd4|\u5206\u6790\u4e00\u4e0b|\u5206\u6790|\u7edf\u8ba1\u4e00\u4e0b|\u7edf\u8ba1|\u5e2e\u5fd9\u67e5\u4e0b)\s*",
    re.IGNORECASE,
)
LEADING_LISTING_PATTERN = re.compile(
    r"^(?:\u6709(?:\u54ea(?:\u4e9b|\u51e0)|\u4ec0\u4e48)|\u54ea(?:\u4e9b|\u51e0)|\u4ec0\u4e48|\u5217\u51fa|\u5217\u4e00\u4e0b|\u67e5\u627e|\u641c\u7d22|\u5339\u914d|\u67e5\u770b|\u67e5\u8be2|\u786e\u8ba4(?:\u4e00\u4e0b)?)\s*",
    re.IGNORECASE,
)
COMPARISON_SPLIT_PATTERN = re.compile(r"\b(?:vs|pk)\b|\u5bf9\u6bd4|\u6bd4\u8f83|\u4ee5\u53ca|\u548c|\u4e0e|\u8ddf|\u53ca|\u3001|,|\uff0c|;|\uff1b|/", re.IGNORECASE)
CONTEXT_SPLIT_PATTERN = re.compile(r"\s*(?:\u91cc\u7684|\u4e2d\u7684|\u5185\u7684|\u4e0b\u7684|\u91cc|\u4e2d|\u5185|\u4e0b)\s*", re.IGNORECASE)
PROJECT_HINT_PATTERN = re.compile(r"[\u4e00-\u9fffA-Za-z0-9_#()\uff08\uff09\-\u00b7\s]{2,40}(?:\u9879\u76ee|\u5e73\u53f0|\u7cfb\u7edf)")
EXPLICIT_DATE_PATTERN = re.compile(r"(?:\d{4}[-/\u5e74]\d{1,2}(?:[-/\u6708]\d{1,2}\u65e5?)?|\d{4}\u5e74\d{1,2}\u6708)")
TRAILING_TIME_PATTERN = re.compile(
    r"\s*(?:\u6700\u8fd1|\u8fd1\u4e00\u5468|\u8fd17\u5929|\u6700\u8fd1\u4e00\u5468|\u6700\u8fd17\u5929|\u6700\u8fd1\u4e00\u4e2a\u6708|\u8fd1\u4e00\u4e2a\u6708|\u672c\u5468|\u4e0a\u5468|\u4eca\u5929|\u6628\u65e5|\u6628\u5929|\u672c\u6708|\u4e0a\u6708|\u4eca\u5e74|\u53bb\u5e74|\u672c\u5b63\u5ea6|\u4e0a\u5b63\u5ea6).*$",
    re.IGNORECASE,
)
TRAILING_METRIC_PATTERN = re.compile(
    r"\s*(?:\u7684)?(?:\u7528\u7535\u91cf|\u7528\u80fd\u91cf|\u7d2f\u8ba1\u7535\u91cf|\u7d2f\u8ba1\u80fd\u8017|\u6709\u529f\u529f\u7387|\u65e0\u529f\u529f\u7387|\u89c6\u5728\u529f\u7387|\u7ebf\u7535\u538b|\u76f8\u7535\u538b|\u529f\u7387\u56e0\u6570|\u6e29\u5ea6|\u7528\u7535|\u7528\u80fd|\u7535\u91cf|\u8017\u7535|\u80fd\u8017|\u529f\u7387|\u7535\u6d41|\u7535\u538b|\u9891\u7387)(?:\u60c5\u51b5|\u6570\u636e|\u8d8b\u52bf|\u6ce2\u52a8|\u5206\u6790|\u5bf9\u6bd4|\u6bd4\u8f83|\u53d8\u5316|\u8bb0\u5f55)?$",
    re.IGNORECASE,
)
TRAILING_DETECT_PATTERN = re.compile(
    r"\s*(?:\u6709(?:\u54ea(?:\u4e9b|\u51e0)|\u4ec0\u4e48)|\u54ea(?:\u4e9b|\u51e0)|\u4ec0\u4e48)?(?:\u6570\u636e\u7c7b\u578b|\u7c7b\u578b\u7684\u6570\u636e|\u6570\u636e\u79cd\u7c7b|\u6570\u636e\u79cd\u7c7b|\u6570\u636e\u9879|\u6570\u636e\u6307\u6807|\u6307\u6807\u7c7b\u578b|\u6d4b\u70b9(?:\u7c7b\u578b)?)\s*[\uff1f?]?$",
    re.IGNORECASE,
)
TRAILING_DECISION_PATTERN = re.compile(
    r"\s*(?:\u662f)?(?:\u4e0a\u5347|\u4e0b\u964d|\u589e\u957f|\u51cf\u5c11|\u504f\u9ad8|\u504f\u4f4e|\u6b63\u5e38)(?:\u8fd8\u662f|\u6216)?(?:\u4e0a\u5347|\u4e0b\u964d|\u589e\u957f|\u51cf\u5c11|\u504f\u9ad8|\u504f\u4f4e|\u6b63\u5e38)?\s*[\uff1f?]?$",
    re.IGNORECASE,
)
TRAILING_GENERIC_PATTERN = re.compile(r"\s*(?:\u7684\u6570\u636e|\u60c5\u51b5|\u8d8b\u52bf|\u6ce2\u52a8|\u5206\u6790|\u5bf9\u6bd4|\u6bd4\u8f83|\u53d8\u5316|\u8bb0\u5f55|\u8d70\u52bf)$", re.IGNORECASE)
CONTAINS_TARGET_PATTERN = re.compile(
    r"(?:\u8bbe\u5907\u540d\u79f0|\u8bbe\u5907\u540d|\u540d\u79f0|\u540d\u5b57|\u4ee3\u53f7|\u7f16\u7801|\u7f16\u53f7)?\s*(?:\u4e2d)?(?:\u5305\u542b|\u542b\u6709|\u5e26\u6709|\u91cc\u6709)\s*(?P<target>[A-Za-z0-9_#\-\u4e00-\u9fff]+)",
    re.IGNORECASE,
)
LEADING_CONTAINS_PATTERN = re.compile(
    r"^(?:(?:\u8bbe\u5907\u540d\u79f0|\u8bbe\u5907\u540d|\u540d\u79f0|\u540d\u5b57|\u4ee3\u53f7|\u7f16\u7801|\u7f16\u53f7)\s*)?(?:\u4e2d)?(?:\u5305\u542b|\u542b\u6709|\u5e26\u6709|\u91cc\u6709)\s*",
    re.IGNORECASE,
)
GENERIC_EMPTY_TOKENS = {
    "\u8bbe\u5907", "\u6570\u636e", "\u60c5\u51b5", "\u8d8b\u52bf", "\u6ce2\u52a8", "\u5206\u6790", "\u5bf9\u6bd4", "\u6bd4\u8f83", "\u53d8\u5316", "\u8bb0\u5f55", "\u8d70\u52bf", "\u67e5\u8be2", "\u67e5\u4e00\u4e0b", "\u67e5\u4e0b",
    "\u6709\u54ea\u4e9b", "\u54ea\u4e9b", "\u4ec0\u4e48", "\u5217\u51fa", "\u5217\u4e00\u4e0b",
}
SENSOR_KEYWORDS = (
    "\u7528\u7535", "\u7528\u80fd", "\u7535\u91cf", "\u8017\u7535", "\u80fd\u8017", "\u529f\u7387", "\u7535\u6d41", "\u7535\u538b", "\u7ebf\u7535\u538b", "\u76f8\u7535\u538b", "\u8d8b\u52bf", "\u6ce2\u52a8", "\u5cf0\u8c37", "\u5f02\u5e38", "\u6570\u636e", "\u60c5\u51b5", "\u529f\u7387\u56e0\u6570", "\u9891\u7387", "\u6e29\u5ea6",
)
TIME_KEYWORDS = (
    "\u6700\u8fd1", "\u8fd1\u4e00\u5468", "\u8fd17\u5929", "\u6700\u8fd1\u4e00\u5468", "\u6700\u8fd17\u5929", "\u6700\u8fd1\u4e00\u4e2a\u6708", "\u8fd1\u4e00\u4e2a\u6708", "\u672c\u5468", "\u4e0a\u5468", "\u4eca\u5929", "\u6628\u65e5", "\u6628\u5929", "\u672c\u6708", "\u4e0a\u6708", "\u4eca\u5e74", "\u53bb\u5e74", "\u672c\u5b63\u5ea6", "\u4e0a\u5b63\u5ea6",
)
PAGINATION_KEYWORDS = ("\u6bcf\u9875", "\u5206\u9875", "page_size", "page size", "\u7b2c1\u9875", "\u524d50\u6761", "50\u6761")
DEVICE_LISTING_KEYWORDS = ("\u8bbe\u5907\u5217\u8868", "\u5217\u51fa\u8bbe\u5907", "\u5217\u4e00\u4e0b\u8bbe\u5907", "\u6709\u54ea\u4e9b\u8bbe\u5907", "\u641c\u7d22\u8bbe\u5907", "\u67e5\u627e\u8bbe\u5907", "\u5339\u914d\u8bbe\u5907")
DEVICE_LISTING_PATTERN = re.compile(
    r"(?:\u641c\u7d22|\u67e5\u627e|\u67e5\u8be2|\u5339\u914d|\u5217\u51fa|\u5217\u4e00\u4e0b|\u770b\u4e00\u4e0b|\u770b\u770b).{0,20}\u8bbe\u5907|"
    r"\u8bbe\u5907.{0,12}(?:\u5217\u8868|\u6e05\u5355|\u6709\u54ea\u4e9b|\u54ea\u4e9b|\u5339\u914d|\u641c\u7d22\u7ed3\u679c)",
    re.IGNORECASE,
)
PROJECT_DEVICE_LISTING_PATTERN = re.compile(
    r"(?P<project>[\u4e00-\u9fffA-Za-z0-9_#()\uff08\uff09\-\u00b7\s]{2,40}?(?:\u9879\u76ee|\u5e73\u53f0|\u7cfb\u7edf))\s*(?:\u91cc\u6709|\u4e2d\u6709|\u4e0b\u6709|\u6709)?\s*(?:\u54ea\u4e9b|\u4ec0\u4e48|\u5217\u51fa|\u67e5\u770b)?\s*\u8bbe\u5907",
    re.IGNORECASE,
)

PROJECT_LISTING_KEYWORDS = ("\u6709\u54ea\u4e9b", "\u54ea\u51e0\u4e2a", "\u5217\u51fa", "\u5217\u8868", "\u6240\u6709", "\u5168\u90e8", "\u53ef\u7528")
PROJECT_STATS_KEYWORDS = ("\u6392\u540d", "\u6700\u591a", "\u6700\u5c11", "\u6570\u91cf", "\u8bbe\u5907\u6570", "\u8bbe\u5907\u6570\u91cf", "\u54ea\u4e2a\u9879\u76ee", "\u5404\u9879\u76ee", "\u6bcf\u4e2a\u9879\u76ee")
DETECT_DATA_TYPE_KEYWORDS = (
    "\u6709\u54ea\u4e9b\u6570\u636e", "\u6709\u4ec0\u4e48\u6570\u636e", "\u6570\u636e\u7c7b\u578b", "\u54ea\u4e9b\u6570\u636e\u8bb0\u5f55", "\u6709\u54ea\u4e9b\u7c7b\u578b\u7684\u6570\u636e", "\u6709\u54ea\u4e9b\u7c7b\u578b", "\u652f\u6301\u4ec0\u4e48\u6570\u636e", "\u6709\u54ea\u51e0\u79cd\u6570\u636e",
)
COMPARISON_HINT_KEYWORDS = ("\u6bd4\u8f83", "\u5bf9\u6bd4", "vs", "pk", "\u54ea\u4e2a\u66f4", "\u6392\u884c", "\u5dee\u5f02")
RANK_HIGH_HINT_KEYWORDS = ("\u6700\u9ad8", "\u6700\u5927", "\u6700\u591a", "\u5cf0\u503c", "top")
RANK_LOW_HINT_KEYWORDS = ("\u6700\u4f4e", "\u6700\u5c0f", "\u6700\u5c11", "\u8c37\u503c")
RANK_POINT_HINT_KEYWORDS = ("\u65f6\u95f4\u70b9", "\u65f6\u523b", "\u6570\u636e\u70b9", "\u8bb0\u5f55", "\u6761\u8bb0\u5f55", "\u6761\u6570\u636e", "\u4e2a\u65f6\u523b")
RANK_DAY_HINT_KEYWORDS = ("\u6309\u5929", "\u6309\u65e5", "\u6bcf\u5929", "\u6bcf\u65e5", "\u5929\u7ef4\u5ea6")
RANK_HOUR_HINT_KEYWORDS = ("\u6309\u5c0f\u65f6", "\u6bcf\u5c0f\u65f6", "\u5c0f\u65f6", "\u65f6\u7ef4\u5ea6")
RANK_WEEK_HINT_KEYWORDS = ("\u6309\u5468", "\u6bcf\u5468", "\u5468\u7ef4\u5ea6", "\u6bcf\u661f\u671f", "\u5468\u6c47\u603b")
RANK_MONTH_HINT_KEYWORDS = ("\u6309\u6708", "\u6bcf\u6708", "\u6708\u7ef4\u5ea6", "\u6708\u6c47\u603b")
RANK_LIMIT_PATTERNS = (
    re.compile(r"\u524d\s*(?P<count>\d{1,3}|[\u96f6\u4e00\u4e8c\u4e24\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341]{1,3})\s*\u4e2a?"),
    re.compile(r"top\s*(?P<count>\d{1,3})", re.IGNORECASE),
    re.compile(r"(?P<count>\d{1,3}|[\u96f6\u4e00\u4e8c\u4e24\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341]{1,3})\s*\u6761(?:\u8bb0\u5f55|\u6570\u636e)?"),
)
TREND_DECISION_KEYWORDS = (
    "\u4e0a\u5347\u8fd8\u662f\u4e0b\u964d",
    "\u4e0a\u6da8\u8fd8\u662f\u4e0b\u8dcc",
    "\u589e\u957f\u8fd8\u662f\u51cf\u5c11",
    "\u5347\u9ad8\u8fd8\u662f\u964d\u4f4e",
    "\u662f\u5347\u8fd8\u662f\u964d",
    "\u8d8b\u52bf\u5982\u4f55",
    "\u8d70\u52bf\u5982\u4f55",
    "\u662f\u4e0a\u5347\u5417",
    "\u662f\u4e0b\u964d\u5417",
)
ANOMALY_POINT_KEYWORDS = (
    "异常",
    "异常点",
    "异常时间点",
    "异常时段",
    "离群",
    "离群点",
    "异常用电",
)
ANOMALY_TIME_HINT_KEYWORDS = ("时间点", "时点", "时刻", "时段", "记录")

EXACT_METRIC_TAG_PATTERNS = (
    ("ua", (r"(?<![a-z0-9])ua(?![a-z0-9])", "a相电压", "a相电压值")),
    ("ub", (r"(?<![a-z0-9])ub(?![a-z0-9])", "b相电压", "b相电压值")),
    ("uc", (r"(?<![a-z0-9])uc(?![a-z0-9])", "c相电压", "c相电压值")),
    ("ia", (r"(?<![a-z0-9])ia(?![a-z0-9])", "a相电流", "a相电流值")),
    ("ib", (r"(?<![a-z0-9])ib(?![a-z0-9])", "b相电流", "b相电流值")),
    ("ic", (r"(?<![a-z0-9])ic(?![a-z0-9])", "c相电流", "c相电流值")),
    ("uab", (r"(?<![a-z0-9])uab(?![a-z0-9])", "ab线电压", "ab相间电压")),
    ("ubc", (r"(?<![a-z0-9])ubc(?![a-z0-9])", "bc线电压", "bc相间电压")),
    ("uca", (r"(?<![a-z0-9])uca(?![a-z0-9])", "ca线电压", "ca相间电压")),
)

METRIC_SYNONYMS = (
    ("u_phase", ("相间电压", "线间电压")),
    ("u_line", ("三相电压", "线电压", "相电压", "电压", "voltage")),
    ("i", ("三相电流", "电流", "安培", "current")),
    ("p", ("有功功率", "无功功率", "视在功率", "功率", "power")),
    ("qf", ("功率因数", "功因", "pf", "cos", "power factor")),
    ("t", ("温度", "环境温度", "室温", "temperature")),
    ("ep", ("累计电量", "累计能耗", "用电量", "用电", "用能", "电量", "耗电", "能耗", "energy")),
)


@dataclass(frozen=True)
class ParsedQueryEntities:
    current_question: str
    explicit_device_codes: Tuple[str, ...]
    search_targets: Tuple[str, ...]
    project_hints: Tuple[str, ...]
    query_mode: str
    inferred_data_type: Optional[str]
    has_sensor_intent: bool
    has_detect_data_types_intent: bool
    has_project_listing_intent: bool
    has_project_stats_intent: bool
    has_device_listing_intent: bool
    has_comparison_intent: bool
    has_pagination_intent: bool
    has_time_reference: bool
    has_ranked_point_intent: bool
    ranking_order: Optional[str]
    ranking_limit: Optional[int]
    ranking_granularity: Optional[str]
    has_trend_decision_intent: bool
    has_anomaly_point_intent: bool


AGGREGATE_SCOPE_TERMS = ("汇总", "合并", "合计", "统一汇总", "整体汇总", "一并汇总")
AGGREGATE_SCOPE_QUANTIFIERS = ("所有", "全部")


def allows_explicit_multi_scope_aggregation(user_query: str, target: str) -> bool:
    current_question = extract_current_question_text(user_query)
    normalized_question = re.sub(r"\s+", "", str(current_question or "").lower())
    normalized_target = re.sub(r"\s+", "", str(target or "").lower())
    if not normalized_question or not normalized_target or normalized_target not in normalized_question:
        return False
    if not any(term in normalized_question for term in AGGREGATE_SCOPE_TERMS):
        return False
    window_patterns = [
        rf"(?:所有|全部)[^，。；、,.;]{{0,6}}{re.escape(normalized_target)}",
        rf"{re.escape(normalized_target)}[^，。；、,.;]{{0,6}}(?:所有|全部)",
    ]
    return any(re.search(pattern, normalized_question, re.IGNORECASE) for pattern in window_patterns)


def extract_current_question_text(user_query: str) -> str:
    if not user_query:
        return ""
    if CURRENT_QUESTION_MARKER in user_query:
        return user_query.rsplit(CURRENT_QUESTION_MARKER, 1)[-1].strip()
    return str(user_query).strip()


@lru_cache(maxsize=2048)
def parse_query_entities(user_query: str) -> ParsedQueryEntities:
    current_question = extract_current_question_text(user_query)
    normalized_lower = current_question.lower()
    compact_lower = re.sub(r"\s+", "", normalized_lower)
    explicit_device_codes = _extract_explicit_device_codes(current_question)
    requested_metric_tags = extract_requested_metric_tags(current_question)
    inferred_data_type = _infer_data_type_from_requested_tags(requested_metric_tags) or _infer_data_type(compact_lower)
    has_time_reference = any(keyword in current_question for keyword in TIME_KEYWORDS) or bool(EXPLICIT_DATE_PATTERN.search(current_question))
    has_sensor_intent = bool(inferred_data_type) or any(keyword in current_question for keyword in SENSOR_KEYWORDS) or has_time_reference
    has_detect_data_types_intent = any(keyword in compact_lower for keyword in DETECT_DATA_TYPE_KEYWORDS)
    has_comparison_intent = len(explicit_device_codes) > 1 or any(keyword in compact_lower for keyword in COMPARISON_HINT_KEYWORDS)
    has_pagination_intent = any(keyword in current_question for keyword in PAGINATION_KEYWORDS)
    project_scoped_device_hint = _extract_project_scoped_device_listing_hint(current_question)
    has_device_listing_intent = (not has_sensor_intent) and (
        _has_explicit_device_listing_language(current_question, compact_lower)
        or bool(project_scoped_device_hint)
    )
    has_project_listing_intent = (not has_sensor_intent) and (not has_device_listing_intent) and ("\u9879\u76ee" in current_question) and any(keyword in compact_lower for keyword in PROJECT_LISTING_KEYWORDS)
    has_project_stats_intent = (not has_sensor_intent) and ("\u9879\u76ee" in current_question) and any(keyword in compact_lower for keyword in PROJECT_STATS_KEYWORDS)
    ranking_order, ranking_limit, ranking_granularity, has_ranked_point_intent = _extract_ranked_point_semantics(current_question, compact_lower)
    ranking_granularity = ranking_granularity or _extract_bucket_granularity_hint(current_question)
    has_trend_decision_intent = _extract_trend_decision_intent(current_question, compact_lower)
    has_anomaly_point_intent = _extract_anomaly_point_intent(current_question, compact_lower)
    search_targets = _extract_search_targets(current_question, has_comparison_intent, explicit_device_codes)
    project_hints = _extract_project_hints(current_question, explicit_device_codes, search_targets)
    query_mode = _infer_query_mode(
        has_sensor_intent=has_sensor_intent,
        has_detect_data_types_intent=has_detect_data_types_intent,
        has_project_listing_intent=has_project_listing_intent,
        has_project_stats_intent=has_project_stats_intent,
        has_device_listing_intent=has_device_listing_intent,
        has_comparison_intent=has_comparison_intent,
        has_ranked_point_intent=has_ranked_point_intent,
        ranking_granularity=ranking_granularity,
        has_trend_decision_intent=has_trend_decision_intent,
        has_anomaly_point_intent=has_anomaly_point_intent,
    )
    return ParsedQueryEntities(
        current_question=current_question,
        explicit_device_codes=explicit_device_codes,
        search_targets=search_targets,
        project_hints=project_hints,
        query_mode=query_mode,
        inferred_data_type=inferred_data_type,
        has_sensor_intent=has_sensor_intent,
        has_detect_data_types_intent=has_detect_data_types_intent,
        has_project_listing_intent=has_project_listing_intent,
        has_project_stats_intent=has_project_stats_intent,
        has_device_listing_intent=has_device_listing_intent,
        has_comparison_intent=has_comparison_intent,
        has_pagination_intent=has_pagination_intent,
        has_time_reference=has_time_reference,
        has_ranked_point_intent=has_ranked_point_intent,
        ranking_order=ranking_order,
        ranking_limit=ranking_limit,
        ranking_granularity=ranking_granularity,
        has_trend_decision_intent=has_trend_decision_intent,
        has_anomaly_point_intent=has_anomaly_point_intent,
    )


def _parse_small_natural_number(token: str) -> Optional[int]:
    value = str(token or "").strip()
    if not value:
        return None
    if value.isdigit():
        return int(value)
    digits = {
        "\u96f6": 0,
        "\u4e00": 1,
        "\u4e8c": 2,
        "\u4e24": 2,
        "\u4e09": 3,
        "\u56db": 4,
        "\u4e94": 5,
        "\u516d": 6,
        "\u4e03": 7,
        "\u516b": 8,
        "\u4e5d": 9,
    }
    if value == "\u5341":
        return 10
    if "\u5341" in value:
        left, right = value.split("\u5341", 1)
        if "\u5341" in right:
            return None
        tens = 1 if left == "" else digits.get(left)
        ones = 0 if right == "" else digits.get(right)
        if tens is None or ones is None:
            return None
        return tens * 10 + ones
    return digits.get(value)


def _extract_rank_limit(current_question: str) -> Optional[int]:
    for pattern in RANK_LIMIT_PATTERNS:
        match = pattern.search(current_question or "")
        if not match:
            continue
        parsed = _parse_small_natural_number(match.group("count"))
        if parsed is not None and parsed > 0:
            return min(parsed, 20)
    return None


def _extract_ranked_point_semantics(current_question: str, compact_lower: str) -> Tuple[Optional[str], Optional[int], Optional[str], bool]:
    has_high = any(keyword in current_question for keyword in RANK_HIGH_HINT_KEYWORDS) or "top" in compact_lower
    has_low = any(keyword in current_question for keyword in RANK_LOW_HINT_KEYWORDS)
    if has_high == has_low:
        return None, None, None, False

    rank_limit = _extract_rank_limit(current_question)
    has_point_hint = any(keyword in current_question for keyword in RANK_POINT_HINT_KEYWORDS)
    has_natural_day_top1 = any(keyword in current_question for keyword in ("哪天", "哪日"))
    has_natural_hour_top1 = any(keyword in current_question for keyword in ("哪个小时", "几点", "何时"))
    if rank_limit is None and not has_point_hint and not has_natural_day_top1 and not has_natural_hour_top1:
        return None, None, None, False

    granularity = None
    if any(keyword in current_question for keyword in RANK_DAY_HINT_KEYWORDS) or has_natural_day_top1:
        granularity = "day"
    elif any(keyword in current_question for keyword in RANK_HOUR_HINT_KEYWORDS) or has_natural_hour_top1:
        granularity = "hour"
    elif any(keyword in current_question for keyword in RANK_WEEK_HINT_KEYWORDS):
        granularity = "week"
    elif any(keyword in current_question for keyword in RANK_MONTH_HINT_KEYWORDS):
        granularity = "month"

    return ("desc" if has_high else "asc"), (rank_limit or 1), granularity, True


def _extract_trend_decision_intent(current_question: str, compact_lower: str) -> bool:
    if TRAILING_DECISION_PATTERN.search(current_question or ""):
        return True
    if any(keyword in current_question for keyword in TREND_DECISION_KEYWORDS):
        return True
    return any(token in compact_lower for token in ("\u589e\u52a0\u8fd8\u662f\u51cf\u5c11", "\u4e0a\u5347\u8fd8\u662f\u4e0b\u964d", "\u4e0a\u6da8\u8fd8\u662f\u4e0b\u8dcc"))


def _extract_bucket_granularity_hint(current_question: str) -> Optional[str]:
    if any(keyword in current_question for keyword in RANK_DAY_HINT_KEYWORDS):
        return "day"
    if any(keyword in current_question for keyword in RANK_HOUR_HINT_KEYWORDS):
        return "hour"
    if any(keyword in current_question for keyword in RANK_WEEK_HINT_KEYWORDS):
        return "week"
    if any(keyword in current_question for keyword in RANK_MONTH_HINT_KEYWORDS):
        return "month"
    return None


def _extract_anomaly_point_intent(current_question: str, compact_lower: str) -> bool:
    if not current_question:
        return False
    has_anomaly_language = any(keyword in current_question for keyword in ANOMALY_POINT_KEYWORDS)
    if not has_anomaly_language:
        return False
    if any(keyword in current_question for keyword in ANOMALY_TIME_HINT_KEYWORDS):
        return True
    return any(
        token in compact_lower
        for token in (
            "有没有异常",
            "有无异常",
            "是否异常",
            "异常用电",
            "异常数据",
            "异常波动",
        )
    )


def _infer_query_mode(
    *,
    has_sensor_intent: bool,
    has_detect_data_types_intent: bool,
    has_project_listing_intent: bool,
    has_project_stats_intent: bool,
    has_device_listing_intent: bool,
    has_comparison_intent: bool,
    has_ranked_point_intent: bool,
    ranking_granularity: Optional[str],
    has_trend_decision_intent: bool,
    has_anomaly_point_intent: bool,
) -> str:
    if has_detect_data_types_intent:
        return "detect_data_types"
    if has_project_listing_intent:
        return "project_listing"
    if has_project_stats_intent:
        return "project_stats"
    if has_device_listing_intent and not has_sensor_intent:
        return "device_listing"
    if has_anomaly_point_intent and has_sensor_intent and not has_comparison_intent:
        return "anomaly_points"
    if has_trend_decision_intent and has_sensor_intent and not has_comparison_intent:
        return "trend_decision"
    if has_ranked_point_intent and has_sensor_intent:
        return "ranked_buckets" if ranking_granularity in {"day", "hour", "week", "month"} else "ranked_timepoints"
    if has_comparison_intent and has_sensor_intent:
        return "comparison"
    if has_sensor_intent:
        return "sensor_query"
    return "general"


def normalize_search_target(keyword: str) -> str:
    if not keyword:
        return ""
    value = str(keyword).strip()
    value = re.sub(r"([\u4e00-\u9fff])([A-Za-z])", r"\1 \2", value)
    value = re.sub(r"([A-Za-z])([\u4e00-\u9fff])", r"\1 \2", value)
    value = re.sub(r"^[\s,\uff0c;\uff1b:?\uff1a\uff1f\u3002/]+|[\s,\uff0c;\uff1b:?\uff1a\uff1f\u3002/]+$", "", value)
    value = LEADING_VERB_PATTERN.sub("", value)
    value = LEADING_LISTING_PATTERN.sub("", value)
    value = TRAILING_TIME_PATTERN.sub("", value)
    value = TRAILING_METRIC_PATTERN.sub("", value)
    value = TRAILING_DETECT_PATTERN.sub("", value)
    value = TRAILING_DECISION_PATTERN.sub("", value)
    value = TRAILING_GENERIC_PATTERN.sub("", value)
    value = LEADING_CONTAINS_PATTERN.sub("", value)
    value = re.sub(r"^(?:\u5173\u4e8e|\u6709\u5173|\u9488\u5bf9)\s*", "", value)
    value = re.sub(r"^(?:\u8bbe\u5907|\u88c5\u7f6e)\s*", "", value)
    value = re.sub(r"\s*(?:\u8bbe\u5907|\u88c5\u7f6e)$", "", value)
    value = value.strip(" \u7684")
    value = re.sub(r"\s+", " ", value).strip()
    if value.lower() in GENERIC_EMPTY_TOKENS or value in GENERIC_EMPTY_TOKENS:
        return ""
    return value


def _has_explicit_device_listing_language(current_question: str, compact_lower: str) -> bool:
    if "\u8bbe\u5907" not in current_question:
        return False
    if any(keyword in current_question for keyword in DEVICE_LISTING_KEYWORDS):
        return True
    if DEVICE_LISTING_PATTERN.search(current_question):
        return True
    return any(token in compact_lower for token in ("\u54ea\u4e9b\u8bbe\u5907", "\u4ec0\u4e48\u8bbe\u5907", "\u6709\u54ea\u4e9b\u8bbe\u5907", "\u7535\u68af\u8bbe\u5907"))


def _extract_project_scoped_device_listing_hint(current_question: str) -> Optional[str]:
    match = PROJECT_DEVICE_LISTING_PATTERN.search(current_question or "")
    if not match:
        return None
    project_hint = normalize_search_target(match.group("project"))
    return project_hint or None


def _extract_explicit_device_codes(user_query: str) -> Tuple[str, ...]:
    seen = set()
    results = []
    for code in DEVICE_CODE_PATTERN.findall(user_query or ""):
        normalized = code.strip()
        lookup = normalized.lower()
        if lookup in seen:
            continue
        seen.add(lookup)
        results.append(normalized)
    return tuple(results)


def _infer_data_type_from_requested_tags(requested_tags: Tuple[str, ...]) -> Optional[str]:
    tags = tuple(str(tag or "").strip().lower() for tag in (requested_tags or ()) if str(tag or "").strip())
    if not tags:
        return None
    unique_tags = set(tags)
    if unique_tags <= {"ua", "ub", "uc"}:
        return tags[0] if len(tags) == 1 else "u_line"
    if unique_tags <= {"ia", "ib", "ic"}:
        return tags[0] if len(tags) == 1 else "i"
    if unique_tags <= {"uab", "ubc", "uca"}:
        return tags[0] if len(tags) == 1 else "u_phase"
    return tags[0]


def extract_requested_metric_tags(user_query: str) -> Tuple[str, ...]:
    current_question = extract_current_question_text(user_query)
    text = str(current_question or "").lower()
    if not text:
        return ()

    results = []
    seen = set()
    for tag, patterns in EXACT_METRIC_TAG_PATTERNS:
        for pattern in patterns:
            matched = re.search(pattern, text) if pattern.startswith("(?") else pattern in text
            if matched:
                if tag not in seen:
                    seen.add(tag)
                    results.append(tag)
                break
    return tuple(results)


def _infer_data_type(compact_lower: str) -> Optional[str]:
    for data_type, synonyms in METRIC_SYNONYMS:
        if any(keyword in compact_lower for keyword in synonyms):
            return data_type
    return None


def _extract_search_targets(current_question: str, comparison_mode: bool, explicit_device_codes: Tuple[str, ...]) -> Tuple[str, ...]:
    if len(explicit_device_codes) > 1:
        return explicit_device_codes
    if len(explicit_device_codes) == 1:
        return explicit_device_codes

    base_text = LEADING_VERB_PATTERN.sub("", current_question or "")

    contains_match = CONTAINS_TARGET_PATTERN.search(base_text)
    if contains_match:
        contains_target = normalize_search_target(contains_match.group("target"))
        if contains_target:
            return (contains_target,)

    context_parts = [part for part in CONTEXT_SPLIT_PATTERN.split(base_text, maxsplit=1) if part.strip()]
    if len(context_parts) == 2:
        contextual_target = normalize_search_target(context_parts[1])
        if contextual_target:
            return (contextual_target,)

    if comparison_mode:
        segments = COMPARISON_SPLIT_PATTERN.split(base_text)
    else:
        segments = [base_text]

    seen = set()
    results = []
    for segment in segments:
        normalized = normalize_search_target(segment)
        if not normalized:
            continue
        lookup = normalized.lower()
        if lookup in seen:
            continue
        seen.add(lookup)
        results.append(normalized)

    if results:
        return tuple(results)

    fallback = normalize_search_target(base_text)
    if fallback:
        return (fallback,)
    return ()


def _extract_project_hints(
    current_question: str,
    explicit_device_codes: Tuple[str, ...],
    search_targets: Tuple[str, ...],
) -> Tuple[str, ...]:
    base_text = LEADING_VERB_PATTERN.sub("", current_question or "")
    for device_code in explicit_device_codes:
        base_text = base_text.replace(device_code, " ")
    for target in search_targets:
        if not DEVICE_CODE_PATTERN.fullmatch(target or ""):
            base_text = base_text.replace(target, " ")

    hints = []
    seen = set()

    project_scoped_hint = _extract_project_scoped_device_listing_hint(current_question)
    if project_scoped_hint:
        lookup = project_scoped_hint.lower()
        if lookup not in seen:
            seen.add(lookup)
            hints.append(project_scoped_hint)

    context_parts = [part for part in CONTEXT_SPLIT_PATTERN.split(base_text, maxsplit=1) if part.strip()]
    if len(context_parts) == 2:
        maybe_hint = normalize_search_target(context_parts[0])
        if maybe_hint and _looks_like_project_scope(maybe_hint):
            lookup = maybe_hint.lower()
            if lookup not in seen:
                seen.add(lookup)
                hints.append(maybe_hint)

    for match in PROJECT_HINT_PATTERN.finditer(base_text):
        maybe_hint = normalize_search_target(match.group(0))
        if not maybe_hint or not _looks_like_project_scope(maybe_hint):
            continue
        lookup = maybe_hint.lower()
        if lookup in seen:
            continue
        seen.add(lookup)
        hints.append(maybe_hint)

    return tuple(hints)


def _looks_like_project_scope(value: str) -> bool:
    normalized = str(value or "").strip()
    return any(token in normalized for token in ("\u9879\u76ee", "\u5e73\u53f0", "\u7cfb\u7edf"))
