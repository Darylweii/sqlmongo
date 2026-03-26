"""
Sensor Data Tool - 传感器数据查询工具

简化版：只查询原始数据表 source_data_{type}_{YYYYMM}
AI 可以通过 MongoDB Aggregation Pipeline 自己计算统计结果
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
import json
import re

from langchain.tools import tool
from pymongo import MongoClient

from src.router.collection_router import get_target_collections, get_collection_prefix, get_data_tags
from src.fetcher.data_fetcher import DataFetcher, SensorDataResult
from src.analysis import InsightEngine
from src.agent.query_entities import extract_requested_metric_tags
from src.agent.query_plan import coerce_query_plan, fallback_query_plan
from src.agent.query_plan_state import build_query_plan_context as build_query_plan_context_from_state
from src.cache.cache_manager import CacheManager
from src.compressor.context_compressor import ContextCompressor, OutputFormat
from src.exceptions import (
    CircuitBreakerError,
    InvalidDateRangeError,
    DataFetcherError,
    CacheConnectionError,
)


logger = logging.getLogger(__name__)

# Global component instances
_mongo_client: Optional[MongoClient] = None
_cache_manager: Optional[CacheManager] = None
_data_fetcher: Optional[DataFetcher] = None
_context_compressor: Optional[ContextCompressor] = None

# Configuration
LONG_QUERY_THRESHOLD_DAYS = 90  # 超过90天自动分页
MAX_RECORDS_THRESHOLD = 2000


def configure_sensor_tool(
    mongo_uri: str,
    database_name: str = "sensor_db",
    redis_url: Optional[str] = None,
    max_records: int = 2000,
    cache_ttl: int = 3600,
    max_tokens: int = 4000
) -> None:
    """配置传感器工具的全局组件"""
    global _mongo_client, _cache_manager, _data_fetcher, _context_compressor
    
    _mongo_client = MongoClient(mongo_uri)
    _data_fetcher = DataFetcher(
        mongo_client=_mongo_client,
        database_name=database_name,
        max_records=max_records,
        cache_ttl=cache_ttl
    )
    
    if redis_url:
        _cache_manager = CacheManager(redis_url, default_ttl=cache_ttl)
    
    _context_compressor = ContextCompressor(max_tokens=max_tokens)


def get_data_fetcher() -> DataFetcher:
    if _data_fetcher is None:
        raise RuntimeError("DataFetcher 未配置")
    return _data_fetcher


def get_cache_manager() -> Optional[CacheManager]:
    return _cache_manager


def get_context_compressor() -> ContextCompressor:
    return _context_compressor or ContextCompressor()


def _parse_datetime(time_str: str) -> datetime:
    """解析日期时间字符串"""
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(time_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"无效的时间格式: {time_str}")


def _is_long_query(start_time: str, end_time: str, threshold_days: int = 90) -> bool:
    """Check whether the query spans a long time range."""
    try:
        start_dt = _parse_datetime(start_time)
        end_dt = _parse_datetime(end_time)
        return (end_dt - start_dt).days > threshold_days
    except ValueError:
        return False


def _short_text(text: str, limit: int = 120) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(value) <= limit:
        return value
    return value[: max(limit - 3, 0)] + "..."


def _sample_items(values, limit: int = 6):
    if not values:
        return []
    return list(values)[:limit]


def _build_sensor_query_plan_context(
    device_codes: List[str],
    start_time: str,
    end_time: str,
    data_type: str = "ep",
    user_query: str = "",
    query_plan: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    normalized_devices = [str(code).strip() for code in (device_codes or []) if str(code or "").strip()]
    comparison_targets = normalized_devices if len(normalized_devices) > 1 else []
    target_label = " vs ".join(normalized_devices) if len(normalized_devices) > 1 else (normalized_devices[0] if normalized_devices else "")
    requested_tags = _resolve_requested_tags(data_type=data_type, user_query=user_query, query_plan=query_plan)

    state: Dict[str, Any] = {
        "intent": {
            "target": target_label,
            "search_targets": normalized_devices,
            "comparison_targets": comparison_targets,
            "data_type": data_type,
            "requested_tags": requested_tags,
            "time_start": start_time,
            "time_end": end_time,
            "query_mode": "comparison" if comparison_targets else "sensor_query",
            "response_style": "structured_analysis",
            "is_comparison": bool(comparison_targets),
        }
    }

    resolved_plan = coerce_query_plan(query_plan)
    if resolved_plan is None and user_query:
        resolved_plan = fallback_query_plan(user_query)
    if resolved_plan is not None:
        state["query_plan"] = resolved_plan.to_dict()

    return build_query_plan_context_from_state(state)


_PHASE_TAG_FAMILIES = {
    "ua": "u_line",
    "ub": "u_line",
    "uc": "u_line",
    "ia": "i",
    "ib": "i",
    "ic": "i",
    "uab": "u_phase",
    "ubc": "u_phase",
    "uca": "u_phase",
}


def _normalize_requested_tags(tags: Any) -> List[str]:
    results: List[str] = []
    seen = set()
    for tag in tags or []:
        text = str(tag or "").strip().lower()
        if not text or text in seen:
            continue
        seen.add(text)
        results.append(text)
    return results


def _resolve_requested_tags(
    *,
    data_type: str,
    user_query: str = "",
    query_plan: Optional[Dict[str, Any]] = None,
) -> List[str]:
    plan = coerce_query_plan(query_plan)
    if plan and isinstance(plan.raw_plan, dict):
        requested = _normalize_requested_tags(plan.raw_plan.get("requested_tags"))
        if requested:
            return requested

    requested = _normalize_requested_tags(extract_requested_metric_tags(user_query))
    if requested:
        return requested

    normalized_data_type = str(data_type or "").strip().lower()
    if normalized_data_type in _PHASE_TAG_FAMILIES:
        return [normalized_data_type]
    return []


def _resolve_analysis_data_type(data_type: str, requested_tags: List[str]) -> str:
    normalized_data_type = str(data_type or "").strip().lower()
    tags = _normalize_requested_tags(requested_tags)
    if len(tags) == 1:
        return tags[0]
    if tags:
        families = {(_PHASE_TAG_FAMILIES.get(tag) or normalized_data_type) for tag in tags}
        if len(families) == 1:
            return next(iter(families))
    return normalized_data_type or "ep"


def _enrich_query_info_with_query_plan_context(
    query_info: Optional[Dict[str, Any]],
    *,
    device_codes: List[str],
    start_time: str,
    end_time: str,
    data_type: str = "ep",
    user_query: str = "",
    query_plan: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    enriched = dict(query_info or {})
    enriched["query_plan_context"] = _build_sensor_query_plan_context(
        device_codes=device_codes,
        start_time=start_time,
        end_time=end_time,
        data_type=data_type,
        user_query=user_query,
        query_plan=query_plan,
    )
    return enriched


CHART_INTENT_KEYWORDS = (
    "图表",
    "画图",
    "画出来",
    "可视化",
    "折线图",
    "柱状图",
    "雷达图",
    "箱线图",
    "热力图",
    "曲线图",
    "趋势图",
    "对比图",
    "chart",
    "plot",
    "echart",
    "echarts",
)


_SMALL_CHINESE_NUMBERS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}

_FOCUSED_HIGH_KEYWORDS = ("最高", "最大", "峰值")
_FOCUSED_LOW_KEYWORDS = ("最低", "最小", "谷值")
_FOCUSED_TIMEPOINT_KEYWORDS = (
    "时间点",
    "时刻",
    "数据点",
    "记录",
    "什么时候",
    "何时",
    "峰值时间",
    "谷值时间",
)
_FOCUSED_LIMIT_PATTERNS = (
    re.compile(r"前\s*(?P<count>\d{1,3}|[零一二两三四五六七八九十]{1,3})\s*个?"),
    re.compile(r"top\s*(?P<count>\d{1,3})", re.IGNORECASE),
)


def _parse_small_natural_number(token: str) -> Optional[int]:
    value = str(token or "").strip()
    if not value:
        return None
    if value.isdigit():
        return int(value)
    if value == "十":
        return 10
    if "十" in value:
        left, right = value.split("十", 1)
        if "十" in right:
            return None
        tens = 1 if left == "" else _SMALL_CHINESE_NUMBERS.get(left)
        ones = 0 if right == "" else _SMALL_CHINESE_NUMBERS.get(right)
        if tens is None or ones is None:
            return None
        return tens * 10 + ones
    return _SMALL_CHINESE_NUMBERS.get(value)


def _extract_focused_limit(user_query: str) -> Optional[int]:
    text = str(user_query or "")
    for pattern in _FOCUSED_LIMIT_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        count = _parse_small_natural_number(match.group("count"))
        if count is not None and count > 0:
            return min(count, 20)
    return None


def _extract_focused_timepoint_intent(user_query: str) -> Optional[Dict[str, Any]]:
    text = str(user_query or "").strip().lower()
    if not text:
        return None

    has_high = any(keyword in text for keyword in _FOCUSED_HIGH_KEYWORDS)
    has_low = any(keyword in text for keyword in _FOCUSED_LOW_KEYWORDS)
    if has_high == has_low:
        return None

    requested_limit = _extract_focused_limit(text)
    has_timepoint_language = any(keyword in text for keyword in _FOCUSED_TIMEPOINT_KEYWORDS)
    if requested_limit is None and not has_timepoint_language:
        return None

    limit = requested_limit or 1
    return {
        "mode": "ranked_timepoints",
        "order": "desc" if has_high else "asc",
        "limit": limit,
        "requested_limit": limit,
    }


def _extract_record_value(record: Dict[str, Any]) -> Optional[float]:
    if not isinstance(record, dict):
        return None
    for key in ("val", "value", "average", "total", "diff", "max", "min"):
        value = record.get(key)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _extract_record_time(record: Dict[str, Any]) -> str:
    if not isinstance(record, dict):
        return ""
    for key in ("logTime", "time", "dataTime", "date", "month", "year"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _format_focused_value(value: float) -> Any:
    if float(value).is_integer():
        return int(value)
    return round(float(value), 2)


CUMULATIVE_DATA_TYPES = {"ep", "epzyz", "fz-ep", "gffddl", "dcdcdljl", "dcdfdljl"}


def _parse_focused_datetime(value: str) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    normalized = text.replace("/", "-")
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            continue
    return None


def _is_cumulative_data_type(data_type: str) -> bool:
    return str(data_type or "").strip().lower() in CUMULATIVE_DATA_TYPES


def _bucket_start(dt: datetime, granularity: str) -> datetime:
    if granularity == "day":
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    if granularity == "hour":
        return dt.replace(minute=0, second=0, microsecond=0)
    if granularity == "week":
        base = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        return base - timedelta(days=base.weekday())
    if granularity == "month":
        return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return dt


def _bucket_label(dt: datetime, granularity: str) -> str:
    if granularity == "day":
        return dt.strftime("%Y-%m-%d")
    if granularity == "hour":
        return dt.strftime("%Y-%m-%d %H:00")
    if granularity == "week":
        end_dt = dt + timedelta(days=6)
        return f"{dt.strftime("%Y-%m-%d")}~{end_dt.strftime("%Y-%m-%d")}"
    if granularity == "month":
        return dt.strftime("%Y-%m")
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _build_enriched_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    enriched_rows: List[Dict[str, Any]] = []
    for item in records or []:
        value = _extract_record_value(item)
        time_label = _extract_record_time(item)
        dt = _parse_focused_datetime(time_label)
        if value is None or not time_label or dt is None:
            continue
        enriched_rows.append(
            {
                "time": time_label,
                "dt": dt,
                "device": str(item.get("device") or ""),
                "tag": str(item.get("tag") or ""),
                "value": value,
            }
        )
    return sorted(enriched_rows, key=lambda row: (row["dt"], row.get("device") or "", row.get("tag") or ""))


def _build_bucket_rows(records: List[Dict[str, Any]], data_type: str, granularity: str) -> List[Dict[str, Any]]:
    buckets: Dict[datetime, List[Dict[str, Any]]] = {}
    for row in _build_enriched_rows(records):
        buckets.setdefault(_bucket_start(row["dt"], granularity), []).append(row)

    aggregated_rows: List[Dict[str, Any]] = []
    is_cumulative = _is_cumulative_data_type(data_type)
    for bucket_dt, rows in sorted(buckets.items(), key=lambda item: item[0]):
        values = [row["value"] for row in rows]
        if not values:
            continue
        if is_cumulative:
            bucket_value = max(values) - min(values)
            aggregate_method = "bucket_diff"
        else:
            bucket_value = sum(values) / len(values)
            aggregate_method = "bucket_avg"

        aggregated_rows.append(
            {
                "time": _bucket_label(bucket_dt, granularity),
                "dt": bucket_dt,
                "value": round(bucket_value, 2),
                "sample_count": len(rows),
                "granularity": granularity,
                "aggregate_method": aggregate_method,
            }
        )
    return aggregated_rows


def _granularity_label(granularity: str) -> str:
    mapping = {
        "hour": "按小时",
        "day": "按天",
        "week": "按周",
        "month": "按月",
    }
    return mapping.get(str(granularity or "").strip(), "按周期")


def _build_bucket_summary_result(parsed, analysis: Dict[str, Any], records: List[Dict[str, Any]], data_type: str) -> Optional[Dict[str, Any]]:
    granularity = str(parsed.ranking_granularity or "").strip().lower()
    if granularity not in {"hour", "day", "week", "month"}:
        return None
    bucket_rows = _build_bucket_rows(records, data_type, granularity)
    if not bucket_rows:
        return None

    metric_name = str(analysis.get("metric") or "数值")
    unit = str(analysis.get("unit") or "")
    granularity_label = _granularity_label(granularity)
    aggregate_note = (
        "累计型指标按每个周期内末值减首值聚合。"
        if _is_cumulative_data_type(data_type)
        else "非累计型指标按每个周期内平均值聚合。"
    )

    table_rows = []
    result_rows = []
    for row in bucket_rows:
        table_rows.append(
            {
                "周期": row["time"],
                "聚合值": _format_focused_value(row["value"]),
                "样本数": row.get("sample_count", 0),
            }
        )
        result_rows.append(dict(row))

    return {
        "mode": "bucket_summary",
        "metric": metric_name,
        "unit": unit,
        "granularity": granularity,
        "headline": f"已按{granularity_label}汇总{metric_name}，共 {len(result_rows)} 个周期",
        "rows": result_rows,
        "aggregation_note": aggregate_note,
        "table": {
            "headers": ["周期", "聚合值", "样本数"],
            "rows": table_rows,
            "page_size": len(table_rows),
            "total_count": len(table_rows),
            "has_more": False,
            "view_label": "问题直答",
        },
    }


def _select_trend_basis_rows(records: List[Dict[str, Any]], data_type: str) -> Tuple[List[Dict[str, Any]], str]:
    enriched_rows = _build_enriched_rows(records)
    if len(enriched_rows) < 2:
        return enriched_rows, "raw"

    span_hours = (enriched_rows[-1]["dt"] - enriched_rows[0]["dt"]).total_seconds() / 3600
    preferred_granularity = None
    if _is_cumulative_data_type(data_type):
        if span_hours >= 24:
            preferred_granularity = "day"
        elif span_hours >= 2:
            preferred_granularity = "hour"
    else:
        if span_hours >= 48:
            preferred_granularity = "day"
        elif span_hours >= 4:
            preferred_granularity = "hour"

    if preferred_granularity:
        bucket_rows = _build_bucket_rows(records, data_type, preferred_granularity)
        if len(bucket_rows) >= 2:
            return bucket_rows, preferred_granularity
    return enriched_rows, "raw"


def _build_ranked_timepoint_result(parsed, analysis: Dict[str, Any], records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    enriched_rows = _build_enriched_rows(records)
    if not enriched_rows or not parsed.ranking_order:
        return None

    reverse = parsed.ranking_order == "desc"
    ranked_rows = sorted(
        enriched_rows,
        key=lambda row: (row["value"], row["time"], row.get("device") or "", row.get("tag") or ""),
        reverse=reverse,
    )[: (parsed.ranking_limit or 1)]
    if not ranked_rows:
        return None

    metric_name = str(analysis.get("metric") or "数值")
    unit = str(analysis.get("unit") or "")
    order_label = "从高到低" if reverse else "从低到高"
    extreme_label = "最高" if reverse else "最低"

    table_rows = []
    result_rows = []
    for index, row in enumerate(ranked_rows, start=1):
        table_rows.append(
            {
                "排名": index,
                "时间": row["time"],
                "设备": row.get("device") or "-",
                "标签": row.get("tag") or "-",
                "数值": _format_focused_value(row["value"]),
            }
        )
        result_rows.append({**row, "rank": index})

    return {
        "mode": parsed.query_mode,
        "order": parsed.ranking_order,
        "limit": len(result_rows),
        "requested_limit": parsed.ranking_limit or len(result_rows),
        "metric": metric_name,
        "unit": unit,
        "headline": f"已按{metric_name}{order_label}排序，返回{extreme_label}的前 {len(result_rows)} 个时间点",
        "rows": result_rows,
        "table": {
            "headers": ["排名", "时间", "设备", "标签", "数值"],
            "rows": table_rows,
            "page_size": len(table_rows),
            "total_count": len(table_rows),
            "has_more": False,
            "view_label": "问题直答",
        },
    }


def _build_ranked_bucket_result(parsed, analysis: Dict[str, Any], records: List[Dict[str, Any]], data_type: str) -> Optional[Dict[str, Any]]:
    granularity = parsed.ranking_granularity or "day"
    bucket_rows = _build_bucket_rows(records, data_type, granularity)
    if not bucket_rows or not parsed.ranking_order:
        return None

    reverse = parsed.ranking_order == "desc"
    ranked_rows = sorted(bucket_rows, key=lambda row: (row["value"], row["time"]), reverse=reverse)[: (parsed.ranking_limit or 1)]
    if not ranked_rows:
        return None

    metric_name = str(analysis.get("metric") or "数值")
    unit = str(analysis.get("unit") or "")
    granularity_label = "按天" if granularity == "day" else "按小时"
    order_label = "从高到低" if reverse else "从低到高"
    aggregate_note = (
        "累计型指标按每个周期内末值减首值聚合。"
        if _is_cumulative_data_type(data_type)
        else "非累计型指标按每个周期内平均值聚合。"
    )

    table_rows = []
    result_rows = []
    for index, row in enumerate(ranked_rows, start=1):
        table_rows.append(
            {
                "排名": index,
                "周期": row["time"],
                "聚合值": _format_focused_value(row["value"]),
                "样本数": row.get("sample_count", 0),
            }
        )
        result_rows.append({**row, "rank": index})

    return {
        "mode": parsed.query_mode,
        "order": parsed.ranking_order,
        "limit": len(result_rows),
        "requested_limit": parsed.ranking_limit or len(result_rows),
        "metric": metric_name,
        "unit": unit,
        "granularity": granularity,
        "headline": f"已按{granularity_label}{metric_name}{order_label}排序，返回前 {len(result_rows)} 个周期",
        "rows": result_rows,
        "aggregation_note": aggregate_note,
        "table": {
            "headers": ["排名", "周期", "聚合值", "样本数"],
            "rows": table_rows,
            "page_size": len(table_rows),
            "total_count": len(table_rows),
            "has_more": False,
            "view_label": "问题直答",
        },
    }


def _build_trend_decision_result(parsed, analysis: Dict[str, Any], records: List[Dict[str, Any]], data_type: str) -> Optional[Dict[str, Any]]:
    basis_rows, basis_granularity = _select_trend_basis_rows(records, data_type)
    if len(basis_rows) < 2:
        return None

    values = [float(row["value"]) for row in basis_rows if row.get("value") is not None]
    if len(values) < 2:
        return None

    window = max(1, min(3, len(values) // 3 or 1))
    start_mean = sum(values[:window]) / window
    end_mean = sum(values[-window:]) / window
    if abs(start_mean) < 1e-9:
        change_rate = 0.0 if abs(end_mean) < 1e-9 else 100.0
    else:
        change_rate = round(((end_mean - start_mean) / abs(start_mean)) * 100, 2)

    if change_rate >= 5:
        direction = "up"
        direction_label = "上升"
    elif change_rate <= -5:
        direction = "down"
        direction_label = "下降"
    else:
        direction = "stable"
        direction_label = "稳定"

    metric_name = str(analysis.get("metric") or "数值")
    unit = str(analysis.get("unit") or "")
    if basis_granularity == "day":
        basis_label = "按天"
    elif basis_granularity == "hour":
        basis_label = "按小时"
    else:
        basis_label = "按原始时序"

    aggregate_note = (
        "累计型指标趋势判断按周期增量均值比较。"
        if _is_cumulative_data_type(data_type) and basis_granularity in {"day", "hour"}
        else "趋势判断按首尾窗口均值比较。"
    )
    headline = f"{metric_name}整体呈{direction_label}趋势" if direction != "stable" else f"{metric_name}整体基本稳定"

    return {
        "mode": parsed.query_mode,
        "metric": metric_name,
        "unit": unit,
        "direction": direction,
        "direction_label": direction_label,
        "change_rate": change_rate,
        "basis_granularity": basis_granularity,
        "headline": headline,
        "start_mean": round(start_mean, 2),
        "end_mean": round(end_mean, 2),
        "start_label": basis_rows[0].get("time") or "-",
        "end_label": basis_rows[-1].get("time") or "-",
        "aggregation_note": aggregate_note,
        "basis_label": basis_label,
    }

def _build_anomaly_points_result(parsed, analysis: Dict[str, Any], records: List[Dict[str, Any]], data_type: str) -> Optional[Dict[str, Any]]:
    basis_rows, basis_granularity = _select_trend_basis_rows(records, data_type)
    metric_name = str(analysis.get("metric") or "数值")
    unit = str(analysis.get("unit") or "")
    if basis_granularity == "day":
        basis_label = "按天"
    elif basis_granularity == "hour":
        basis_label = "按小时"
    else:
        basis_label = "按原始时序"

    values = [float(row["value"]) for row in basis_rows if row.get("value") is not None]
    detection_note = (
        "累计型指标已先按周期折算后，再使用 IQR 方法检测异常点。"
        if _is_cumulative_data_type(data_type) and basis_granularity in {"day", "hour"}
        else "已使用 IQR 方法检测异常点。"
    )
    if len(values) < 4:
        return {
            "mode": parsed.query_mode,
            "metric": metric_name,
            "unit": unit,
            "basis_granularity": basis_granularity,
            "basis_label": basis_label,
            "sample_count": len(values),
            "anomaly_count": 0,
            "anomaly_ratio_pct": 0.0,
            "rows": [],
            "headline": f"{metric_name}样本量不足，暂时无法稳定识别异常点",
            "aggregation_note": detection_note,
            "insufficient_samples": True,
            "table": {
                "headers": ["时间", "数值"],
                "rows": [],
                "page_size": 0,
                "total_count": 0,
                "has_more": False,
                "view_label": "异常点",
            },
        }

    q1, _, q3 = InsightEngine._quartiles(values)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    anomaly_rows = InsightEngine._detect_anomaly_points(basis_rows, q1, q3)

    ranked_rows = []
    for row in anomaly_rows:
        value = float(row.get("value") or 0.0)
        if value > upper_bound:
            severity = value - upper_bound
            direction = "high"
        else:
            severity = lower_bound - value
            direction = "low"
        ranked_rows.append(
            {
                **row,
                "severity": round(severity, 2),
                "direction": direction,
            }
        )

    ranked_rows = sorted(
        ranked_rows,
        key=lambda item: (item.get("severity") or 0.0, item.get("time") or ""),
        reverse=True,
    )
    requested_limit = parsed.ranking_limit or 5
    limit = max(1, min(requested_limit, 20))
    selected_rows = [{**row, "rank": index} for index, row in enumerate(ranked_rows[:limit], start=1)]

    include_device = any(row.get("device") for row in selected_rows)
    include_tag = any(row.get("tag") for row in selected_rows)
    headers = ["排名", "时间"]
    if include_device:
        headers.append("设备")
    if include_tag:
        headers.append("标签")
    headers.extend(["数值", "异常强度"])

    table_rows = []
    for row in selected_rows:
        item = {
            "排名": row["rank"],
            "时间": row.get("time") or "-",
        }
        if include_device:
            item["设备"] = row.get("device") or "-"
        if include_tag:
            item["标签"] = row.get("tag") or "-"
        item["数值"] = _format_focused_value(row.get("value") or 0.0)
        item["异常强度"] = _format_focused_value(row.get("severity") or 0.0)
        table_rows.append(item)

    anomaly_count = len(ranked_rows)
    anomaly_ratio_pct = round((anomaly_count / len(values)) * 100, 2) if values else 0.0
    if anomaly_count <= 0:
        headline = f"{basis_label}未发现明显的{metric_name}异常点"
    else:
        headline = f"{basis_label}共识别到 {anomaly_count} 个{metric_name}异常点"

    return {
        "mode": parsed.query_mode,
        "metric": metric_name,
        "unit": unit,
        "basis_granularity": basis_granularity,
        "basis_label": basis_label,
        "sample_count": len(values),
        "anomaly_count": anomaly_count,
        "anomaly_ratio_pct": anomaly_ratio_pct,
        "lower_bound": round(lower_bound, 2),
        "upper_bound": round(upper_bound, 2),
        "limit": len(selected_rows),
        "requested_limit": requested_limit,
        "rows": selected_rows,
        "headline": headline,
        "aggregation_note": detection_note,
        "table": {
            "headers": headers,
            "rows": table_rows,
            "page_size": len(table_rows),
            "total_count": len(table_rows),
            "has_more": False,
            "view_label": "异常点",
        },
    }



def _apply_analysis_scope_to_focused_result(focused: Optional[Dict[str, Any]], analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(focused, dict):
        return focused
    scope_label = str((analysis or {}).get("analysis_scope_label") or "").strip()
    if not scope_label:
        return focused

    enriched = dict(focused)
    enriched["analysis_scope_label"] = scope_label
    headline = str(enriched.get("headline") or "").strip()
    if headline and scope_label not in headline:
        enriched["headline"] = f"{scope_label}：{headline}"
    return enriched


def _build_focused_result(
    records: List[Dict[str, Any]],
    analysis: Dict[str, Any],
    user_query: str,
    data_type: str,
    query_plan: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if not isinstance(analysis, dict) or analysis.get("mode") == "comparison":
        return None

    parsed = coerce_query_plan(query_plan) or fallback_query_plan(user_query)
    focused_result = None
    if parsed.query_mode == "trend_decision" and parsed.has_trend_decision_intent:
        focused_result = _build_trend_decision_result(parsed, analysis, records, data_type)
    elif parsed.query_mode == "anomaly_points" and parsed.has_anomaly_point_intent:
        focused_result = _build_anomaly_points_result(parsed, analysis, records, data_type)
    elif parsed.query_mode == "ranked_buckets" and parsed.has_ranked_point_intent:
        focused_result = _build_ranked_bucket_result(parsed, analysis, records, data_type)
    elif parsed.has_ranked_point_intent and parsed.ranking_order:
        focused_result = _build_ranked_timepoint_result(parsed, analysis, records)
    elif parsed.aggregation == "bucket" and parsed.ranking_granularity in {"hour", "day", "week", "month"}:
        focused_result = _build_bucket_summary_result(parsed, analysis, records, data_type)
    return _apply_analysis_scope_to_focused_result(focused_result, analysis)


def _has_chart_intent(user_query: str) -> bool:
    text = (user_query or "").strip().lower()
    if not text:
        return False
    if any(keyword in text for keyword in CHART_INTENT_KEYWORDS):
        return True
    explicit_patterns = (
        r"(画|生成|绘制|做|出).{0,4}(图|图表)",
        r"(可视化|趋势图|对比图|折线图|柱状图|雷达图|箱线图|热力图)",
    )
    return any(re.search(pattern, text) for pattern in explicit_patterns)


@tool
def fetch_sensor_data(
    devices: List[str],
    start_time: str,
    end_time: str,
    output_format: str = "minimal"
) -> Dict[str, Any]:
    """
    查询指定设备在指定时间范围内的时序数据。
    
    Args:
        devices: 设备代号列表
        start_time: 开始时间 (YYYY-MM-DD)
        end_time: 结束时间 (YYYY-MM-DD)
        output_format: 输出格式 (minimal/csv/markdown/json)
    
    Returns:
        包含时序数据的字典
    """
    if not devices:
        return {"error": "缺少设备代号", "success": False}
    
    if not start_time or not end_time:
        return {"error": "缺少时间范围", "success": False}
    
    devices = [d.strip() for d in devices if d and d.strip()]
    start_time = start_time.strip()
    end_time = end_time.strip()
    
    try:
        start_dt = _parse_datetime(start_time)
        end_dt = _parse_datetime(end_time)
        
        if len(end_time) == 10:
            end_dt = end_dt.replace(hour=23, minute=59, second=59)
        
        if start_dt > end_dt:
            return {"error": "开始时间不能晚于结束时间", "success": False}
        
        # 获取集合
        collections = get_target_collections(start_time, end_time)
        
        # 查询数据
        data_fetcher = get_data_fetcher()
        sensor_result = data_fetcher.fetch_sync(
            collections=collections,
            devices=devices,
            tgs=None,
            start_time=start_dt,
            end_time=end_dt
        )
        
        # 压缩数据
        analysis, chart_specs = InsightEngine.build(
            sensor_result.data,
            sensor_result.statistics,
            device_codes=devices,
        )

        compressor = get_context_compressor()
        format_map = {
            "minimal": OutputFormat.MINIMAL,
            "csv": OutputFormat.CSV,
            "markdown": OutputFormat.MARKDOWN,
            "json": OutputFormat.JSON
        }
        out_format = format_map.get(output_format.lower(), OutputFormat.MINIMAL)
        compressed_data = compressor.compress(sensor_result.data, out_format)
        
        return {
            "data": compressed_data,
            "total_count": sensor_result.total_count,
            "is_sampled": sensor_result.is_sampled,
            "statistics": sensor_result.statistics,
            "analysis": analysis,
            "chart_specs": chart_specs,
            "show_charts": False,
            "success": True,
            "query_info": _enrich_query_info_with_query_plan_context(
                sensor_result.query_info,
                device_codes=devices,
                start_time=start_time,
                end_time=end_time,
                data_type="ep",
            )
        }
        
    except Exception as e:
        logger.error(f"查询错误: {e}")
        return {"error": str(e), "success": False}


def fetch_sensor_data_with_components(
    device_codes: List[str],
    start_time: str,
    end_time: str,
    data_fetcher: DataFetcher,
    cache_manager: Optional[CacheManager] = None,
    compressor: Optional[ContextCompressor] = None,
    output_format: str = "minimal",
    data_type: str = "ep",
    page: int = 1,
    page_size: int = 0,
    tg_values: Optional[List[str]] = None,
    user_query: str = "",
    query_plan: Optional[Dict[str, Any]] = None,
    use_aggregation: bool = False,
    llm = None,
    value_filter: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Use the provided components to fetch sensor data from raw collections.

    Args:
        value_filter: Numeric filter condition, e.g. {"gt": 100}, {"lt": 50}, {"gte": 100}, {"lte": 50}
    """
    if not device_codes:
        return {"error": "缺少设备代号", "success": False}

    if not start_time or not end_time:
        return {"error": "缺少时间范围", "success": False}

    device_codes = [code.strip() for code in device_codes if code and code.strip()]
    start_time = start_time.strip()
    end_time = end_time.strip()
    request_log = {
        "event": "sensor.fetch.request",
        "device_codes": _sample_items(device_codes, limit=8),
        "device_count": len(device_codes),
        "tg_values": _sample_items(tg_values, limit=8),
        "tg_count": len(tg_values or []),
        "start_time": start_time,
        "end_time": end_time,
        "data_type": data_type,
        "page": page,
        "page_size": page_size,
        "output_format": output_format,
        "use_aggregation": use_aggregation,
        "value_filter": value_filter,
        "user_query": _short_text(user_query),
        "query_plan_mode": (query_plan or {}).get("query_mode") if isinstance(query_plan, dict) else None,
    }
    logger.info("%s", json.dumps(request_log, ensure_ascii=False, default=str))

    try:
        start_dt = _parse_datetime(start_time)
        end_dt = _parse_datetime(end_time)

        if len(end_time) == 10:
            end_dt = end_dt.replace(hour=23, minute=59, second=59)

        if start_dt > end_dt:
            return {"error": "开始时间不能晚于结束时间", "success": False}

        cache_hit = False
        cache_key = None
        if cache_manager:
            try:
                requested_tags = _resolve_requested_tags(data_type=data_type, user_query=user_query, query_plan=query_plan)
                cache_key = cache_manager._generate_key(
                    device_codes,
                    start_time,
                    end_time,
                    data_type=data_type,
                    tags=requested_tags,
                    tg_values=tg_values,
                    page=page,
                    page_size=page_size,
                    output_format=output_format,
                    value_filter=value_filter,
                )
                cached_result = cache_manager.get(cache_key)
                if cached_result:
                    cached_result = dict(cached_result)
                    cached_result["cache_hit"] = True
                    cached_result["query_info"] = _enrich_query_info_with_query_plan_context(
                        cached_result.get("query_info"),
                        device_codes=device_codes,
                        start_time=start_time,
                        end_time=end_time,
                        data_type=data_type,
                        user_query=user_query,
                        query_plan=query_plan,
                    )
                    logger.info(
                        "%s",
                        json.dumps(
                            {
                                **request_log,
                                "event": "sensor.fetch.cache_hit",
                                "page": cached_result.get("page"),
                                "page_size": cached_result.get("page_size"),
                                "total_count": cached_result.get("total_count"),
                            },
                            ensure_ascii=False,
                            default=str,
                        ),
                    )
                    return cached_result
            except CacheConnectionError as e:
                logger.warning(
                    "sensor.fetch.cache.unavailable %s",
                    json.dumps({**request_log, "event": "sensor.fetch.cache.unavailable", "error": str(e)}, ensure_ascii=False, default=str),
                )

        is_long_query = _is_long_query(start_time, end_time, LONG_QUERY_THRESHOLD_DAYS)
        if is_long_query and page_size == 0:
            page_size = 500
            logger.info(
                "%s",
                json.dumps(
                    {**request_log, "event": "sensor.fetch.auto_pagination", "page_size": page_size},
                    ensure_ascii=False,
                    default=str,
                ),
            )

        requested_tags = _resolve_requested_tags(data_type=data_type, user_query=user_query, query_plan=query_plan)
        analysis_data_type = _resolve_analysis_data_type(data_type, requested_tags)
        collection_prefix = get_collection_prefix(data_type)
        collections = get_target_collections(start_time, end_time, collection_prefix)
        data_tags = requested_tags or get_data_tags(collection_prefix)
        logger.info(
            "%s",
            json.dumps(
                {
                    **request_log,
                    "event": "sensor.fetch.route",
                    "collection_prefix": collection_prefix,
                    "collection_count": len(collections),
                    "collections": _sample_items(collections, limit=6),
                    "tags": _sample_items(data_tags, limit=6),
                },
                ensure_ascii=False,
                default=str,
            ),
        )

        sensor_result = data_fetcher.fetch_sync(
            collections=collections,
            devices=device_codes,
            tgs=tg_values,
            start_time=start_dt,
            end_time=end_dt,
            tags=data_tags,
            page=page,
            page_size=page_size,
            value_filter=value_filter
        )

        analysis, chart_specs = InsightEngine.build(
            sensor_result.data,
            sensor_result.statistics,
            data_type=analysis_data_type,
            device_codes=device_codes,
            user_query=user_query,
        )
        focused_result = _build_focused_result(sensor_result.data, analysis, user_query, analysis_data_type, query_plan=query_plan)

        if compressor is None:
            compressor = ContextCompressor()

        format_map = {
            "minimal": OutputFormat.MINIMAL,
            "csv": OutputFormat.CSV,
            "markdown": OutputFormat.MARKDOWN,
            "json": OutputFormat.JSON
        }
        out_format = format_map.get(output_format.lower(), OutputFormat.MINIMAL)

        if out_format == OutputFormat.JSON:
            compressed_data = json.dumps(sensor_result.data, default=str, ensure_ascii=False)
        else:
            compressed_data = compressor.compress(sensor_result.data, out_format)

        show_charts = _has_chart_intent(user_query)

        result = {
            "data": compressed_data,
            "total_count": sensor_result.total_count,
            "is_sampled": sensor_result.is_sampled,
            "statistics": sensor_result.statistics,
            "analysis": analysis,
            "chart_specs": chart_specs,
            "focused_result": focused_result,
            "focused_table": focused_result.get("table") if isinstance(focused_result, dict) else None,
            "show_charts": show_charts,
            "failed_collections": sensor_result.failed_collections,
            "is_long_query": is_long_query,
            "cache_hit": cache_hit,
            "success": True,
            "query_info": _enrich_query_info_with_query_plan_context(
                sensor_result.query_info,
                device_codes=device_codes,
                start_time=start_time,
                end_time=end_time,
                data_type=data_type,
                user_query=user_query,
                query_plan=query_plan,
            ),
            "page": sensor_result.page,
            "page_size": sensor_result.page_size,
            "total_pages": sensor_result.total_pages,
            "has_more": sensor_result.has_more
        }

        if cache_manager and cache_key:
            try:
                if cache_manager.is_recent_query(start_time, end_time):
                    cache_manager.set(cache_key, result)
            except CacheConnectionError as e:
                logger.warning(
                    "sensor.fetch.cache_store.failed %s",
                    json.dumps({**request_log, "event": "sensor.fetch.cache_store.failed", "error": str(e)}, ensure_ascii=False, default=str),
                )

        logger.info(
            "%s",
            json.dumps(
                {
                    **request_log,
                    "event": "sensor.fetch.result",
                    "collection_count": len(collections),
                    "total_count": sensor_result.total_count,
                    "page": sensor_result.page,
                    "page_size": sensor_result.page_size,
                    "total_pages": sensor_result.total_pages,
                    "has_more": sensor_result.has_more,
                    "is_sampled": sensor_result.is_sampled,
                    "failed_collections": sensor_result.failed_collections,
                    "analysis_mode": analysis.get("mode") if isinstance(analysis, dict) else None,
                    "chart_count": len(chart_specs or []),
                    "show_charts": show_charts,
                },
                ensure_ascii=False,
                default=str,
            ),
        )
        return result

    except (CircuitBreakerError, InvalidDateRangeError, DataFetcherError) as e:
        logger.warning(
            "sensor.fetch.failed %s",
            json.dumps({**request_log, "event": "sensor.fetch.failed", "error": str(e)}, ensure_ascii=False, default=str),
        )
        return {"error": str(e), "success": False}
    except ValueError as e:
        logger.warning(
            "sensor.fetch.invalid_input %s",
            json.dumps({**request_log, "event": "sensor.fetch.invalid_input", "error": str(e)}, ensure_ascii=False, default=str),
        )
        return {"error": str(e), "success": False}
    except Exception as e:
        logger.exception(
            "sensor.fetch.error %s",
            json.dumps({**request_log, "event": "sensor.fetch.error", "error": str(e)}, ensure_ascii=False, default=str),
        )
        return {"error": f"未知错误: {str(e)}", "success": False}

def detect_device_data_types(
    device_codes: List[str],
    data_fetcher: DataFetcher,
    tg_values: Optional[List[str]] = None,
    month: str = None
) -> Dict[str, Any]:
    """
    探测设备有哪些数据类型
    
    Args:
        device_codes: 设备代号列表
        data_fetcher: DataFetcher 实例
        month: 月份 YYYYMM，默认当前月
    
    Returns:
        各数据类型的记录数
    """
    if not device_codes:
        return {"error": "缺少设备代号", "success": False}
    
    if month is None:
        month = datetime.now().strftime("%Y%m")
    
    # 常见数据类型
    data_types = ["ep", "i", "u_line", "p", "qf", "t"]
    
    results = {}
    db = data_fetcher.mongo_client[data_fetcher.database_name]
    
    for data_type in data_types:
        collection_name = f"source_data_{data_type}_{month}"
        try:
            if collection_name in db.list_collection_names():
                query = {"device": {"$in": device_codes}}
                if tg_values:
                    query["tg"] = {"$in": tg_values}
                count = db[collection_name].count_documents(query, limit=1)  # 只检查是否存在
                if count > 0:
                    # 获取实际数量
                    total = db[collection_name].count_documents(query)
                    results[data_type] = total
        except Exception as e:
            logger.error(f"探测 {data_type} 失败: {e}")
    
    return {
        "success": True,
        "device_codes": device_codes,
        "tg_values": tg_values or [],
        "month": month,
        "available_types": results,
        "summary": ", ".join([f"{k}({v}条)" for k, v in results.items()]) if results else "无数据"
    }
