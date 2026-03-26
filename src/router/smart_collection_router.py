"""
Smart collection router.

This simplified router always chooses raw monthly collections and leaves
aggregation decisions to downstream MongoDB queries or the agent layer.
"""

from datetime import datetime
from typing import List, Optional, Tuple

from src.router.collection_router import get_collection_prefix, get_target_collections


def get_smart_collections(
    start_time: str,
    end_time: str,
    data_type: str = "ep",
    user_query: str = "",
    force_level: Optional[str] = None,
    mongo_client=None,
) -> Tuple[List[str], str, str]:
    """Return collections, chosen level, and human-readable reason."""
    collection_prefix = get_collection_prefix(data_type)
    collections = get_target_collections(start_time, end_time, collection_prefix)

    if not collections:
        return [], "raw", "未找到可用的数据集合"

    try:
        start_dt = datetime.strptime(start_time, "%Y-%m-%d")
        end_dt = datetime.strptime(end_time, "%Y-%m-%d")
        time_span = (end_dt - start_dt).days
        reason = f"时间跨度 {time_span} 天，使用原始数据集合"
    except Exception:
        reason = "使用原始数据集合"

    return collections, "raw", reason
