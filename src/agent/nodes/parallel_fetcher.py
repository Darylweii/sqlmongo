"""
Parallel Fetcher Node - 并行数据获取节点。

优先基于 QueryPlan 派生出来的统一查询上下文执行 MongoDB 获取，并生成后续汇总节点可直接消费的统计结果。
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.agent.query_plan_state import (
    build_query_plan_context,
    get_time_range_from_state,
    is_comparison_query,
)
from src.agent.types import GraphState, NODE_PARALLEL_FETCHER
from src.exceptions import DataFetcherError
from src.fetcher.data_fetcher import DataFetcher, SensorDataResult


logger = logging.getLogger(__name__)


class ParallelFetcherNode:
    """并行数据获取节点。"""

    def __init__(self, data_fetcher: DataFetcher):
        self.data_fetcher = data_fetcher

    def _build_query_info(
        self,
        *,
        state: GraphState,
        base_query_info: Optional[Dict[str, Any]],
        total_count: int,
        is_sampled: bool = False,
        failed_collections: Optional[List[str]] = None,
        target: Optional[str] = None,
    ) -> Dict[str, Any]:
        query_info = dict(base_query_info or {})
        query_info["total_count"] = total_count
        query_info["is_sampled"] = bool(is_sampled)
        if failed_collections:
            query_info["failed_collections"] = list(failed_collections)
        if target:
            query_info["target"] = target
        query_info["query_plan_context"] = build_query_plan_context(state)
        return query_info

    def __call__(self, state: GraphState) -> GraphState:
        history = list(state.get("history", []))
        device_codes = list(state.get("device_codes") or [])
        collections = list(state.get("collections") or [])
        data_tags = state.get("data_tags")
        comparison_device_groups = state.get("comparison_device_groups") or {}
        comparison_scope_groups = state.get("comparison_scope_groups") or {}
        time_start, time_end = get_time_range_from_state(state)
        is_comparison = is_comparison_query(state) and bool(comparison_device_groups)

        if not device_codes:
            return {
                **state,
                "error": "设备列表为空，无法获取时序数据。",
                "error_node": NODE_PARALLEL_FETCHER,
                "history": history + [{
                    "node": NODE_PARALLEL_FETCHER,
                    "result": "错误: 设备列表为空",
                }],
            }

        if not collections:
            return {
                **state,
                "error": "集合列表为空，无法获取时序数据。",
                "error_node": NODE_PARALLEL_FETCHER,
                "history": history + [{
                    "node": NODE_PARALLEL_FETCHER,
                    "result": "错误: 集合列表为空",
                }],
            }

        if not time_start or not time_end:
            return {
                **state,
                "error": "时间范围缺失，无法获取时序数据。",
                "error_node": NODE_PARALLEL_FETCHER,
                "history": history + [{
                    "node": NODE_PARALLEL_FETCHER,
                    "result": "错误: 时间范围缺失",
                }],
            }

        try:
            start_dt = datetime.strptime(time_start, "%Y-%m-%d")
            end_dt = datetime.strptime(time_end, "%Y-%m-%d").replace(hour=23, minute=59, second=59)

            if is_comparison:
                return self._fetch_comparison_data(
                    state=state,
                    collections=collections,
                    data_tags=data_tags,
                    start_dt=start_dt,
                    end_dt=end_dt,
                    comparison_device_groups=comparison_device_groups,
                    comparison_scope_groups=comparison_scope_groups,
                    history=history,
                )

            result: SensorDataResult = self.data_fetcher.fetch_sync(
                collections=collections,
                devices=device_codes,
                tgs=None,
                start_time=start_dt,
                end_time=end_dt,
                tags=data_tags,
            )

            query_info = self._build_query_info(
                state=state,
                base_query_info=result.query_info,
                total_count=int(result.total_count or 0),
                is_sampled=bool(getattr(result, "is_sampled", False)),
                failed_collections=list(getattr(result, "failed_collections", []) or []),
            )
            if result.statistics:
                query_info["statistics"] = dict(result.statistics)

            failed_collections = list(getattr(result, "failed_collections", []) or [])
            if failed_collections and len(failed_collections) == len(collections):
                return {
                    **state,
                    "error": f"所有集合查询失败: {', '.join(failed_collections)}",
                    "error_node": NODE_PARALLEL_FETCHER,
                    "raw_data": [],
                    "statistics": None,
                    "query_info": query_info,
                    "history": history + [{
                        "node": NODE_PARALLEL_FETCHER,
                        "result": f"错误: 所有 {len(failed_collections)} 个集合均失败",
                    }],
                }

            result_summary = f"获取 {int(result.total_count or 0)} 条数据记录"
            if failed_collections:
                result_summary += f"，{len(failed_collections)} 个集合失败"

            logger.info(
                "parallel_fetcher.ok devices=%s collections=%s total=%s sampled=%s",
                len(device_codes),
                len(collections),
                int(result.total_count or 0),
                bool(getattr(result, "is_sampled", False)),
            )

            return {
                **state,
                "raw_data": list(result.data or []),
                "total_count": int(result.total_count or 0),
                "statistics": result.statistics,
                "query_info": query_info,
                "history": history + [{
                    "node": NODE_PARALLEL_FETCHER,
                    "result": result_summary,
                }],
            }

        except ValueError as exc:
            logger.error("parallel_fetcher.invalid_time_range error=%s", exc)
            return {
                **state,
                "error": f"时间格式解析失败: {str(exc)}",
                "error_node": NODE_PARALLEL_FETCHER,
                "history": history + [{
                    "node": NODE_PARALLEL_FETCHER,
                    "result": "错误: 时间格式解析失败",
                }],
            }
        except DataFetcherError as exc:
            logger.error("parallel_fetcher.fetch_failed error=%s", exc)
            return {
                **state,
                "error": f"数据获取失败: {str(exc)}",
                "error_node": NODE_PARALLEL_FETCHER,
                "history": history + [{
                    "node": NODE_PARALLEL_FETCHER,
                    "result": f"错误: {str(exc)}",
                }],
            }
        except Exception as exc:
            logger.exception("parallel_fetcher.unexpected_error error=%s", exc)
            return {
                **state,
                "error": f"数据获取异常: {str(exc)}",
                "error_node": NODE_PARALLEL_FETCHER,
                "history": history + [{
                    "node": NODE_PARALLEL_FETCHER,
                    "result": f"异常: {str(exc)}",
                }],
            }

    def _fetch_comparison_data(
        self,
        *,
        state: GraphState,
        collections: List[str],
        data_tags: Optional[List[str]],
        start_dt: datetime,
        end_dt: datetime,
        comparison_device_groups: Dict[str, List[str]],
        comparison_scope_groups: Dict[str, List[Dict[str, Any]]],
        history: list,
    ) -> GraphState:
        comparison_statistics: Dict[str, Dict[str, Any]] = {}
        comparison_raw_data: List[Dict[str, Any]] = []
        total_count = 0
        target_query_infos: Dict[str, Dict[str, Any]] = {}
        filter_info = state.get("filter_info") or {}

        async def fetch_target_async(target: str, devices: List[str]):
            scope_rows = [row for row in (comparison_scope_groups.get(target) or []) if isinstance(row, dict)]
            scoped_devices = list(dict.fromkeys([
                str(row.get("device") or "").strip() for row in scope_rows if str(row.get("device") or "").strip()
            ])) or list(devices)
            scoped_tgs = list(dict.fromkeys([
                str(row.get("tg") or "").strip() for row in scope_rows if str(row.get("tg") or "").strip()
            ]))
            try:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: self.data_fetcher.fetch_sync(
                        collections=collections,
                        devices=scoped_devices,
                        tgs=scoped_tgs or None,
                        start_time=start_dt,
                        end_time=end_dt,
                        tags=data_tags,
                    ),
                )
                stats = dict(result.statistics or {})
                stats["count"] = int(result.total_count or 0)
                stats["devices"] = list(scoped_devices)
                stats["tgs"] = list(scoped_tgs)
                query_info = self._build_query_info(
                    state=state,
                    base_query_info=result.query_info,
                    total_count=int(result.total_count or 0),
                    is_sampled=bool(getattr(result, "is_sampled", False)),
                    failed_collections=list(getattr(result, "failed_collections", []) or []),
                    target=target,
                )
                if stats:
                    query_info["statistics"] = dict(stats)
                query_info["device_codes"] = list(scoped_devices)
                query_info["tg_values"] = list(scoped_tgs)
                return target, stats, query_info, int(result.total_count or 0), list(result.data or [])
            except Exception as exc:
                logger.warning("parallel_fetcher.comparison_target_failed target=%s error=%s", target, exc)
                return target, {"error": str(exc), "count": 0, "devices": list(scoped_devices), "tgs": list(scoped_tgs)}, {
                    "target": target,
                    "error": str(exc),
                    "device_codes": list(scoped_devices),
                    "tg_values": list(scoped_tgs),
                    "query_plan_context": build_query_plan_context(state),
                }, 0, []

        async def fetch_all_targets():
            tasks = [
                fetch_target_async(target, devices)
                for target, devices in comparison_device_groups.items()
            ]
            return await asyncio.gather(*tasks, return_exceptions=True)

        try:
            asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, fetch_all_targets())
                results = future.result()
        except RuntimeError:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            results = loop.run_until_complete(fetch_all_targets())

        for result in results:
            if isinstance(result, Exception):
                logger.error("parallel_fetcher.comparison_batch_failed error=%s", result)
                continue
            target, stats, query_info, count, records = result
            comparison_statistics[target] = stats
            target_query_infos[target] = query_info
            total_count += count
            if records:
                comparison_raw_data.extend([item for item in records if isinstance(item, dict)])

        result_summary = f"对比统计 {len(comparison_device_groups)} 个目标，共 {total_count} 条数据"
        logger.info(
            "parallel_fetcher.comparison_ok targets=%s total=%s raw_records=%s filter_strategy=%s",
            len(comparison_device_groups),
            total_count,
            len(comparison_raw_data),
            filter_info.get("strategy", "none"),
        )

        return {
            **state,
            "raw_data": comparison_raw_data,
            "total_count": total_count,
            "statistics": None,
            "comparison_statistics": comparison_statistics,
            "query_info": {
                "mode": "comparison",
                "targets": target_query_infos,
                "comparison_scope_groups": comparison_scope_groups,
                "total_count": total_count,
                "query_plan_context": build_query_plan_context(state),
            },
            "history": history + [{
                "node": NODE_PARALLEL_FETCHER,
                "result": result_summary,
            }],
        }
