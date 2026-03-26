"""
Sharding Router Node - 分片路由计算节点

使用纯 Python 代码根据时间范围计算目标 MongoDB 集合名称。

功能：
- 根据时间范围和数据类型计算集合名称列表
- 实现熔断机制（集合数量超过阈值时触发）
- 支持多种数据类型的集合前缀映射

需求引用：
- 需求 4.2: 使用纯 Python 代码计算集合名称
- 需求 4.4: 熔断机制
- 需求 4.5: 数据类型前缀映射
"""

import logging
from typing import Optional

from src.agent.types import GraphState, NODE_SHARDING_ROUTER
from src.agent.query_plan_state import (
    get_data_type_from_state,
    get_intent_from_state,
    get_query_mode_from_state,
    get_query_plan_from_state,
    get_requested_tags_from_state,
    get_time_range_from_state,
)
from src.router.collection_router import (
    get_collection_prefix,
    get_data_tags,
    get_target_collections,
)
from src.exceptions import CircuitBreakerError, InvalidDateRangeError


logger = logging.getLogger(__name__)


class ShardingRouterNode:
    """
    分片路由计算节点 - 使用纯 Python 代码
    
    根据时间范围和数据类型计算目标 MongoDB 集合名称。
    
    核心职责：
    1. 根据 intent 中的时间范围计算目标集合
    2. 根据数据类型获取集合前缀
    3. 实现熔断机制防止查询范围过大
    
    Attributes:
        max_collections: 最大允许的集合数量阈值（默认 50）
    """
    
    def __init__(self, max_collections: int = 50):
        """
        初始化分片路由节点
        
        Args:
            max_collections: 最大允许的集合数量阈值，超过此值触发熔断
        """
        self.max_collections = max_collections
    
    def __call__(self, state: GraphState) -> GraphState:
        """
        计算目标数据分片。

        优先读取 query_plan 中已经规范化的时间范围与数据类型；若缺失，则回退到 intent。
        """
        history = list(state.get("history", []))
        intent = get_intent_from_state(state)
        query_plan = get_query_plan_from_state(state)

        if not intent and query_plan is None:
            return {
                **state,
                "error": "缺少意图或 QueryPlan，无法计算数据分片。",
                "error_node": NODE_SHARDING_ROUTER,
                "history": history + [{
                    "node": NODE_SHARDING_ROUTER,
                    "result": "错误: 缺少意图或 QueryPlan"
                }]
            }

        time_start, time_end = get_time_range_from_state(state)
        data_type = get_data_type_from_state(state, default="ep")
        query_mode = get_query_mode_from_state(state)

        if not time_start or not time_end:
            return {
                **state,
                "error": "缺少时间范围，无法计算数据分片。",
                "error_node": NODE_SHARDING_ROUTER,
                "history": history + [{
                    "node": NODE_SHARDING_ROUTER,
                    "result": "错误: 缺少时间范围"
                }]
            }

        try:
            collection_prefix = get_collection_prefix(data_type)
            collections = get_target_collections(
                start_date=time_start,
                end_date=time_end,
                collection_prefix=collection_prefix,
                max_collections=self.max_collections
            )
            requested_tags = get_requested_tags_from_state(state)
            data_tags = requested_tags or get_data_tags(collection_prefix)

            logger.info(
                f"分片路由完成: query_mode={query_mode}, 时间范围={time_start}~{time_end}, "
                f"数据类型={data_type}, 分片数={len(collections)}"
            )

            return {
                **state,
                "collections": collections,
                "data_tags": data_tags,
                "history": history + [{
                    "node": NODE_SHARDING_ROUTER,
                    "result": f"计算出 {len(collections)} 个数据分片"
                }]
            }

        except CircuitBreakerError as e:
            logger.warning(f"分片路由触发熔断: {e}")
            return {
                **state,
                "error": str(e),
                "error_node": NODE_SHARDING_ROUTER,
                "history": history + [{
                    "node": NODE_SHARDING_ROUTER,
                    "result": f"错误: {e.collection_count} 个分片超过上限 {e.max_allowed}"
                }]
            }

        except InvalidDateRangeError as e:
            logger.error(f"分片路由日期范围无效: {e}")
            return {
                **state,
                "error": str(e),
                "error_node": NODE_SHARDING_ROUTER,
                "history": history + [{
                    "node": NODE_SHARDING_ROUTER,
                    "result": f"错误: {str(e)}"
                }]
            }

        except Exception as e:
            logger.exception(f"分片路由执行异常: {e}")
            return {
                **state,
                "error": f"分片路由执行异常: {str(e)}",
                "error_node": NODE_SHARDING_ROUTER,
                "history": history + [{
                    "node": NODE_SHARDING_ROUTER,
                    "result": f"错误: {str(e)}"
                }]
            }
