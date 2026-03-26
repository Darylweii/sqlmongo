"""
Parallel MongoDB data fetcher.

This module provides `DataFetcher`, which queries one or more monthly MongoDB
collections, merges results, supports pagination, and degrades gracefully when
part of the query fails.

Expected MongoDB shape:
- collection name: `source_data_<type>_YYYYMM`
- fields: `device`, `logTime`, `dataTime`, `val`, `tg`, `tag`
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from pymongo import MongoClient

logger = logging.getLogger(__name__)
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

from src.exceptions import DataFetcherError


@dataclass
class SensorDataResult:
    """
    Result container for sensor data queries.
    
    Attributes:
        data: List of sensor records sorted by time
        total_count: Total number of records before any sampling
        is_sampled: Whether the data was downsampled
        statistics: Optional statistics summary (min, max, avg)
        failed_collections: List of collections that failed to query
        query_info: Query information (MongoDB query details)
        page: Current page number (1-based)
        page_size: Number of records per page
        total_pages: Total number of pages
        has_more: Whether there are more pages
    """
    data: List[Dict[str, Any]]
    total_count: int
    is_sampled: bool = False
    statistics: Optional[Dict[str, float]] = None
    failed_collections: List[str] = field(default_factory=list)
    query_info: Optional[Dict[str, Any]] = None
    page: int = 1
    page_size: int = 0
    total_pages: int = 1
    has_more: bool = False


class DataFetcher:
    """
    时序数据获取器
    
    Handles parallel queries across multiple MongoDB collections,
    merges results, sorts by time, and applies downsampling when needed.
    
    实际 MongoDB 文档结构:
    {
        "_id": ObjectId,
        "tg": "TG233",
        "device": "b1_b14",  # 设备代号
        "tag": "ep",
        "logTime": "2025-02-01 00:00:22",  # 日志时间
        "dataTime": "2025-1-31 23:58:19.687",  # 数据时间
        "val": 45122.0  # 传感器值
    }
    """
    
    def __init__(
        self,
        mongo_client: MongoClient,
        database_name: str = "sensor_db",
        max_records: int = 2000,
        cache_ttl: int = 3600
    ):
        """
        Initialize the DataFetcher.
        
        Args:
            mongo_client: MongoDB client instance
            database_name: Name of the MongoDB database
            max_records: Maximum records before downsampling (default: 2000)
            cache_ttl: Cache TTL in seconds (default: 3600)
        """
        self.mongo_client = mongo_client
        self.database_name = database_name
        self.max_records = max_records
        self.cache_ttl = cache_ttl
    
    @property
    def db(self):
        """Get the MongoDB database instance."""
        return self.mongo_client[self.database_name]

    async def fetch_parallel(
        self,
        collections: List[str],
        devices: List[str],
        tgs: Optional[List[str]],
        start_time: datetime,
        end_time: datetime,
        tags: Optional[List[str]] = None,
        page: int = 1,
        page_size: int = 0,
        value_filter: Optional[Dict] = None,
        use_db_pagination: bool = True  # 新增：是否在数据库层面分页
    ) -> SensorDataResult:
        """
        并发查询多个集合并聚合结果
        
        Args:
            collections: List of target collection names
            devices: List of device codes to query (对应 MongoDB 的 device 字段)
            start_time: Start time for the query range
            end_time: End time for the query range
            tags: Optional list of tag values to filter (如 ["ia", "ib", "ic"])
            page: Page number (1-based), default 1
            page_size: Records per page, 0 means no pagination
            value_filter: Optional value filter {"gt": 100}, {"lt": 50}, {"gte": 100}, {"lte": 50}
        
        Returns:
            SensorDataResult containing merged and sorted data
        """
        if not collections:
            return SensorDataResult(data=[], total_count=0)
        
        if not devices:
            return SensorDataResult(data=[], total_count=0)
        
        # 构建查询信息
        start_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
        end_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
        mongo_query = {
            "device": {"$in": devices},
            "logTime": {"$gte": start_str, "$lte": end_str}
        }
        if tgs:
            mongo_query["tg"] = {"$in": tgs}
        if tags:
            mongo_query["tag"] = {"$in": tags}
        
        # 添加数值过滤条件
        if value_filter:
            val_condition = {}
            if "gt" in value_filter:
                val_condition["$gt"] = value_filter["gt"]
            if "gte" in value_filter:
                val_condition["$gte"] = value_filter["gte"]
            if "lt" in value_filter:
                val_condition["$lt"] = value_filter["lt"]
            if "lte" in value_filter:
                val_condition["$lte"] = value_filter["lte"]
            if val_condition:
                mongo_query["val"] = val_condition
        
        query_info = {
            "type": "MongoDB",
            "database": self.database_name,
            "collections": collections,
            "query": mongo_query,
            "query_string": f'db.{collections[0]}.find({mongo_query})'
        }
        
        # Create tasks for parallel execution
        # 如果启用数据库分页且只有一个集合，直接在数据库层面分页
        if use_db_pagination and page_size > 0 and len(collections) == 1:
            logger.info(f"[数据库分页] 使用 MongoDB skip/limit 分页")
            tasks = [
                self._fetch_from_collection_paginated(
                    collection_name=collections[0],
                    devices=devices,
                    tgs=tgs,
                    start_time=start_time,
                    end_time=end_time,
                    tags=tags,
                    value_filter=value_filter,
                    page=page,
                    page_size=page_size
                )
            ]
        else:
            # 多集合或不分页时，使用原来的方式
            tasks = [
                self._fetch_from_collection(
                    collection_name=coll,
                    devices=devices,
                    tgs=tgs,
                    start_time=start_time,
                    end_time=end_time,
                    tags=tags,
                    value_filter=value_filter
                )
                for coll in collections
            ]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and track failures
        all_data: List[Dict[str, Any]] = []
        failed_collections: List[str] = []
        
        for coll_name, result in zip(collections, results):
            if isinstance(result, Exception):
                failed_collections.append(coll_name)
            else:
                all_data.extend(result)
        
        # If all collections failed, raise error
        if len(failed_collections) == len(collections) and collections:
            raise DataFetcherError(
                f"所有集合查询失败: {failed_collections}"
            )
        
        # 如果使用了数据库分页，需要获取总数
        if use_db_pagination and page_size > 0 and len(collections) == 1:
            # 获取总记录数
            loop = asyncio.get_event_loop()
            collection = self.db[collections[0]]
            
            # 构建查询条件（与分页查询相同）
            start_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
            end_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
            count_query = {
                "device": {"$in": devices},
                "logTime": {"$gte": start_str, "$lte": end_str}
            }
            if tgs:
                count_query["tg"] = {"$in": tgs}
            if tags:
                count_query["tag"] = {"$in": tags}
            if value_filter:
                val_condition = {}
                if "gt" in value_filter:
                    val_condition["$gt"] = value_filter["gt"]
                if "gte" in value_filter:
                    val_condition["$gte"] = value_filter["gte"]
                if "lt" in value_filter:
                    val_condition["$lt"] = value_filter["lt"]
                if "lte" in value_filter:
                    val_condition["$lte"] = value_filter["lte"]
                if val_condition:
                    count_query["val"] = val_condition
            
            total_count = await loop.run_in_executor(
                None,
                lambda: collection.count_documents(count_query)
            )
            logger.info(f"[数据库分页] 总记录数: {total_count}")
            
            # 数据已经在数据库层面排序和分页，不需要再排序
            statistics = self._compute_statistics(all_data) if len(all_data) > 0 else None
            
            total_pages = (total_count + page_size - 1) // page_size if page_size > 0 else 1
            has_more = page < total_pages
            
            return SensorDataResult(
                data=all_data,
                total_count=total_count,
                is_sampled=False,
                statistics=statistics,
                failed_collections=failed_collections,
                query_info=query_info,
                page=page,
                page_size=page_size,
                total_pages=total_pages,
                has_more=has_more
            )
        
        # 原来的内存分页逻辑
        # Sort by time
        all_data = self._sort_by_time(all_data)
        
        total_count = len(all_data)
        statistics = self._compute_statistics(all_data) if total_count > 0 else None
        
        # 分页处理
        total_pages = 1
        has_more = False
        is_sampled = False
        
        logger.info(f"[分页处理] 总记录数: {total_count}, page: {page}, page_size: {page_size}")
        
        if page_size > 0 and total_count > page_size:
            # 使用分页
            total_pages = (total_count + page_size - 1) // page_size
            page = max(1, min(page, total_pages))  # 确保页码有效
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            
            logger.info(f"[分页处理] 总页数: {total_pages}, 当前页: {page}")
            logger.info(f"[分页处理] 索引范围: {start_idx} ~ {end_idx}")
            
            if all_data:
                logger.info(f"[分页处理] 分页前第一条: logTime={all_data[0].get('logTime')}, _id={all_data[0].get('_id')}")
                logger.info(f"[分页处理] 分页前最后一条: logTime={all_data[-1].get('logTime')}, _id={all_data[-1].get('_id')}")
            
            all_data = all_data[start_idx:end_idx]
            has_more = page < total_pages
            
            if all_data:
                logger.info(f"[分页处理] 分页后记录数: {len(all_data)}")
                logger.info(f"[分页处理] 分页后第一条: logTime={all_data[0].get('logTime')}, _id={all_data[0].get('_id')}")
                logger.info(f"[分页处理] 分页后最后一条: logTime={all_data[-1].get('logTime')}, _id={all_data[-1].get('_id')}")
        elif total_count > self.max_records:
            # 降采样
            all_data = self._downsample(all_data, self.max_records)
            is_sampled = True
        
        return SensorDataResult(
            data=all_data,
            total_count=total_count,
            is_sampled=is_sampled,
            statistics=statistics,
            failed_collections=failed_collections,
            query_info=query_info,
            page=page,
            page_size=page_size if page_size > 0 else total_count,
            total_pages=total_pages,
            has_more=has_more
        )
    
    async def _fetch_from_collection(
        self,
        collection_name: str,
        devices: List[str],
        tgs: Optional[List[str]],
        start_time: datetime,
        end_time: datetime,
        tags: Optional[List[str]] = None,
        value_filter: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch data from a single collection.
        
        适配实际字段:
        - device: 设备代号
        - logTime: 日志时间 (字符串格式 "YYYY-MM-DD HH:MM:SS")
        - tag: 数据类型标签 (如 ep, ia, ib, ic 等)
        """
        collection: Collection = self.db[collection_name]
        
        # 转换时间为字符串格式进行比较 (因为 logTime 是字符串)
        start_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
        end_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
        
        query = {
            "device": {"$in": devices},
            "logTime": {
                "$gte": start_str,
                "$lte": end_str
            }
        }
        if tgs:
            query["tg"] = {"$in": tgs}
        
        # 添加 tag 过滤
        if tags:
            query["tag"] = {"$in": tags}
        
        # 添加数值过滤条件
        if value_filter:
            val_condition = {}
            if "gt" in value_filter:
                val_condition["$gt"] = value_filter["gt"]
            if "gte" in value_filter:
                val_condition["$gte"] = value_filter["gte"]
            if "lt" in value_filter:
                val_condition["$lt"] = value_filter["lt"]
            if "lte" in value_filter:
                val_condition["$lte"] = value_filter["lte"]
            if val_condition:
                query["val"] = val_condition
        
        # Run synchronous MongoDB query in executor
        loop = asyncio.get_event_loop()
        try:
            # 打印查询信息
            logger.info(f"[MongoDB查询] 集合: {collection_name}")
            logger.info(f"[MongoDB查询] 条件: {query}")
            logger.info(f"[MongoDB查询] 排序: [('logTime', 1), ('_id', 1)]")
            
            cursor = await loop.run_in_executor(
                None,
                lambda: list(collection.find(query).sort([("logTime", 1), ("_id", 1)]))  # 按时间和_id排序，确保稳定
            )
            
            logger.info(f"[MongoDB查询] 返回记录数: {len(cursor)}")
            if cursor:
                logger.info(f"[MongoDB查询] 第一条: logTime={cursor[0].get('logTime')}, _id={cursor[0].get('_id')}")
                logger.info(f"[MongoDB查询] 最后一条: logTime={cursor[-1].get('logTime')}, _id={cursor[-1].get('_id')}")
            
            # Process documents
            for doc in cursor:
                if "_id" in doc:
                    doc["_id"] = str(doc["_id"])
                # 解析 logTime 为 datetime 用于排序
                if "logTime" in doc and isinstance(doc["logTime"], str):
                    try:
                        doc["_parsed_time"] = datetime.strptime(
                            doc["logTime"], "%Y-%m-%d %H:%M:%S"
                        )
                    except ValueError:
                        doc["_parsed_time"] = datetime.min
            return cursor
        except PyMongoError as e:
            raise DataFetcherError(f"查询集合 {collection_name} 失败: {e}")
    
    async def _fetch_from_collection_paginated(
        self,
        collection_name: str,
        devices: List[str],
        tgs: Optional[List[str]],
        start_time: datetime,
        end_time: datetime,
        tags: Optional[List[str]] = None,
        value_filter: Optional[Dict] = None,
        page: int = 1,
        page_size: int = 50
    ) -> List[Dict[str, Any]]:
        """
        从单个集合获取数据 - 使用 MongoDB skip/limit 分页
        
        这个方法在数据库层面进行分页，避免加载全部数据到内存
        """
        collection: Collection = self.db[collection_name]
        
        # 转换时间为字符串格式
        start_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
        end_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
        
        query = {
            "device": {"$in": devices},
            "logTime": {
                "$gte": start_str,
                "$lte": end_str
            }
        }
        if tgs:
            query["tg"] = {"$in": tgs}
        
        # 添加 tag 过滤
        if tags:
            query["tag"] = {"$in": tags}
        
        # 添加数值过滤条件
        if value_filter:
            val_condition = {}
            if "gt" in value_filter:
                val_condition["$gt"] = value_filter["gt"]
            if "gte" in value_filter:
                val_condition["$gte"] = value_filter["gte"]
            if "lt" in value_filter:
                val_condition["$lt"] = value_filter["lt"]
            if "lte" in value_filter:
                val_condition["$lte"] = value_filter["lte"]
            if val_condition:
                query["val"] = val_condition
        
        # 计算 skip
        skip = (page - 1) * page_size
        
        # Run synchronous MongoDB query in executor
        loop = asyncio.get_event_loop()
        try:
            # 打印查询信息
            logger.info(f"[MongoDB分页查询] 集合: {collection_name}")
            logger.info(f"[MongoDB分页查询] 条件: {query}")
            logger.info(f"[MongoDB分页查询] 排序: [('logTime', 1), ('_id', 1)]")
            logger.info(f"[MongoDB分页查询] 分页: skip={skip}, limit={page_size}")
            
            cursor = await loop.run_in_executor(
                None,
                lambda: list(
                    collection.find(query)
                    .sort([("logTime", 1), ("_id", 1)])
                    .skip(skip)
                    .limit(page_size)
                )
            )
            
            logger.info(f"[MongoDB分页查询] 返回记录数: {len(cursor)}")
            if cursor:
                logger.info(f"[MongoDB分页查询] 第一条: logTime={cursor[0].get('logTime')}, _id={cursor[0].get('_id')}")
                logger.info(f"[MongoDB分页查询] 最后一条: logTime={cursor[-1].get('logTime')}, _id={cursor[-1].get('_id')}")
            
            # Process documents
            for doc in cursor:
                if "_id" in doc:
                    doc["_id"] = str(doc["_id"])
                # 解析 logTime 为 datetime 用于排序
                if "logTime" in doc and isinstance(doc["logTime"], str):
                    try:
                        doc["_parsed_time"] = datetime.strptime(
                            doc["logTime"], "%Y-%m-%d %H:%M:%S"
                        )
                    except ValueError:
                        doc["_parsed_time"] = datetime.min
            return cursor
        except PyMongoError as e:
            raise DataFetcherError(f"查询集合 {collection_name} 失败: {e}")
    
    def _sort_by_time(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort data by time field in ascending order, with _id as secondary sort for stability."""
        logger.info(f"[内存排序] 排序前记录数: {len(data)}")
        if data:
            logger.info(f"[内存排序] 第一条: logTime={data[0].get('logTime')}, _id={data[0].get('_id')}")
            logger.info(f"[内存排序] 最后一条: logTime={data[-1].get('logTime')}, _id={data[-1].get('_id')}")
        
        sorted_data = sorted(
            data,
            key=lambda x: (x.get("_parsed_time", x.get("logTime", "")), x.get("_id", ""))
        )
        
        logger.info(f"[内存排序] 排序后记录数: {len(sorted_data)}")
        if sorted_data:
            logger.info(f"[内存排序] 第一条: logTime={sorted_data[0].get('logTime')}, _id={sorted_data[0].get('_id')}")
            logger.info(f"[内存排序] 最后一条: logTime={sorted_data[-1].get('logTime')}, _id={sorted_data[-1].get('_id')}")
        
        return sorted_data

    def _downsample(
        self, 
        data: List[Dict[str, Any]], 
        target_size: int
    ) -> List[Dict[str, Any]]:
        """
        降采样处理
        
        Reduces the data size by selecting evenly spaced samples.
        """
        if len(data) <= target_size:
            return data
        
        if target_size <= 0:
            return []
        
        if target_size == 1:
            return [data[0]]
        
        if target_size == 2:
            return [data[0], data[-1]]
        
        step = (len(data) - 1) / (target_size - 1)
        
        result = []
        for i in range(target_size):
            index = int(i * step)
            index = min(index, len(data) - 1)
            result.append(data[index])
        
        return result
    
    def _compute_statistics(
        self, 
        data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        计算丰富的统计摘要
        
        包含：
        - 基础统计：min, max, avg, sum, count
        - 分布统计：std（标准差）, median（中位数）, p5/p95（分位数）
        - 趋势分析：trend（上升/下降/稳定）, change_rate（变化率）
        - 异常检测：anomaly_count（异常值数量）
        - 时间分布：peak_hour（峰值时段）, low_hour（低谷时段）
        """
        if not data:
            return {"min": 0.0, "max": 0.0, "avg": 0.0, "count": 0}
        
        # 提取数值和时间
        values = []
        time_values = {}  # hour -> [values]
        
        for record in data:
            val = record.get("val")
            if val is not None:
                try:
                    v = float(val)
                    values.append(v)
                    
                    # 按小时分组
                    parsed_time = record.get("_parsed_time")
                    if parsed_time:
                        hour = parsed_time.hour
                        if hour not in time_values:
                            time_values[hour] = []
                        time_values[hour].append(v)
                except (TypeError, ValueError):
                    continue
        
        if not values:
            return {"min": 0.0, "max": 0.0, "avg": 0.0, "count": len(data)}
        
        # 基础统计
        n = len(values)
        total = sum(values)
        avg = total / n
        min_val = min(values)
        max_val = max(values)
        
        # 排序用于分位数计算
        sorted_values = sorted(values)
        
        # 中位数
        if n % 2 == 0:
            median = (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
        else:
            median = sorted_values[n // 2]
        
        # 分位数 (p5, p25, p75, p95)
        def percentile(sorted_vals, p):
            idx = int(len(sorted_vals) * p / 100)
            idx = max(0, min(idx, len(sorted_vals) - 1))
            return sorted_vals[idx]
        
        p5 = percentile(sorted_values, 5)
        p25 = percentile(sorted_values, 25)
        p75 = percentile(sorted_values, 75)
        p95 = percentile(sorted_values, 95)
        
        # 标准差
        variance = sum((v - avg) ** 2 for v in values) / n
        std = variance ** 0.5
        
        # 变异系数 (CV) - 标准差/平均值，衡量相对波动
        cv = (std / avg * 100) if avg != 0 else 0
        
        # 异常值检测 (IQR 方法)
        iqr = p75 - p25
        lower_bound = p25 - 1.5 * iqr
        upper_bound = p75 + 1.5 * iqr
        anomaly_count = sum(1 for v in values if v < lower_bound or v > upper_bound)
        
        # 趋势分析（比较前后 1/3 数据的平均值）
        third = max(1, n // 3)
        first_third_avg = sum(values[:third]) / third
        last_third_avg = sum(values[-third:]) / third
        
        change_rate = ((last_third_avg - first_third_avg) / first_third_avg * 100) if first_third_avg != 0 else 0
        
        if change_rate > 5:
            trend = "上升"
        elif change_rate < -5:
            trend = "下降"
        else:
            trend = "稳定"
        
        # 时间分布分析
        time_distribution = None
        if time_values:
            hour_avgs = {h: sum(vs) / len(vs) for h, vs in time_values.items()}
            if hour_avgs:
                peak_hour = max(hour_avgs, key=hour_avgs.get)
                low_hour = min(hour_avgs, key=hour_avgs.get)
                time_distribution = {
                    "peak_hour": peak_hour,
                    "peak_value": round(hour_avgs[peak_hour], 2),
                    "low_hour": low_hour,
                    "low_value": round(hour_avgs[low_hour], 2),
                }
        
        result = {
            # 基础统计
            "min": round(min_val, 2),
            "max": round(max_val, 2),
            "avg": round(avg, 2),
            "sum": round(total, 2),
            "count": n,
            
            # 分布统计
            "median": round(median, 2),
            "std": round(std, 2),
            "cv": round(cv, 2),  # 变异系数 %
            "p5": round(p5, 2),
            "p25": round(p25, 2),
            "p75": round(p75, 2),
            "p95": round(p95, 2),
            
            # 趋势分析
            "trend": trend,
            "change_rate": round(change_rate, 2),  # 变化率 %
            
            # 异常检测
            "anomaly_count": anomaly_count,
            "anomaly_ratio": round(anomaly_count / n * 100, 2),  # 异常比例 %
        }
        
        # 时间分布（如果有）
        if time_distribution:
            result["time_distribution"] = time_distribution
        
        return result

    def fetch_sync(
        self,
        collections: List[str],
        devices: List[str],
        tgs: Optional[List[str]],
        start_time: datetime,
        end_time: datetime,
        tags: Optional[List[str]] = None,
        page: int = 1,
        page_size: int = 0,
        value_filter: Optional[Dict] = None
    ) -> SensorDataResult:
        """Synchronous wrapper for fetch_parallel."""
        try:
            # 尝试获取当前事件循环
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果事件循环正在运行（如在 FastAPI 中），使用 run_in_executor
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        self._fetch_sync_blocking,
                        collections, devices, tgs, start_time, end_time, tags, page, page_size, value_filter
                    )
                    return future.result()
            else:
                # 如果没有运行的事件循环，使用 asyncio.run
                return asyncio.run(
                    self.fetch_parallel(
                        collections=collections,
                        devices=devices,
                        tgs=tgs,
                        start_time=start_time,
                        end_time=end_time,
                        tags=tags,
                        page=page,
                        page_size=page_size,
                        value_filter=value_filter
                    )
                )
        except RuntimeError:
            # 如果没有事件循环，创建新的
            return asyncio.run(
                self.fetch_parallel(
                    collections=collections,
                    devices=devices,
                    tgs=tgs,
                    start_time=start_time,
                    end_time=end_time,
                    tags=tags,
                    page=page,
                    page_size=page_size,
                    value_filter=value_filter
                )
            )
    
    def _fetch_sync_blocking(
        self,
        collections: List[str],
        devices: List[str],
        tgs: Optional[List[str]],
        start_time: datetime,
        end_time: datetime,
        tags: Optional[List[str]] = None,
        page: int = 1,
        page_size: int = 0,
        value_filter: Optional[Dict] = None
    ) -> SensorDataResult:
        """在新线程中运行异步代码"""
        return asyncio.run(
            self.fetch_parallel(
                collections=collections,
                devices=devices,
                tgs=tgs,
                start_time=start_time,
                end_time=end_time,
                tags=tags,
                page=page,
                page_size=page_size,
                value_filter=value_filter
            )
        )


    def execute_aggregation_pipeline(
        self,
        collection_name: str,
        pipeline: List[Dict[str, Any]]
    ) -> SensorDataResult:
        """
        执行 MongoDB Aggregation Pipeline
        
        Args:
            collection_name: 集合名称
            pipeline: 聚合管道
        
        Returns:
            SensorDataResult
        """
        try:
            collection = self.db[collection_name]
            
            # 执行聚合
            cursor = list(collection.aggregate(pipeline))
            
            # 处理结果
            for doc in cursor:
                if "_id" in doc:
                    doc["_id"] = str(doc["_id"])
            
            # 计算统计信息
            statistics = None
            if cursor:
                values = []
                for doc in cursor:
                    for key in ["total", "average", "max", "min", "val", "diff"]:
                        if key in doc and doc[key] is not None:
                            try:
                                values.append(float(doc[key]))
                            except:
                                pass
                            break
                
                if values:
                    statistics = {
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values),
                        "count": len(cursor)
                    }
            
            # 构建查询信息
            import json
            query_info = {
                "type": "MongoDB Aggregation",
                "database": self.database_name,
                "collections": [collection_name],
                "pipeline": pipeline,
                "query_string": f"db.{collection_name}.aggregate({json.dumps(pipeline, default=str, ensure_ascii=False)})"
            }
            
            return SensorDataResult(
                data=cursor,
                total_count=len(cursor),
                is_sampled=False,
                statistics=statistics,
                failed_collections=[],
                query_info=query_info,
                page=1,
                page_size=len(cursor),
                total_pages=1,
                has_more=False
            )
            
        except Exception as e:
            logger.error(f"执行聚合管道失败: {e}")
            return SensorDataResult(
                data=[],
                total_count=0,
                is_sampled=False,
                statistics=None,
                failed_collections=[collection_name],
                query_info={"error": str(e)},
                page=1,
                page_size=0,
                total_pages=1,
                has_more=False
            )

    def fetch_aggregated_sync(
        self,
        collections: List[str],
        query: Dict[str, Any],
        data_level: str,
        page: int = 1,
        page_size: int = 0
    ) -> SensorDataResult:
        """
        同步查询聚合数据（日/月/年）
        
        实际预聚合表结构：
        - day_data_*: deviceName, date(YYYY-MM-DD), ep/max/min/avg, tag
        - month_data_*: deviceName, date(YYYY-MM), ep, tag
        - year_data_*: deviceName, date(YYYY), ep, tag
        
        Args:
            collections: 集合名列表
            query: MongoDB 查询条件
            data_level: 数据级别 (daily/monthly/yearly)
            page: 页码
            page_size: 每页记录数
        
        Returns:
            SensorDataResult
        """
        if not collections:
            return SensorDataResult(data=[], total_count=0)
        
        # 查询信息
        query_info = {
            "type": "MongoDB Pre-aggregated",
            "database": self.database_name,
            "collections": collections,
            "query": query,
            "data_level": data_level,
            "query_string": f'db.{collections[0]}.find({query})'
        }
        
        all_data = []
        failed_collections = []
        
        for coll_name in collections:
            try:
                collection = self.db[coll_name]
                cursor = list(collection.find(query).sort([("logTime", 1), ("_id", 1)]))  # 按时间和_id排序，确保稳定
                
                # 处理文档 - 适配实际预聚合表结构
                for doc in cursor:
                    if "_id" in doc:
                        doc["_id"] = str(doc["_id"])
                    
                    # 统一字段名 - deviceName -> device
                    if "deviceName" in doc:
                        doc["device"] = doc["deviceName"]
                    
                    # 所有预聚合表都使用 date 字段，只是格式不同
                    # daily: YYYY-MM-DD, monthly: YYYY-MM, yearly: YYYY
                    doc["logTime"] = doc.get("date", "")
                    
                    # 获取值字段 - 尝试多种可能的字段名
                    # 不同的预聚合表使用不同的值字段:
                    # - ep: 电量表 (day_data_ep, day_data_elec)
                    # - avg/max/min: 电流/电压/温度/湿度表
                    # - p: 功率表
                    # - qf: 功率因数表
                    # - power: 光伏发电量/负载电量表
                    # - loadrate: 负载率表
                    # - val: 频率表 (day_data_f)
                    val = None
                    for field in ["ep", "avg", "p", "qf", "power", "loadrate", "max", "min", "total_val", "avg_val", "val"]:
                        if field in doc and doc[field] is not None:
                            try:
                                val = float(doc[field])
                                break
                            except (ValueError, TypeError):
                                continue
                    doc["val"] = val if val is not None else 0
                    
                    # 添加额外的统计字段（如果存在）
                    for stat_field in ["max", "min", "avg", "peak", "plain", "valley", "power", "p", "qf", "epPeak", "epPlain", "epValley"]:
                        if stat_field in doc:
                            try:
                                doc[f"stat_{stat_field}"] = float(doc[stat_field])
                            except (ValueError, TypeError):
                                pass
                    
                    # 添加解析时间用于排序
                    log_time = doc.get("logTime", "")
                    if log_time:
                        try:
                            if data_level == "yearly" or len(log_time) == 4:
                                # YYYY 格式
                                doc["_parsed_time"] = datetime(int(log_time), 1, 1)
                            elif data_level == "monthly" or len(log_time) == 7:
                                # YYYY-MM 格式
                                year, month = log_time.split("-")
                                doc["_parsed_time"] = datetime(int(year), int(month), 1)
                            else:
                                # YYYY-MM-DD 格式
                                doc["_parsed_time"] = datetime.strptime(log_time, "%Y-%m-%d")
                        except:
                            doc["_parsed_time"] = datetime.min
                
                all_data.extend(cursor)
                
            except Exception as e:
                failed_collections.append(coll_name)
                logger.warning("aggregate.collection.failed collection=%s error=%s", coll_name, e)
        
        # 排序
        all_data = self._sort_by_time(all_data)
        
        total_count = len(all_data)
        statistics = self._compute_aggregated_statistics(all_data) if total_count > 0 else None
        
        # 分页处理
        total_pages = 1
        has_more = False
        is_sampled = False
        
        if page_size > 0 and total_count > page_size:
            total_pages = (total_count + page_size - 1) // page_size
            page = max(1, min(page, total_pages))
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            all_data = all_data[start_idx:end_idx]
            has_more = page < total_pages
        elif total_count > self.max_records:
            all_data = self._downsample(all_data, self.max_records)
            is_sampled = True
        
        return SensorDataResult(
            data=all_data,
            total_count=total_count,
            is_sampled=is_sampled,
            statistics=statistics,
            failed_collections=failed_collections,
            query_info=query_info,
            page=page,
            page_size=page_size if page_size > 0 else total_count,
            total_pages=total_pages,
            has_more=has_more
        )
    
    def _compute_aggregated_statistics(
        self, 
        data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """计算聚合数据的统计摘要"""
        if not data:
            return {"min": 0.0, "max": 0.0, "avg": 0.0, "count": 0}
        
        values = []
        for record in data:
            val = record.get("val")
            if val is not None:
                try:
                    values.append(float(val))
                except (TypeError, ValueError):
                    continue
        
        if not values:
            return {"min": 0.0, "max": 0.0, "avg": 0.0, "count": len(data)}
        
        return {
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "sum": sum(values),
            "count": len(data)
        }
