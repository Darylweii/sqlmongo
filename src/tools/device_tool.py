"""
Device Metadata Tool for LangChain Agent integration.

This module provides the find_device_metadata tool that allows the AI Agent
to query device and project metadata from MySQL database.

适配实际数据库结构:
- 返回字段: device (设备代号), name, device_type, project_id, project_name, project_code_name
"""

from typing import List, Dict, Optional
import logging

from langchain.tools import tool

from src.metadata.metadata_engine import MetadataEngine, DeviceInfo
from src.exceptions import DatabaseConnectionError, MetadataEngineError


logger = logging.getLogger(__name__)

# Global metadata engine instance (to be configured at startup)
_metadata_engine: Optional[MetadataEngine] = None


def configure_metadata_engine(db_connection_string: str, cache_size: int = 1000) -> None:
    """
    Configure the global MetadataEngine instance.
    
    Args:
        db_connection_string: SQLAlchemy database connection string
        cache_size: Maximum number of cached query results
    """
    global _metadata_engine
    _metadata_engine = MetadataEngine(db_connection_string, cache_size)


def get_metadata_engine() -> MetadataEngine:
    """
    Get the configured MetadataEngine instance.
    
    Returns:
        The configured MetadataEngine
    
    Raises:
        RuntimeError: If MetadataEngine is not configured
    """
    if _metadata_engine is None:
        raise RuntimeError(
            "MetadataEngine 未配置。请先调用 configure_metadata_engine()"
        )
    return _metadata_engine


@tool
def find_device_metadata(keyword: str) -> List[Dict]:
    """
    查询设备、资产、项目信息，或获取设备代号以便查询时序日志。
    输入通常是项目名称或设备模糊名。
    
    使用此工具可以：
    - 根据设备名称关键词搜索设备
    - 根据项目名称或项目代号搜索项目下的设备
    - 获取 device (设备代号) 用于后续时序数据查询
    
    Args:
        keyword: 项目名称或设备名称关键词
    
    Returns:
        设备列表，每个设备包含:
        - device: 设备代号 (用于查询 MongoDB 时序数据)
        - name: 设备名称
        - device_type: 设备类型
        - project_id: 项目ID
        - project_name: 项目名称
        - project_code_name: 项目代号
    
    Example:
        >>> find_device_metadata("北京电力")
        [{"device": "b1_b14", "name": "1#变压器", "project_id": "4", ...}]
    """
    if not keyword or not keyword.strip():
        return []
    
    try:
        engine = get_metadata_engine()
        devices: List[DeviceInfo] = engine.search_devices(keyword.strip())
        
        # Convert DeviceInfo objects to dictionaries
        result = []
        for device in devices:
            device_dict = device.to_dict()
            # 确保 device 字段存在
            if device_dict.get("device"):
                result.append(device_dict)
        
        logger.info(f"find_device_metadata: 关键词 '{keyword}' 找到 {len(result)} 个设备")
        return result
        
    except DatabaseConnectionError as e:
        logger.error(f"数据库连接失败: {e}")
        return [{"error": f"数据库连接失败: {str(e)}"}]
    except MetadataEngineError as e:
        logger.error(f"元数据查询错误: {e}")
        return [{"error": f"查询失败: {str(e)}"}]
    except RuntimeError as e:
        logger.error(f"配置错误: {e}")
        return [{"error": str(e)}]
    except Exception as e:
        logger.error(f"未知错误: {e}")
        return [{"error": f"查询时发生未知错误: {str(e)}"}]


def find_device_metadata_with_engine(
    keyword: str, 
    engine: MetadataEngine
) -> List[Dict]:
    """
    使用指定的 MetadataEngine 查询设备元数据。
    
    此函数用于测试或需要自定义 engine 的场景。
    
    Args:
        keyword: 项目名称或设备名称关键词
        engine: MetadataEngine 实例
    
    Returns:
        设备列表，最后一个元素包含查询信息
    """
    if not keyword or not keyword.strip():
        return []
    
    try:
        devices, sql_query = engine.search_devices(keyword.strip())
        
        result = []
        for device in devices:
            device_dict = device.to_dict()
            if device_dict.get("device"):
                result.append(device_dict)
        
        # 添加查询信息到结果
        if result:
            result.append({
                "_query_info": {
                    "type": "MySQL",
                    "sql": sql_query
                }
            })
        
        return result
        
    except (DatabaseConnectionError, MetadataEngineError) as e:
        return [{"error": str(e)}]
