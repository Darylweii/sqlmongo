"""
设备语义搜索服务

基于 FAISS 向量索引实现设备、项目、指标的语义搜索。
用于从用户自然语言查询中匹配设备ID、项目信息和数据指标。
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

import numpy as np
import httpx

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """Rerank 结果"""
    index: int
    text: str
    score: float


class LocalReranker:
    """本地 Rerank 服务（OpenAI 兼容接口）"""
    
    def __init__(self, base_url: str, model: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        # 禁用代理
        import os
        for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
            os.environ.pop(key, None)
        os.environ['NO_PROXY'] = '*'
    
    def rerank(self, query: str, documents: List[str], top_n: Optional[int] = None) -> List[RerankResult]:
        """重排序文档"""
        if not documents:
            return []
        
        try:
            # 尝试 /rerank 端点（vLLM reranker 格式）
            response = httpx.post(
                f'{self.base_url}/rerank',
                headers={'Content-Type': 'application/json'},
                json={
                    'model': self.model,
                    'query': query,
                    'documents': documents,
                    'top_n': top_n or len(documents),
                },
                timeout=self.timeout,
            )
            
            if response.status_code == 200:
                data = response.json()
                results = []
                for item in data.get('results', []):
                    results.append(RerankResult(
                        index=item.get('index', 0),
                        text=documents[item.get('index', 0)] if item.get('index', 0) < len(documents) else '',
                        score=item.get('relevance_score', 0.0),
                    ))
                return results
            else:
                logger.warning(f"Rerank API 错误: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Rerank 调用失败: {e}")
            return []
    
    def close(self):
        pass


@dataclass
class DeviceSearchResult:
    """设备搜索结果"""
    type: str  # device, project, metric
    score: float
    semantic_text: str
    metadata: Dict[str, Any]


class DeviceSemanticSearch:
    """
    设备语义搜索服务
    
    使用 FAISS 向量索引进行语义搜索，支持：
    - 设备名称 → device_id 映射
    - 项目名称 → project_id 映射
    - 指标名称 → tag 映射
    """
    
    def __init__(
        self,
        index_path: str = "data/semantic_layer.faiss",
        metadata_path: str = "data/semantic_layer_metadata.json",
        embedding_base_url: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_dimensions: int = 1024,
        rerank_base_url: Optional[str] = None,
        rerank_model: Optional[str] = None,
        use_rerank: bool = True,
        dashscope_api_key: Optional[str] = None,  # 保留兼容性
    ):
        """
        初始化语义搜索服务
        
        Args:
            index_path: FAISS 索引文件路径
            metadata_path: 元数据 JSON 文件路径
            embedding_base_url: Embedding 服务地址
            embedding_model: Embedding 模型名称
            embedding_dimensions: 向量维度
            rerank_base_url: Rerank 服务地址
            rerank_model: Rerank 模型名称
            use_rerank: 是否使用 rerank 重排序（默认 True）
            dashscope_api_key: DashScope API Key（兼容旧配置）
        """
        self.index_path = index_path
        self.metadata_path = metadata_path
        
        # Embedding 配置
        self.embedding_base_url = embedding_base_url or os.getenv("EMBEDDING_BASE_URL", "http://172.16.1.75:8008/v1")
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "qwen3-embedding")
        self.embedding_dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", embedding_dimensions))
        
        # Rerank 配置
        self.rerank_base_url = rerank_base_url or os.getenv("RERANK_BASE_URL", "http://172.16.1.75:8012/v1")
        self.rerank_model = rerank_model or os.getenv("RERANK_MODEL", "qwen3-reranker")
        self.use_rerank = use_rerank
        
        # 兼容旧配置
        self.api_key = dashscope_api_key or os.getenv("DASHSCOPE_API_KEY")
        
        self._index = None
        self._metadata = None
        self._initialized = False
        self._reranker = None
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized
    
    def initialize(self) -> bool:
        """初始化索引和元数据"""
        try:
            import faiss
            
            # 检查文件是否存在
            if not Path(self.index_path).exists():
                logger.error(f"索引文件不存在: {self.index_path}")
                return False
            
            if not Path(self.metadata_path).exists():
                logger.error(f"元数据文件不存在: {self.metadata_path}")
                return False
            
            # 加载索引
            self._index = faiss.read_index(self.index_path)
            logger.info(f"加载 FAISS 索引: {self._index.ntotal} 个向量")
            
            # 加载元数据
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self._metadata = json.load(f)
            logger.info(f"加载元数据: {len(self._metadata)} 条")
            
            # 初始化 reranker（使用本地服务）
            if self.use_rerank:
                self._reranker = LocalReranker(
                    base_url=self.rerank_base_url,
                    model=self.rerank_model,
                )
                logger.info(f"Rerank 已启用: {self.rerank_base_url}")
            
            self._initialized = True
            return True
            
        except Exception as e:
            logger.exception(f"初始化失败: {e}")
            return False
    
    def _get_embedding(self, text: str, max_retries: int = 3) -> Optional[List[float]]:
        """获取文本向量（使用本地 OpenAI 兼容接口）"""
        import os
        # 禁用代理
        for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
            os.environ.pop(key, None)
        os.environ['NO_PROXY'] = '*'
        
        for retry in range(max_retries):
            try:
                response = httpx.post(
                    f'{self.embedding_base_url}/embeddings',
                    headers={'Content-Type': 'application/json'},
                    json={
                        'model': self.embedding_model,
                        'input': text,
                    },
                    timeout=30.0,
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result['data'][0]['embedding']
                else:
                    logger.warning(f"Embedding API 错误: {response.status_code} - {response.text}")
                    
            except Exception as e:
                if retry < max_retries - 1:
                    time.sleep(1)
                else:
                    logger.error(f"获取向量失败: {e}")
        
        return None
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        type_filter: Optional[str] = None,
        min_score: float = 0.3,
        use_rerank: Optional[bool] = None,
    ) -> List[DeviceSearchResult]:
        """
        语义搜索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            type_filter: 类型过滤 (device/project/metric)
            min_score: 最小分数阈值
            use_rerank: 是否使用 rerank（None 使用默认设置）
        
        Returns:
            搜索结果列表
        """
        if not self._initialized:
            logger.warning("语义搜索未初始化")
            return []
        
        try:
            import faiss
            
            # 获取查询向量
            embedding = self._get_embedding(query)
            if embedding is None:
                return []
            
            # 转换为 numpy 数组
            query_vector = np.array([embedding], dtype=np.float32)
            faiss.normalize_L2(query_vector)
            
            # 搜索（多取一些以便过滤和 rerank）
            # 本地 embedding 模型语义理解较弱，增大召回量让 rerank 发挥作用
            should_rerank = (use_rerank if use_rerank is not None else self.use_rerank) and self._reranker
            recall_k = top_k * 30 if should_rerank else (top_k * 3 if type_filter else top_k)
            recall_k = min(recall_k, 200)  # 限制最大召回量
            scores, indices = self._index.search(query_vector, recall_k)
            
            # 构建候选结果
            candidates = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self._metadata):
                    continue
                
                if score < min_score:
                    continue
                
                entry = self._metadata[idx]
                
                # 类型过滤
                if type_filter and entry['type'] != type_filter:
                    continue
                
                candidates.append(DeviceSearchResult(
                    type=entry['type'],
                    score=float(score),
                    semantic_text=entry['semantic_text'],
                    metadata=entry['metadata'],
                ))
            
            # 如果启用 rerank 且有候选结果
            if should_rerank and candidates:
                candidates = self._rerank_results(query, candidates, top_k)
            else:
                # 不 rerank，直接截取 top_k
                candidates = candidates[:top_k]
            
            return candidates
            
        except Exception as e:
            logger.exception(f"搜索失败: {e}")
            return []
    
    def _rerank_results(
        self,
        query: str,
        candidates: List[DeviceSearchResult],
        top_k: int,
    ) -> List[DeviceSearchResult]:
        """
        使用 rerank 模型重排序候选结果
        
        Args:
            query: 原始查询
            candidates: 候选结果列表
            top_k: 返回数量
        
        Returns:
            重排序后的结果列表
        """
        if not self._reranker or not candidates:
            return candidates[:top_k]
        
        try:
            # 提取文档文本用于 rerank
            documents = [c.semantic_text for c in candidates]
            
            # 调用 rerank API
            rerank_results = self._reranker.rerank(
                query=query,
                documents=documents,
                top_n=top_k,
            )
            
            # 根据 rerank 结果重新排序
            reranked = []
            for rr in rerank_results:
                if 0 <= rr.index < len(candidates):
                    candidate = candidates[rr.index]
                    # 更新分数为 rerank 分数
                    reranked.append(DeviceSearchResult(
                        type=candidate.type,
                        score=rr.score,  # 使用 rerank 分数
                        semantic_text=candidate.semantic_text,
                        metadata=candidate.metadata,
                    ))
            
            logger.debug(f"Rerank 完成: {len(candidates)} -> {len(reranked)} 条结果")
            return reranked
            
        except Exception as e:
            logger.warning(f"Rerank 失败，使用原始排序: {e}")
            return candidates[:top_k]
    
    def search_devices(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        搜索设备（混合检索：关键词优先 + 向量补充）
        
        Args:
            query: 查询文本（设备名称、描述等）
            top_k: 返回数量
            min_score: 最小分数
        
        Returns:
            设备信息列表，每个包含 device_id, device_name, project_name, score
        """
        # 1. 先用关键词匹配（字符串包含）
        keyword_results = self._keyword_search_devices(query, top_k * 2)
        
        # 2. 如果关键词匹配到结果，优先使用关键词结果
        if keyword_results:
            # 关键词匹配的结果已经是精确匹配，给予高分
            for r in keyword_results:
                r['score'] = max(r['score'], 0.9)  # 关键词匹配至少 0.9 分
            
            # 如果关键词结果足够，直接返回
            if len(keyword_results) >= top_k:
                return keyword_results[:top_k]
            
            # 关键词结果不足，用向量搜索补充
            vector_results = self.search(query, top_k=top_k * 3, type_filter='device', min_score=min_score)
            
            # 合并结果（去重，关键词结果优先）
            seen_ids = {r['device_id'] for r in keyword_results}
            for r in vector_results:
                if r.metadata.get('device_id') not in seen_ids:
                    keyword_results.append({
                        'device_id': r.metadata.get('device_id'),
                        'device_name': r.metadata.get('device_name'),
                        'device_type': r.metadata.get('device_type'),
                        'project_id': r.metadata.get('project_id'),
                        'project_name': r.metadata.get('project_name'),
                        'tg': r.metadata.get('tg'),
                        'score': r.score * 0.8,  # 向量搜索结果降权
                    })
                    seen_ids.add(r.metadata.get('device_id'))
            
            # 按分数排序（关键词结果会排在前面）
            keyword_results.sort(key=lambda x: x['score'], reverse=True)
            return keyword_results[:top_k]
        
        # 3. 没有关键词匹配，完全依赖向量搜索
        vector_results = self.search(query, top_k=top_k * 3, type_filter='device', min_score=min_score)
        
        results = [
            {
                'device_id': r.metadata.get('device_id'),
                'device_name': r.metadata.get('device_name'),
                'device_type': r.metadata.get('device_type'),
                'project_id': r.metadata.get('project_id'),
                'project_name': r.metadata.get('project_name'),
                'tg': r.metadata.get('tg'),
                'score': r.score,
            }
            for r in vector_results
        ]
        
        # 对纯向量搜索结果使用 rerank
        if self._reranker and results:
            results = self._rerank_device_results(query, results, top_k)
        
        return results[:top_k]
    
    def _keyword_search_devices(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """关键词搜索设备（字符串包含匹配）"""
        if not self._metadata:
            return []
        
        results = []
        query_lower = query.lower()
        
        for entry in self._metadata:
            if entry['type'] != 'device':
                continue
            
            device_name = entry['metadata'].get('device_name', '').lower()
            semantic_text = entry.get('semantic_text', '').lower()
            
            # 检查是否包含查询词
            if query_lower in device_name or query_lower in semantic_text:
                # 计算匹配分数（越短的名称匹配度越高）
                score = len(query) / max(len(device_name), 1)
                results.append({
                    'device_id': entry['metadata'].get('device_id'),
                    'device_name': entry['metadata'].get('device_name'),
                    'device_type': entry['metadata'].get('device_type'),
                    'project_id': entry['metadata'].get('project_id'),
                    'project_name': entry['metadata'].get('project_name'),
                    'tg': entry['metadata'].get('tg'),
                    'score': min(score, 1.0),
                    'match_type': 'keyword',
                })
        
        # 按分数排序
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def _rerank_device_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """对设备结果进行 rerank"""
        if not self._reranker or not results:
            return results
        
        try:
            documents = [r['device_name'] for r in results]
            rerank_results = self._reranker.rerank(query, documents, top_n=top_k)
            
            reranked = []
            for rr in rerank_results:
                if 0 <= rr.index < len(results):
                    result = results[rr.index].copy()
                    result['score'] = rr.score
                    reranked.append(result)
            
            return reranked
        except Exception as e:
            logger.warning(f"Rerank 设备结果失败: {e}")
            return results[:top_k]
    
    def search_projects(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        搜索项目
        
        Args:
            query: 查询文本（项目名称等）
            top_k: 返回数量
            min_score: 最小分数
        
        Returns:
            项目信息列表
        """
        results = self.search(query, top_k=top_k, type_filter='project', min_score=min_score)
        
        return [
            {
                'project_id': r.metadata.get('project_id'),
                'project_name': r.metadata.get('project_name'),
                'project_code': r.metadata.get('project_code'),
                'score': r.score,
            }
            for r in results
        ]
    
    def search_metrics(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        搜索指标类型
        
        Args:
            query: 查询文本（指标名称、描述等）
            top_k: 返回数量
            min_score: 最小分数
        
        Returns:
            指标信息列表
        """
        results = self.search(query, top_k=top_k, type_filter='metric', min_score=min_score)
        
        return [
            {
                'tag': r.metadata.get('tag'),
                'name': r.metadata.get('name'),
                'unit': r.metadata.get('unit'),
                'description': r.metadata.get('description'),
                'score': r.score,
            }
            for r in results
        ]
    
    def resolve_device(self, query: str) -> Optional[Dict[str, Any]]:
        """
        解析设备：从自然语言查询中找到最匹配的设备
        
        Args:
            query: 用户查询（可能包含设备名称）
        
        Returns:
            最匹配的设备信息，或 None
        """
        devices = self.search_devices(query, top_k=1, min_score=0.4)
        return devices[0] if devices else None
    
    def resolve_metric(self, query: str) -> Optional[Dict[str, Any]]:
        """
        解析指标：从自然语言查询中找到最匹配的指标类型
        
        Args:
            query: 用户查询（可能包含指标描述）
        
        Returns:
            最匹配的指标信息，或 None
        """
        metrics = self.search_metrics(query, top_k=1, min_score=0.4)
        return metrics[0] if metrics else None
    
    def close(self):
        """关闭资源"""
        if self._reranker:
            self._reranker.close()
            self._reranker = None
        self._index = None
        self._metadata = None
        self._initialized = False


# 全局单例
_device_search: Optional[DeviceSemanticSearch] = None


def get_device_search() -> DeviceSemanticSearch:
    """获取全局设备搜索实例"""
    global _device_search
    
    if _device_search is None:
        _device_search = DeviceSemanticSearch()
        _device_search.initialize()
    
    return _device_search


def search_devices(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """便捷函数：搜索设备"""
    return get_device_search().search_devices(query, top_k=top_k)


def search_metrics(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """便捷函数：搜索指标"""
    return get_device_search().search_metrics(query, top_k=top_k)
