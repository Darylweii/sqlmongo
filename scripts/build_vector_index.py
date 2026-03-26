"""基于 semantic_entries.json 构建本地 FAISS 索引。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def localize_argparse() -> None:
    translations = {
        'usage: ': '\u7528\u6cd5: ',
        'options': '\u53ef\u9009\u53c2\u6570',
        'positional arguments': '\u4f4d\u7f6e\u53c2\u6570',
        'show this help message and exit': '\u663e\u793a\u6b64\u5e2e\u52a9\u4fe1\u606f\u5e76\u9000\u51fa',
    }
    argparse._ = lambda text: translations.get(text, text)


localize_argparse()

import numpy as np
from dotenv import load_dotenv


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.semantic_layer.embedding import DashScopeEmbedding


DEFAULT_EMBEDDING_MODEL = 'text-embedding-v4'


def create_embedding_client(model_name: str | None = None) -> DashScopeEmbedding:
    load_dotenv()
    api_key = os.getenv('DASHSCOPE_API_KEY', '').strip()
    model = (model_name or os.getenv('EMBEDDING_MODEL') or DEFAULT_EMBEDDING_MODEL).strip()
    dimensions = int(os.getenv('EMBEDDING_DIMENSIONS', '1024'))
    return DashScopeEmbedding(api_key=api_key, model=model, dimensions=dimensions)


def _embed_batches(texts: list[str], batch_size: int, max_retries: int, model_name: str | None) -> list[list[float]]:
    all_embeddings: list[list[float]] = []
    with create_embedding_client(model_name) as embedding_client:
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            for attempt in range(1, max_retries + 1):
                try:
                    embeddings = embedding_client.embed_texts(batch)
                    all_embeddings.extend(embeddings)
                    print(
                        f'已处理 {min(start + len(batch), len(texts))}/{len(texts)} 条， '
                        f'批大小 {len(batch)}，模型 {embedding_client.model}'
                    )
                    break
                except Exception as exc:  # noqa: BLE001
                    if attempt >= max_retries:
                        raise
                    sleep_seconds = min(2 ** attempt, 8)
                    print(f'向量生成失败，{sleep_seconds}s 后重试 ({attempt}/{max_retries}): {exc}')
                    time.sleep(sleep_seconds)
    return all_embeddings


def get_embeddings_batch(
    texts: list[str],
    batch_size: int = 10,
    max_retries: int = 3,
    model_name: str | None = None,
) -> list[list[float]]:
    """分批生成向量，必要时回退到 text-embedding-v4。"""
    if not texts:
        return []

    preferred_model = (model_name or os.getenv('EMBEDDING_MODEL') or DEFAULT_EMBEDDING_MODEL).strip()
    try:
        return _embed_batches(texts, batch_size=batch_size, max_retries=max_retries, model_name=preferred_model)
    except Exception as exc:  # noqa: BLE001
        error_text = str(exc)
        should_fallback = preferred_model != DEFAULT_EMBEDDING_MODEL and any(
            token in error_text for token in ('model_not_found', 'does not exist', '404')
        )
        if not should_fallback:
            raise
        print(f'模型 {preferred_model} 不可用，自动回退到 {DEFAULT_EMBEDDING_MODEL}')
        return _embed_batches(texts, batch_size=batch_size, max_retries=max_retries, model_name=DEFAULT_EMBEDDING_MODEL)


def build_faiss_index(
    entries: list[dict[str, Any]],
    output_path: str = 'data/semantic_layer.faiss',
    batch_size: int = 10,
    model_name: str | None = None,
):
    """构建 FAISS 索引，并保存元数据文件。"""
    import faiss

    valid_entries = [entry for entry in entries if str(entry.get('semantic_text') or '').strip()]
    texts = [str(entry['semantic_text']).strip() for entry in valid_entries]
    if not texts:
        raise ValueError('semantic_entries 中不存在可用的 semantic_text')

    print(f'正在为 {len(texts)} 条文本生成向量...')
    embeddings = get_embeddings_batch(texts, batch_size=batch_size, model_name=model_name)
    vectors = np.array(embeddings, dtype=np.float32)
    print(f'向量矩阵形状: {vectors.shape}')

    faiss.normalize_L2(vectors)
    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)
    print(f'已索引向量数: {index.ntotal}')

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_file))
    print(f'FAISS 索引已保存到: {output_file.as_posix()}')

    metadata_path = output_file.with_name(output_file.stem + '_metadata.json')
    metadata_path.write_text(json.dumps(valid_entries, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'索引元数据已保存到: {metadata_path.as_posix()}')
    return index


def test_search(
    query: str,
    top_k: int = 5,
    index_path: str = 'data/semantic_layer.faiss',
    model_name: str | None = None,
) -> None:
    """对已构建索引执行一次简单检索测试。"""
    import faiss

    index_file = Path(index_path)
    metadata_file = index_file.with_name(index_file.stem + '_metadata.json')
    if not index_file.exists() or not metadata_file.exists():
        raise FileNotFoundError(f'未找到索引文件或元数据文件: {index_file} / {metadata_file}')

    index = faiss.read_index(str(index_file))
    metadata = json.loads(metadata_file.read_text(encoding='utf-8'))
    query_embedding = get_embeddings_batch([query], model_name=model_name)[0]
    query_vector = np.array([query_embedding], dtype=np.float32)
    faiss.normalize_L2(query_vector)
    scores, indices = index.search(query_vector, top_k)

    print(f'\n测试查询: {query}')
    print(f'前 {top_k} 个结果:')
    for rank, (score, index_id) in enumerate(zip(scores[0], indices[0]), start=1):
        entry = metadata[index_id]
        print(f"  {rank}. [{entry.get('type')}] {entry.get('semantic_text')}")
        print(f"     分数: {score:.4f}")
        print(f"     元数据: {entry.get('metadata')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='根据语义条目文件 semantic_entries.json 构建 FAISS 索引')
    parser.add_argument('--input', default='data/semantic_entries.json', help='语义条目文件路径')
    parser.add_argument('--output', default=os.getenv('FAISS_INDEX_PATH', 'data/semantic_layer.faiss'), help='FAISS 输出路径')
    parser.add_argument('--batch-size', type=int, default=10, help='向量批大小')
    parser.add_argument('--model', default=os.getenv('EMBEDDING_MODEL', DEFAULT_EMBEDDING_MODEL), help='向量模型名')
    parser.add_argument('--test-query', default='', help='构建完成后可选执行一次测试查询')
    parser.add_argument('--top-k', type=int, default=5, help='返回条数上限')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print(f'错误：未找到语义条目文件: {input_path.as_posix()}')
        print('请先运行 build_semantic_layer.py')
        return

    entries = json.loads(input_path.read_text(encoding='utf-8'))
    print('=' * 60)
    print('正在构建语义层 FAISS 索引')
    print('=' * 60)
    print(f'已加载条目数: {len(entries)}')

    build_faiss_index(entries, args.output, batch_size=args.batch_size, model_name=args.model)

    print('=' * 60)
    print('FAISS 索引构建完成')
    print('=' * 60)

    if args.test_query:
        test_search(args.test_query, top_k=args.top_k, index_path=args.output, model_name=args.model)


if __name__ == '__main__':
    main()
