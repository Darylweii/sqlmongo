# AI Data Router Agent

智能数据查询助手，基于 LangGraph 的多阶段编排流程，支持自然语言查询设备时序数据。

## 功能特性

- **自然语言查询**：使用中文自然语言查询设备数据
- **语义搜索**：基于向量检索进行设备与指标匹配
- **数据可视化**：支持表格展示与统计分析
- **高性能分页**：支持数据库级分页，适合大数据量查询
- **对话式交互**：支持多轮对话与上下文理解
- **对比分析**：支持多设备数据对比查询

## 快速开始

### 1. 环境要求

- Python 3.10+
- MongoDB 3.4+
- 本地或兼容 OpenAI 接口的 LLM 服务
- Embedding / Rerank 服务（可选，语义层启用时需要）

### 2. 安装依赖

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -e .
```

### 3. 配置环境变量

复制 `env_example` 为 `.env` 后按实际环境填写：

```bash
# MongoDB
MONGODB_HOST=127.0.0.1
MONGODB_PORT=27017
MONGODB_USER=your_mongodb_user_here
MONGODB_PASSWORD=your_mongodb_password_here
MONGODB_AUTH_SOURCE=admin
MONGODB_DATABASE=your_mongodb_database_here

# LLM
LLM_BASE_URL=http://127.0.0.1:8002/v1
LLM_MODEL=your_llm_model_here
LLM_API_KEY=your_llm_api_key_here

# Embedding
EMBEDDING_BASE_URL=http://127.0.0.1:8008/v1
EMBEDDING_MODEL=your_embedding_model_here

# Rerank
RERANK_BASE_URL=http://127.0.0.1:8012/v1
RERANK_MODEL=your_rerank_model_here
```

> 安全提示：`env_example` 只应包含占位符，禁止提交真实账号、密码、API Key 或内网地址。真实配置仅保存在本地 `.env` 中。

### 4. 启动服务

方式 1：一键启动（如果有对应批处理）

```bash
start_all.bat
```

方式 2：分别启动

```bash
python run_backend.py
python run_frontend.py
```

### 5. 访问地址

- 前端页面：`http://localhost:3000`
- 后端 API：`http://localhost:8080`

## 路由分层

- 查询路由分层说明：`docs/query_routing_tiers.md`

## 使用示例

### 自然语言查询

```text
查询 a1_b9 设备今天的电流数据
查询 a1_b9 设备 2024 年 1 月的电量数据
对比 a1_b9 和 b1_b14 的用电量
搜索包含 b9 的设备
有哪些项目可用？
```

### API 调用

```python
import requests

response = requests.post('http://localhost:8080/api/query', json={
    "device_codes": ["a1_b9"],
    "start_time": "2024-01-01",
    "end_time": "2024-01-31",
    "data_type": "ep",
    "page": 1,
    "page_size": 50,
})

data = response.json()
print(f"总记录数: {data['total_count']}")
print(f"当前页: {data['page']}/{data['total_pages']}")
```

## 项目结构

```text
.
├── src/
│   ├── agent/               # 编排与节点逻辑
│   ├── semantic_layer/      # 语义层
│   ├── fetcher/             # MongoDB 数据获取
│   ├── metadata/            # MySQL 元数据访问
│   ├── router/              # 集合路由
│   └── config.py            # 配置管理
├── web/
│   ├── app.py               # FastAPI 后端
│   └── index.html           # 前端页面
├── scripts/                 # 诊断与工具脚本
├── tests/                   # 测试与手工验证脚本
├── data/                    # 本地索引与数据文件
├── run_backend.py
├── run_frontend.py
└── README.md
```

## 核心技术

### 1. LangGraph DAG 编排

典型流程：

```text
用户查询 -> 意图解析 -> 元数据映射 -> 数据路由 -> 数据获取 -> 结果合成
```

### 2. 语义搜索

- 向量库：FAISS
- Embedding 模型：可配置
- Rerank 模型：可配置

### 3. 数据分页

系统支持 MongoDB 层分页，避免一次性加载过多数据：

```python
collection.find(query) \
    .sort([("logTime", 1), ("_id", 1)]) \
    .skip((page - 1) * page_size) \
    .limit(page_size)
```

### 4. 对比查询

支持多设备数据对比，例如：

```text
对比 a1_b9 和 b1_b14 的用电量
```

## 开发与调试

### 常用脚本

```bash
python run_backend.py
python scripts/validation/check_actual_data.py
python tests/manual/legacy/test_full_query.py
```

### 手工测试

```bash
.venv\Scripts\python.exe tests/manual/test_comprehensive.py
```

测试题目见：`tests/manual/30道测试题目.md`

## 测试报告

测试完成后通常会生成：

- `TEST_REPORT_YYYYMMDD_HHMMSS.md`
- `test_results_YYYYMMDD_HHMMSS.json`

建议将这类产物作为本地产出，不纳入源码管理。
