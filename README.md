# sqlmongo

面向设备时序数据查询的智能问答项目。它支持用自然语言提问，结合 MongoDB 时序数据、MySQL 元数据、语义检索与前端界面，完成设备匹配、数据执行、洞察生成和回答组织。

## 项目定位

`sqlmongo` 的目标是把“用户问题 -> QueryPlan -> 数据库执行 -> 回答风格组织”收敛成一条稳定、可观测、可维护的主流程。

当前项目重点包括：

- 中文自然语言查询设备、项目和传感器数据
- 基于 `QueryPlan` 的 DAG / 非 DAG 统一编排
- 设备重码澄清、会话作用域记忆与重置
- MongoDB 分片路由、分页、聚合、对比、异常点分析
- 结合 `FAISS` 与 `Chroma` 的语义检索和候选召回
- Web 页面流式返回、执行过程面板、表格预览和智能分析

## 核心能力

- **自然语言问答**：直接提问设备用电量、电流、电压、异常、对比等问题
- **QueryPlan 主流程**：统一走“问题解析 -> 数据执行 -> 回答组织”
- **重码设备澄清**：命中多个同码设备时，先让用户确认，不默认跨项目汇总
- **会话级作用域记忆**：同一会话内已确认的 alias 可继承，也支持更换和清空
- **单相 / 多相标签查询**：例如 `ua`、`ua/ub`、`ia/ib/ic` 可按请求标签精确执行
- **可观测性**：前端显示执行步骤、用时、表格和智能分析卡片

## 系统架构

主要链路如下：

```text
用户问题
  -> QueryPlan 解析
  -> 设备 / 项目 / 作用域解析
  -> 集合路由与数据分片
  -> MongoDB / MySQL 执行
  -> 统计、洞察与结果综合
  -> answer_style 组织回答
```

`.env` 中的 `ORCHESTRATOR_TYPE` 决定默认编排模式：

- `dag`：基于 `LangGraph` 的 DAG 编排
- `react`：保留的非 DAG 编排链路

## 技术栈

- **后端**：`FastAPI`
- **编排**：`LangGraph`
- **数据库**：`MongoDB`、`MySQL`
- **语义检索**：`FAISS`、`Chroma`
- **模型接入**：兼容 OpenAI 风格接口的 LLM / Embedding / Rerank 服务
- **前端**：静态单页 + API / SSE 流式交互

## 目录结构

```text
.
?? src/
?  ?? agent/                # QueryPlan、编排器、节点实现
?  ?? analysis/             # 统计与洞察生成
?  ?? entity_resolver/      # Chroma 实体解析
?  ?? fetcher/              # MongoDB 数据获取
?  ?? metadata/             # MySQL 元数据访问
?  ?? router/               # 路由和分片逻辑
?  ?? semantic_layer/       # 语义层和 FAISS 检索
?  ?? tools/                # 传感器与设备工具
?  ?? config.py             # 统一配置入口
?? web/
?  ?? app.py                # FastAPI 应用
?  ?? index.html            # 前端页面
?? scripts/                 # 构建、诊断、验证脚本
?? tests/                   # 自动化和手工测试
?? docs/                    # 设计和说明文档
?? data/                    # 本地语义索引和解析数据
?? run_backend.py           # 后端启动脚本
?? run_frontend.py          # 前端启动脚本
?? README.md
```

## data 目录说明

当前运行时会直接使用以下内容：

- `data/chroma_entity_resolver/`
  - Chroma 存储，用于设备候选解析
- `data/semantic_layer.faiss`
  - FAISS 语义检索索引
- `data/semantic_layer_metadata.json`
  - 与 FAISS 索引配套的元数据映射

以下文件主要用于离线构建：

- `data/semantic_entries.json`
  - 构建 FAISS 索引的原始语义条目

以下文件主要是分析产物：

- `data/mongodb_analysis.json`

## 环境要求

- `Python 3.10+`
- 可连接的 `MongoDB`
- 可连接的 `MySQL`
- 兼容 OpenAI 风格接口的 LLM 服务
- 可选的 Embedding / Rerank 服务

## 快速开始

### 1. 创建虚拟环境并安装依赖

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

### 2. 配置环境变量

复制 `env_example` 为 `.env`，再根据本地环境填写。

建议至少检查这些配置项：

```bash
MONGODB_HOST=
MONGODB_PORT=
MONGODB_USER=
MONGODB_PASSWORD=
MONGODB_DATABASE=

MYSQL_HOST=
MYSQL_PORT=
MYSQL_USER=
MYSQL_PASSWORD=

LLM_BASE_URL=
LLM_MODEL=
LLM_API_KEY=

EMBEDDING_BASE_URL=
EMBEDDING_MODEL=
RERANK_BASE_URL=
RERANK_MODEL=

ORCHESTRATOR_TYPE=dag
FAISS_INDEX_PATH=data/semantic_layer.faiss
```

> `.env` 已被忽略，不应提交真实密钥、密码或内网地址。

### 3. 启动服务

```bash
python run_backend.py
python run_frontend.py
```

默认访问地址：

- 前端：`http://localhost:3000`
- 后端 API：`http://localhost:8080`

## 使用示例

### 自然语言问题

```text
a1_b9 设备 2024 年 1 月的电量数据
查询 a2_b1 在 2024 年 1 月 1 日的 ua 是多少
a1_b9 与 b1_b14 哪个用电更多？
有哪些项目可用？
搜索包含 b9 的设备
```

### API 调用示例

```python
import requests

response = requests.post(
    'http://localhost:8080/api/query',
    json={
        'device_codes': ['a1_b9'],
        'start_time': '2024-01-01',
        'end_time': '2024-01-31',
        'data_type': 'ep',
        'page': 1,
        'page_size': 50,
    },
)

payload = response.json()
print(payload.get('total_count'))
```

## 语义索引构建

如果需要重建语义资产，可以按以下顺序执行：

### 1. 生成语义条目

```bash
python scripts/build_semantic_layer.py
```

生成：`data/semantic_entries.json`

### 2. 构建 FAISS 索引

```bash
python scripts/build_vector_index.py
```

生成：

- `data/semantic_layer.faiss`
- `data/semantic_layer_metadata.json`

## 测试与验证

常用命令：

```bash
pytest
python scripts/validation/check_actual_data.py
python scripts/benchmark_semantic_search.py
```

手工测试资料位于：

- `tests/manual/README.md`
- `tests/manual/30道测试题目.md`

## 相关文档

- 查询路由分层说明：`docs/query_routing_tiers.md`
- 语义层配置：`config/semantic_layer.yaml`

## 开发说明

当前代码已经围绕 `QueryPlan` 做了较多收敛，主要包括：

- DAG / 非 DAG 共用 QueryPlan 上下文
- 共用时间解析、设备澄清、作用域记忆
- 单相 / 多相标签查询与前端展示口径统一
- 候选匹配、澄清交互和响应速度持续优化

如果要继续扩展，建议优先关注以下文件：

- `src/agent/query_plan.py`
- `src/agent/query_plan_state.py`
- `src/agent/query_time_range.py`
- `src/agent/dag_orchestrator.py`
- `src/agent/orchestrator.py`

## 许可证

当前仓库未显式声明开源许可证。如果之后要公开发布，建议补充 `LICENSE` 文件并明确使用范围。
