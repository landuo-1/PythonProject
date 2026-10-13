# 企业智能知识库Agent

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.115-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/LangGraph-0.4-orange.svg" alt="LangGraph">
  <img src="https://img.shields.io/badge/LangChain-0.3-orange.svg" alt="LangChain">
  <img src="https://img.shields.io/badge/Streamlit-1.45-red.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/ChromaDB-1.0-purple.svg" alt="ChromaDB">
</p>

> 基于 LangGraph 自纠错工作流 + RAG 混合检索 + 多用户隔离的企业知识库问答系统

## 项目简介

企业智能知识库Agent是一个基于大语言模型的企业知识管理系统。通过 **LangGraph 状态图工作流** 编排意图路由，结合 RAG 混合检索引擎，实现智能化的知识查询和问答服务。系统内置质量评估与自纠错机制，确保回答的准确性和可靠性。

### 核心特性

- **LangGraph 自纠错工作流**：`retrieve → generate → evaluate` 完整闭环，评估不达标时自动重检索重生成
- **RAG 混合检索**：向量检索（ChromaDB + fastembed）+ BM25 关键词检索（bm25s + jieba）+ RRF 融合排序
- **质量评估**：忠实度（来源引用准确性 + 关键词重叠）+ 答案相关性（问题术语命中率），规则检查无额外模型
- **文档两阶段去重**：MD5 哈希精确匹配 + 文件名匹配，支持原上传者覆盖更新
- **多用户隔离**：JWT 认证 + 个人工作区/团队共享库双空间
- **流式对话**：SSE 流式推送，独立线程池（10 workers），客户端断开检测与优雅取消
- **记忆管理**：滑动窗口 + 摘要压缩（压缩后截断历史避免重复压缩）+ 用户偏好注入
- **HyDE 查询改写**：短/模糊查询自动生成假设性文档，提升检索召回率

### 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    前端层 (Streamlit)                        │
│  登录注册 / 对话界面 / 文档管理 / 来源预览 / 质量评估面板      │
└─────────────────────────────────────────────────────────────┘
                              │ HTTP + SSE
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    API层 (FastAPI)                           │
│  JWT认证 / 流式对话 / 文档上传删除 / 知识库管理               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│          LangGraph 自纠错工作流 (agent.py)                    │
│                                                              │
│  START → router → [general / external / external_disabled]   │
│                    └→ retrieve → generate → evaluate          │
│                                  ↑            │               │
│                                  └── retry ───┘               │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   RAG引擎层     │ │   记忆层        │ │   评估层        │
│ 文档解析/分块   │ │ 滑动窗口/摘要   │ │ 来源引用验证    │
│ 向量+BM25检索   │ │ 偏好/知识注入   │ │ 关键词重叠检查  │
│ RRF融合/HyDE    │ │ 知识库感知过滤  │ │ 问题术语匹配    │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

## 快速开始

### 1. 环境准备

```bash
git clone https://github.com/landuo-1/PythonProject.git
cd PythonProject/enterprise-knowledge-agent

python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```env
# MIMO LLM配置（必填）
MIMO_API_KEY=your_mimo_api_key_here
MIMO_BASE_URL=https://token-plan-cn.xiaomimimo.com/v1
MIMO_MODEL=mimo-v2.5

# JWT认证密钥（必填，生产环境请修改为强密钥）
JWT_SECRET_KEY=your-secret-key-change-in-production

# Tavily搜索配置（可选，启用网络搜索功能需要）
TAVILY_API_KEY=your_tavily_api_key_here
```

### 3. 启动服务

```bash
# 终端1：启动后端
python -m backend.main

# 终端2：启动前端
streamlit run frontend/app.py
```

### 4. 访问应用

| 服务 | 地址 | 说明 |
|------|------|------|
| 前端界面 | http://localhost:8501 | 用户交互界面 |
| API文档 | http://localhost:8000/docs | Swagger API文档 |
| 健康检查 | http://localhost:8000/health | 系统状态检查 |

## 使用指南

### 文档上传

1. 在左侧边栏「上传文档」区域选择文件（支持 PDF、Word、TXT、Excel、CSV）
2. 选择存储位置：个人工作区（仅自己可见）或团队共享库（所有用户可见）
3. 可选：勾选「替换已有同名文档」覆盖更新
4. 点击「上传到知识库」

### 智能问答

1. 在底部输入框输入问题
2. 系统自动路由：内部问题走 RAG 检索，外部问题走网络搜索
3. 回答末尾标注参考来源，点击可预览原文片段
4. 启用评估后，回答下方显示质量评分

### 网络搜索

在侧边栏开启「启用网络搜索」后，Agent 可搜索互联网获取实时信息。

## 核心功能详解

### 1. LangGraph 自纠错工作流

```
retrieve(HyDE查询) → generate(LLM) → evaluate(忠实度+相关性)
                                              │
                          ┌───────────────────┼───────────────────┐
                          │                   │                   │
                       通过                 不通过+可重试        不通过+不可重试
                          │                   │                   │
                         END          retrieve(增强查询)→generate→evaluate→END
```

- **retrieve_node**：首次使用 HyDE 改写查询，重试时使用增强查询（基于失败回答优化检索词）
- **generate_node**：LLM 生成回答
- **evaluate_node**：忠实度检查（来源引用是否真实存在）+ 答案相关性检查（是否包含问题关键术语）

### 2. 混合检索引擎

```python
# 向量检索：语义相似性
vector_results = similarity_search_with_relevance_scores(query, k=10)

# BM25检索：关键词匹配
bm25_results = bm25_retriever.retrieve(jieba.lcut(query), k=10)

# RRF融合排序
score = Σ 1/(rrf_k + rank_i)  # rrk_k=60
```

- **BM25 可见性过滤**：独立实现后过滤层，按 owner/visibility 精确匹配
- **chunk_index 元数据**：记录片段在文件内的位置，确保前端预览与引用编号一致
- **可选重排序**：预留 CrossEncoder（BAAI/bge-reranker-base）接口，默认关闭

### 3. 质量评估

| 维度 | 检查内容 | 阈值 |
|------|---------|------|
| 忠实度 | 来源引用准确性 + 关键词重叠 + 回答长度 | ≥ 0.70 |
| 答案相关性 | 问题关键术语命中率 | ≥ 0.50 |

- 规则检查，无额外模型加载，评估延迟 < 100ms
- 不达标时自动触发重检索重生成（最多 1 次重试）

### 4. 多用户隔离

- **JWT 认证**：bcrypt 密码哈希 + JWT Token（24小时有效期）
- **个人工作区**：文档按 `owner + visibility=private` 隔离
- **团队共享库**：文档按 `visibility=public` 全局去重
- **Agent 实例池**：TTL 1小时自动回收 + LRU 驱逐（上限50）+ 线程安全

### 5. 流式对话

- SSE 流式推送，独立 `ThreadPoolExecutor(max_workers=10)`
- 客户端断开三重检测：`asyncio.wait_for` + `request.is_disconnected()` + `CancelledError`
- `stop_event` 跨线程通知后台停止 LLM 生成
- 来源精确过滤：解析 LLM 回答中的《文件名——片段N》引用，只保留实际使用的片段

## 项目结构

```
enterprise-knowledge-agent/
├── backend/
│   ├── main.py                    # FastAPI 入口（lifespan、CORS、路由注册）
│   ├── api/
│   │   ├── auth.py                # 认证接口（注册/登录/JWT）
│   │   ├── chat.py                # 流式对话 + Agent 实例池
│   │   ├── documents.py           # 文档上传/列表/删除/预览
│   │   └── knowledge_base.py      # 系统状态/清空文档
│   ├── core/
│   │   ├── agent.py               # LangGraph 自纠错工作流
│   │   ├── rag.py                 # RAG 引擎（向量+BM25+RRF）
│   │   ├── memory.py              # 记忆管理（滑动窗口+摘要压缩）
│   │   ├── evaluation.py          # 质量评估（规则检查）
│   │   ├── llm.py                 # LLM 封装
│   │   └── tools.py               # 网络搜索
│   ├── config/settings.py         # 配置（pydantic-settings）
│   ├── models/schemas.py          # Pydantic 数据模型
│   └── utils/
│       ├── auth.py                # JWT 认证工具
│       └── file_utils.py          # 文件工具 + MD5 哈希
├── frontend/
│   └── app.py                     # Streamlit 前端
├── tests/
│   └── test_agent.py              # 单元测试
├── Dockerfile                     # 单容器部署
├── Dockerfile.backend             # 后端容器
├── Dockerfile.frontend            # 前端容器
├── docker-compose.yml             # 多容器编排
├── requirements.txt
├── .env.example
└── start.sh                       # 单容器启动脚本
```

## API 接口

### 认证

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/auth/register` | 用户注册（自动登录） |
| POST | `/api/auth/login` | 用户登录 |

### 对话

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/chat/stream` | 流式对话（SSE） |
| GET | `/api/chat/history` | 获取对话历史 |
| DELETE | `/api/chat/history` | 清空对话历史 |

### 文档

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/documents/upload` | 上传文档（支持 replace） |
| POST | `/api/documents/batch-upload` | 批量上传 |
| GET | `/api/documents/list` | 列出文档 |
| GET | `/api/documents/content/{filename}` | 获取文档片段（支持 visibility 过滤） |
| DELETE | `/api/documents/{filename}` | 删除文档 |

### 知识库

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/knowledge-base/system/status` | 系统状态 |
| GET | `/api/knowledge-base/my-info` | 知识库信息 |
| DELETE | `/api/knowledge-base/my-documents` | 清空当前用户文档 |

## 配置说明

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `MIMO_API_KEY` | - | MIMO API 密钥（必填） |
| `JWT_SECRET_KEY` | - | JWT 签名密钥（必填） |
| `TAVILY_API_KEY` | - | Tavily 搜索 API（可选） |
| `RERANKER_ENABLED` | `false` | 开启重排序（需下载模型约1.3GB） |
| `CHUNK_SIZE` | `500` | 文档分块大小 |
| `CHUNK_OVERLAP` | `50` | 分块重叠大小 |
| `EVAL_ENABLED` | `false` | 启用质量评估 |
| `EVAL_FAITHFULNESS_THRESHOLD` | `0.90` | 忠实度阈值 |
| `EVAL_RELEVANCY_THRESHOLD` | `0.75` | 答案相关性阈值 |
| `EVAL_DEGRADATION_STRATEGY` | `retry` | 降级策略 |

## 部署

### Docker Compose（推荐）

```bash
cp .env.example .env
# 编辑 .env 填入配置

docker-compose up -d
```

### 单容器部署

```bash
docker build -t enterprise-knowledge-agent -f Dockerfile .
docker run -p 8000:8000 -p 8501:8501 --env-file .env enterprise-knowledge-agent
```

## 测试

```bash
pytest tests/ -v
```

## 许可证

MIT License
