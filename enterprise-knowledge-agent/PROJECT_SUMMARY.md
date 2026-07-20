# 项目总结 - 企业智能知识库Agent

## 项目概述

**项目名称**：企业智能知识库Agent

**项目定位**：基于 LangGraph 自纠错工作流 + RAG 混合检索 + 多用户隔离的企业知识库问答系统

**技术栈**：Python 3.10+ / FastAPI / LangGraph / LangChain / MIMO-v2.5 / ChromaDB / BM25 / Streamlit / Docker

---

## 已完成功能

### 核心功能

- **LangGraph 自纠错工作流**
  - 完整闭环：`retrieve → generate → evaluate`
  - 评估不达标时自动触发 `enhance_query → retrieve → generate → evaluate`
  - 支持 `general / internal / external / external_disabled / internal_first` 五种路由
  - 关键词快速匹配 + LLM 语义分类 + 规则降级三级路由策略
  - HyDE 查询改写：短/模糊查询自动生成假设性文档，提升检索召回率

- **RAG 混合检索引擎**
  - 向量检索：基于 fastembed (BAAI/bge-small-en-v1.5) 生成语义向量
  - BM25 关键词检索：基于 bm25s + jieba 中文分词，支持异步重建
  - RRF 融合排序：Reciprocal Rank Fusion，两路结果按排名融合
  - BM25 可见性过滤：独立实现后过滤层，确保多用户数据隔离
  - chunk_index 元数据：记录片段在文件内的位置，确保引用编号一致
  - 可选 CrossEncoder 重排序（默认关闭）

- **质量评估（规则检查，无额外模型）**
  - 忠实度：来源引用准确性 + 关键词重叠 + 回答长度合理性
  - 答案相关性：问题关键术语命中率
  - 评估延迟 < 100ms，不达标自动触发重检索重生成

- **多用户认证与隔离**
  - JWT 认证：bcrypt 密码哈希 + PyJWT Token（24小时有效期）
  - 用户数据并发写保护（threading.Lock）
  - 个人工作区：`owner + visibility=private` 隔离
  - 团队共享库：`visibility=public` 全局去重
  - Agent 实例池：TTL 自动回收（1小时）+ LRU 驱逐（上限50）+ 线程安全

- **流式对话**
  - SSE 流式推送 LLM 回答
  - 独立 ThreadPoolExecutor（10 workers），不占用 FastAPI 默认线程池
  - 客户端断开三重检测 + stop_event 跨线程通知
  - 120 秒超时保护
  - 来源精确过滤：解析《文件名——片段N》引用，只保留实际使用的片段

- **记忆管理**
  - 滑动窗口：消息超阈值时自动压缩旧消息为 LLM 摘要
  - 压缩后截断历史：避免重复压缩同一批旧消息
  - 并发压缩保护：`_compress_lock` 双重检查锁
  - 知识库感知：空知识库时自动过滤历史中的知识库问答

- **文档管理**
  - 支持 PDF、Word、Excel、TXT、CSV 等多种格式
  - 两阶段去重：MD5 哈希精确匹配 + 文件名匹配
  - 文档替换：原上传者可覆盖更新（`is_document_owner` 权限检查）
  - 文档预览：按可见性精确过滤，个人库和公共库同名文件互不干扰
  - chunk_index 元数据：确保前端预览与 LLM 引用编号一致

- **网络搜索**
  - Tavily API 集成，智能时效筛选（根据关键词自动选择 time_range）
  - search_depth="advanced" 深度搜索
  - 15 秒超时保护

### 工程化功能

- **后端 API**
  - FastAPI 构建的 RESTful API
  - 自动生成 API 文档（Swagger/ReDoc）
  - JWT 认证中间件（兼容 Swagger UI Authorize）
  - CORS 配置
  - 全局异常处理器

- **安全性**
  - JWT_SECRET_KEY 启动时安全检查
  - 用户文件写并发锁
  - 路径遍历防护（_sanitize_user_id）
  - MD5 哈希去重（compute_file_hash）

- **部署支持**
  - Docker 容器化（单容器 / 多容器两种方案）
  - Docker Compose 编排（前后端分离 + 健康检查）

- **日志规范**
  - 统一使用 logging.getLogger(__name__)
  - 结构化日志格式
  - 无 print 残留

---

## 技术架构

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
│          LangGraph 自纠错工作流                              │
│  router → retrieve → generate → evaluate                    │
│                       ↑            │                         │
│                       └── retry ───┘                        │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   RAG引擎层     │ │   记忆层        │ │   评估层        │
│ 向量+BM25检索   │ │ 滑动窗口/摘要   │ │ 来源引用验证    │
│ RRF融合/HyDE    │ │ 偏好/知识注入   │ │ 关键词重叠检查  │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

### 技术栈详情

| 层级 | 技术选择 | 说明 |
|------|----------|------|
| **前端** | Streamlit 1.45+ | 快速构建交互界面，@st.fragment 独立渲染 |
| **后端** | FastAPI 0.115+ | 异步高性能 Web 框架 |
| **工作流** | LangGraph 0.4+ | 状态图编排自纠错工作流 |
| **AI 框架** | LangChain 0.3+ | LLM 调用、文档处理 |
| **LLM** | MIMO-v2.5 | 小米大模型 |
| **向量数据库** | ChromaDB 1.0+ | 轻量级向量存储 |
| **Embedding** | fastembed (BAAI/bge-small-en-v1.5) | 本地 Embedding 模型 |
| **BM25** | bm25s + jieba | 中文分词 + BM25 稀疏检索 |
| **检索融合** | RRF | 向量 + BM25 排名融合 |
| **认证** | bcrypt + PyJWT | 密码哈希 + JWT Token |
| **文档处理** | pypdf / python-docx / pandas | 多格式文档解析 |
| **网络搜索** | Tavily | 外部实时信息检索 |
| **部署** | Docker + Docker Compose | 容器化部署 |

---

## 项目数据

### 功能指标

| 指标 | 数值 | 说明 |
|------|------|------|
| 支持文档格式 | 5种 | PDF、Word、Excel、TXT、CSV |
| 检索策略 | 混合检索 | 向量 + BM25 + RRF 融合 |
| 意图路由 | 5种 | general / internal / external / external_disabled / internal_first |
| 质量评估 | 2维度 | 忠实度 + 答案相关性（规则检查） |
| 多用户支持 | JWT认证 | bcrypt + PyJWT + 空间隔离 |
| 流式输出 | SSE | 独立线程池 + 客户端断开检测 |

### 代码结构

| 模块 | 文件数 | 说明 |
|------|--------|------|
| 后端 API | 4 | auth / chat / documents / knowledge_base |
| 核心逻辑 | 6 | agent / rag / memory / evaluation / llm / tools |
| 工具函数 | 2 | auth / file_utils |
| 前端 | 1 | app.py |
| 测试 | 1 | test_agent.py |
| 部署 | 4 | Dockerfile × 2 + docker-compose + start.sh |

---

## 求职亮点

### 技术深度

1. **LangGraph 自纠错工作流**：retrieve → generate → evaluate 完整闭环，评估不达标自动重检索重生成
2. **RAG 全流程**：文档解析 → 分块 → 向量化 → 混合检索 → RRF 融合 → LLM 生成
3. **质量评估**：忠实度（来源引用验证）+ 答案相关性（术语匹配），规则检查无额外模型
4. **HyDE 查询改写**：短/模糊查询自动生成假设性文档，提升检索召回率
5. **BM25 可见性过滤**：独立实现后过滤层，确保多用户数据隔离

### 工程能力

1. **多用户架构**：JWT 认证 + Agent 实例池 + TTL 回收 + 长期记忆持久化
2. **流式架构**：SSE 流式推送 + 独立线程池 + 客户端断开检测 + 来源精确过滤
3. **前后端分离**：FastAPI RESTful API + Streamlit 交互界面
4. **容器化部署**：Docker 多容器编排 + 健康检查
5. **安全性**：MD5 哈希去重、路径遍历防护、JWT 安全检查、日志规范化

---

## 后续优化方向

### 功能扩展
1. 多模态支持：图片、音频等多媒体文档
2. 权限管理：管理员角色，文档审核流程
3. 知识图谱：构建企业知识图谱

### 技术优化
1. 检索优化：引入 CrossEncoder 重排序（已预留接口）
2. 缓存机制：热门查询结果缓存
3. 监控告警：Prometheus 指标
