# 快速启动指南

## 前置条件

- Python 3.10+
- MIMO API Key（小米大模型，必填）
- JWT Secret Key（必填，用于用户认证）
- Tavily API Key（可选，用于网络搜索）

## 快速启动（3步）

### 步骤 1：配置环境变量

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

### 步骤 2：启动后端

```bash
cd enterprise-knowledge-agent
python -m backend.main
```

后端服务将在 http://localhost:8000 启动。首次启动会后台预加载 embedding 模型（约 1-2 秒）。

### 步骤 3：启动前端

```bash
streamlit run frontend/app.py
```

前端界面将在 http://localhost:8501 启动。

## 访问应用

| 服务 | 地址 | 说明 |
|------|------|------|
| 前端界面 | http://localhost:8501 | 用户交互界面 |
| API文档 | http://localhost:8000/docs | Swagger API文档 |
| 健康检查 | http://localhost:8000/health | 系统状态检查 |

## 使用流程

### 1. 注册账号

首次使用点击「注册」标签，输入用户ID（2-20字符）和密码（至少4字符），注册后自动登录。

### 2. 上传文档

1. 在左侧边栏选择文件（支持 PDF、Word、TXT、Excel、CSV）
2. 选择存储位置：个人工作区或团队共享库
3. 可选：勾选「替换已有同名文档」
4. 点击「上传到知识库」

### 3. 开始对话

在底部输入框输入问题，系统会：
1. 自动判断问题类型（内部知识/外部搜索/通用回复）
2. 从知识库检索相关内容或搜索互联网
3. 基于检索结果生成回答（流式输出）
4. 回答末尾标注参考来源，点击可预览原文片段

### 4. 文档去重与替换

- 上传同名同内容的文件：自动跳过（MD5 哈希精确匹配）
- 上传同名不同内容的文件：提示是否替换（需勾选「替换已有同名文档」）
- 只有原上传者才能替换自己的文档

## 配置说明

### 环境变量

| 变量名 | 说明 | 必填 | 默认值 |
|--------|------|------|--------|
| `MIMO_API_KEY` | MIMO API 密钥 | 是 | - |
| `MIMO_BASE_URL` | MIMO API 地址 | 否 | `https://token-plan-cn.xiaomimimo.com/v1` |
| `MIMO_MODEL` | 模型名称 | 否 | `mimo-v2.5` |
| `JWT_SECRET_KEY` | JWT 签名密钥 | 是 | - |
| `TAVILY_API_KEY` | Tavily 搜索 API | 否 | - |
| `RERANKER_ENABLED` | 开启重排序 | 否 | `false` |
| `EVAL_ENABLED` | 启用质量评估 | 否 | `false` |
| `EVAL_FAITHFULNESS_THRESHOLD` | 忠实度阈值 | 否 | `0.70` |
| `EVAL_RELEVANCY_THRESHOLD` | 答案相关性阈值 | 否 | `0.50` |
| `EVAL_DEGRADATION_STRATEGY` | 降级策略 | 否 | `retry` |
| `AGENT_MAX_RETRIES` | 评估不达标时最大重试次数 | 否 | `1` |

### 内置配置

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `CHROMA_PERSIST_DIR` | ChromaDB 存储目录 | `backend/data/chroma_db` |
| `CHROMA_COLLECTION_NAME` | ChromaDB 集合名 | `enterprise_knowledge` |
| `CHUNK_SIZE` | 文档分块大小 | `500` |
| `CHUNK_OVERLAP` | 分块重叠大小 | `50` |
| `MAX_FILE_SIZE` | 最大上传文件大小 | `10MB` |

### Agent 实例池配置

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `_AGENT_TTL` | 实例最大空闲时间 | `3600` 秒 |
| `_MAX_AGENTS` | 最大在线实例数 | `50` |
| `_stream_executor` | 流式线程池大小 | `10` |

## 常见问题

### Q: 启动后前端显示「后端服务连接超时」

确认后端已启动，检查 http://localhost:8000/health 是否返回 `ready: true`。

### Q: 文档上传后检索不到

首次上传时 embedding 模型需要下载（约 500MB），请等待后台预加载完成。

### Q: 删除知识库后 LLM 仍引用旧内容

系统会自动清理对话摘要。如仍有残留，点击侧边栏「清空对话」重置历史。

### Q: 网络搜索不可用

确认 `.env` 中配置了 `TAVILY_API_KEY`，并在侧边栏开启「启用网络搜索」。

### Q: ChromaDB 初始化失败

```bash
rm -rf backend/data/chroma_db backend/data/bm25_index
python -m backend.main
```

## 运行测试

```bash
pytest tests/ -v
```

## 提示

- 首次启动时系统会自动下载 Embedding 模型（约 500MB），后续启动使用缓存
- 文档上传后自动进行向量化和 BM25 索引构建
- 对话历史自动保存，刷新页面不会丢失
- JWT Token 有效期 24 小时，过期后需重新登录
- Agent 实例超过 1 小时未访问会自动回收，长期记忆会持久化保存
