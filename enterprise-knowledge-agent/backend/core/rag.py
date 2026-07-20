"""RAG引擎模块 - 实现文档处理、向量存储、混合检索（向量 + BM25）"""

import logging
import os
import threading
from typing import List, Dict, Any, Optional
from pathlib import Path

from backend.config.settings import settings

logger = logging.getLogger(__name__)

# 模块级依赖检查（避免每次解析时重复 import）
_HAS_DOCX = False
try:
    import docx  # noqa: F401
    _HAS_DOCX = True
except ImportError:
    pass

_HAS_DOCX2TXT = False
try:
    import docx2txt  # noqa: F401
    _HAS_DOCX2TXT = True
except ImportError:
    pass


def _create_text_splitter(chunk_size: int = 500, chunk_overlap: int = 50):
    """
    创建文本分割器（优先使用 RecursiveCharacterTextSplitter，降级到 SimpleTextSplitter）

    RecursiveCharacterTextSplitter 按段落/句子边界切分，保持语义完整性。
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        )
    except ImportError:
        logger.warning("[RAG] langchain_text_splitters 不可用，降级到简单分割器")

    # 降级方案：简单按字符数切分
    from langchain_core.documents import Document

    class _SimpleSplitter:
        def __init__(self, cs, co):
            self.chunk_size = cs
            self.chunk_overlap = co

        def split_text(self, text):
            chunks = []
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunks.append(text[start:end])
                start = end - self.chunk_overlap
            return chunks

        def split_documents(self, documents):
            result = []
            for doc in documents:
                for chunk in self.split_text(doc.page_content):
                    result.append(Document(page_content=chunk, metadata=doc.metadata.copy()))
            return result

    return _SimpleSplitter(chunk_size, chunk_overlap)


class _EmbeddingSingleton:
    """Embedding模型单例 - 避免每个RAGEngine实例都加载一次模型"""
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            from fastembed import TextEmbedding
            from langchain_core.embeddings import Embeddings

            class FastEmbedEmbeddings(Embeddings):
                def __init__(self):
                    self.model = TextEmbedding('BAAI/bge-small-en-v1.5')

                def embed_documents(self, texts):
                    return [[float(x) for x in vec] for vec in self.model.embed(texts)]

                def embed_query(self, text):
                    result = list(self.model.embed([text]))
                    return [float(x) for x in result[0]]

            cls._instance = FastEmbedEmbeddings()
        return cls._instance


class RAGEngine:
    """RAG引擎 - 管理文档处理、存储和混合检索（向量检索 + BM25稀疏检索）"""

    def __init__(self):
        """初始化RAG引擎"""
        # 向量检索相关（embedding 模型全局只加载一次）
        self._embeddings = _EmbeddingSingleton.get()
        self._vector_store = None
        self._text_splitter = None

        # BM25稀疏检索相关
        self._bm25_retriever = None
        self._bm25_corpus: List[Dict] = []  # [{"id": str, "content": str}, ...]
        self._bm25_dirty = False  # 标记是否有未同步的变更
        self._bm25_rebuilding = False  # 标记是否有重建线程在运行
        self._bm25_lock = threading.Lock()  # 保护异步重建时的并发访问
        self._bm25_dir = Path(settings.CHROMA_PERSIST_DIR).parent / "bm25_index"
        os.makedirs(self._bm25_dir, exist_ok=True)

        # 重排序器（延迟加载，首次需下载模型约 1.3GB）
        # 通过 settings.RERANKER_ENABLED 或 .env 中 RERANKER_ENABLED=true 开启
        self._reranker = None
        self._reranker_enabled = settings.RERANKER_ENABLED

    @property
    def embeddings(self):
        """返回Embedding模型"""
        return self._embeddings

    @property
    def vector_store(self):
        """延迟加载向量存储"""
        if self._vector_store is None:
            from langchain_community.vectorstores import Chroma
            os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
            self._vector_store = Chroma(
                collection_name=settings.CHROMA_COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=settings.CHROMA_PERSIST_DIR
            )
        return self._vector_store

    @property
    def text_splitter(self):
        """返回文本分割器（优先 RecursiveCharacterTextSplitter）"""
        if self._text_splitter is None:
            self._text_splitter = _create_text_splitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP
            )
        return self._text_splitter

    # ==================== BM25 索引管理 ====================

    def _get_bm25_path(self) -> str:
        """获取BM25索引文件路径"""
        return str(self._bm25_dir / "bm25_index.bm25")

    def _get_bm25_corpus_path(self) -> str:
        """获取BM25语料库持久化文件路径"""
        return str(self._bm25_dir / "bm25_corpus.json")

    def _save_bm25_corpus(self):
        """将BM25语料库持久化到磁盘"""
        import json
        path = self._get_bm25_corpus_path()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._bm25_corpus, f, ensure_ascii=False)
        except Exception as e:
            logger.error(f"[BM25] 语料库保存失败: {e}")

    def _load_bm25_corpus(self) -> list:
        """从磁盘加载BM25语料库"""
        import json
        path = self._get_bm25_corpus_path()
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"[BM25] 语料库加载失败: {e}")
        return []

    def _init_bm25(self):
        """
        初始化BM25索引：从磁盘加载语料库和索引
        """
        # 1. 从磁盘加载语料库
        if not self._bm25_corpus:
            self._bm25_corpus = self._load_bm25_corpus()

        index_path = self._get_bm25_path()

        if os.path.exists(index_path) and self._bm25_corpus:
            # 2a. 有索引文件 + 有语料 → 加载索引
            import bm25s
            logger.debug(f"[BM25] 加载已有索引: {index_path}")
            self._bm25_retriever = bm25s.BM25.load(index_path, load_corpus=True)
        elif self._bm25_corpus:
            # 2b. 有语料但没索引 → 重建索引
            logger.debug(f"[BM25] 从 {len(self._bm25_corpus)} 个文档重建索引")
            self._build_bm25_index()
        else:
            logger.debug("[BM25] 无语料，跳过初始化")

    def _build_bm25_index(self):
        """
        从 _bm25_corpus 构建BM25索引并持久化到磁盘
        """
        import bm25s
        import jieba

        index_path = self._get_bm25_path()

        # 语料为空时：清空内存索引并保存空状态到磁盘
        if not self._bm25_corpus:
            self._bm25_retriever = None
            # 删除磁盘上的索引文件
            import shutil
            if Path(index_path).exists():
                shutil.rmtree(index_path, ignore_errors=True)
            self._save_bm25_corpus()
            logger.debug(f"[BM25] 语料为空，已清空索引: {index_path}")
            return

        # 对语料进行中文分词
        corpus_tokens = [jieba.lcut(doc["content"]) for doc in self._bm25_corpus]

        # 创建BM25索引
        self._bm25_retriever = bm25s.BM25(k1=1.5, b=0.75, corpus=self._bm25_corpus)
        self._bm25_retriever.index(corpus_tokens)

        # 持久化到磁盘（索引 + 语料库）
        self._bm25_retriever.save(index_path)
        self._save_bm25_corpus()
        logger.debug(f"[BM25] 索引已保存至: {index_path}（{len(self._bm25_corpus)} 条语料）")

    def _add_to_bm25(self, doc_id: str, content: str, file_path: str = "", metadata: Optional[Dict] = None):
        """
        将文档添加到BM25语料库（不立即重建索引）

        Args:
            doc_id: 文档ID
            content: 文档内容
            file_path: 来源文件路径（用于删除时匹配）
            metadata: 文档元数据（original_filename, owner, visibility 等）
        """
        entry = {"id": doc_id, "content": content, "file_path": file_path}
        if metadata:
            entry["metadata"] = metadata
        self._bm25_corpus.append(entry)

    def _rebuild_bm25_if_needed(self):
        """
        标记 BM25 索引需要重建，并在后台线程异步执行

        上传/删除文档时调用，不阻塞 API 响应。
        搜索时通过 _ensure_bm25_ready 等待重建完成。
        如果已有重建线程在运行，仅标记 dirty 不创建新线程。
        """
        self._bm25_dirty = True
        # 如果已有重建线程在运行，不重复创建
        if self._bm25_rebuilding:
            return
        self._bm25_rebuilding = True
        threading.Thread(target=self._async_rebuild_bm25, daemon=True).start()

    def _async_rebuild_bm25(self):
        """后台线程执行 BM25 索引重建"""
        try:
            with self._bm25_lock:
                if self._bm25_dirty:
                    self._build_bm25_index()
                    self._bm25_dirty = False
        finally:
            self._bm25_rebuilding = False

    def _ensure_bm25_ready(self):
        """确保 BM25 索引可用（如果 dirty 则等待重建完成）"""
        if self._bm25_dirty:
            # 等待后台重建完成
            with self._bm25_lock:
                pass  # 获取锁说明重建已完成
        elif self._bm25_retriever is None:
            self._init_bm25()

    # ==================== 文档处理 ====================

    def add_document(self, file_path: str, metadata: Optional[Dict] = None, content_hash: str = "") -> int:
        """
        添加文档到知识库（向量库 + BM25语料库）

        Args:
            file_path: 文件路径
            metadata: 额外的元数据
            content_hash: 文件内容 MD5 哈希（可选，用于内容级去重）

        Returns:
            添加的文档块数量
        """
        import uuid

        # 解析文档
        logger.info(f"[RAG] 开始解析文档: {file_path}")
        documents = self._parse_document(file_path)

        if not documents:
            logger.warning(f"[RAG] 文档解析结果为空: {file_path}")
            return 0

        logger.info(f"[RAG] 文档解析成功: {file_path}, 原始内容长度={sum(len(d.page_content) for d in documents)}")

        # 添加元数据（包含原始文件名，用于去重和删除）
        original_filename = Path(file_path).name
        # 去掉时间戳前缀：20260624_153805_xxx.pdf -> xxx.pdf
        if len(original_filename) > 15 and original_filename[8] == "_" and original_filename[15] == "_":
            original_filename = original_filename[16:]

        for doc in documents:
            doc.metadata["file_path"] = file_path
            doc.metadata["original_filename"] = original_filename
            if content_hash:
                doc.metadata["content_hash"] = content_hash
            if metadata:
                doc.metadata.update(metadata)
            # 确保 visibility 和 owner 有默认值
            if "visibility" not in doc.metadata:
                doc.metadata["visibility"] = "private"
            if "owner" not in doc.metadata:
                doc.metadata["owner"] = ""

        # 分割文档
        split_docs = self.text_splitter.split_documents(documents)
        logger.info(f"[RAG] 文档分割完成: {len(split_docs)} 个片段")

        # 记录每个片段在文件内的位置（用于前端精确溯源）
        for chunk_idx, doc in enumerate(split_docs, 1):
            doc.metadata["chunk_index"] = chunk_idx

        # 1. 添加到向量存储
        logger.info(f"[RAG] 添加到向量库: {original_filename}")
        self.vector_store.add_documents(split_docs)
        logger.info(f"[RAG] 向量库添加完成: {len(split_docs)} 个片段, visibility={documents[0].metadata.get('visibility')}")

        # 2. 添加到BM25语料库（保存完整元数据用于来源展示）
        logger.info(f"[RAG] 添加到BM25语料库: {len(split_docs)} 条")
        for doc in split_docs:
            doc_id = str(uuid.uuid4())
            self._add_to_bm25(
                doc_id, doc.page_content, file_path=file_path,
                metadata={k: v for k, v in doc.metadata.items() if k in ("original_filename", "owner", "visibility", "source", "content_hash")}
            )

        # 3. 重建BM25索引
        logger.info(f"[RAG] 触发BM25索引重建")
        self._rebuild_bm25_if_needed()

        return len(split_docs)

    def delete_document(self, file_path: str, original_filename: str = "", owner: str = "", visibility: str = "") -> int:
        """
        从知识库中删除指定文档（向量库 + BM25语料库）

        Args:
            file_path: 文件路径
            original_filename: 原始文件名（用于匹配，更可靠）
            owner: 文档所有者（防止误删其他用户同名文件）
            visibility: 文档可见性（"private"/"public"/空）

        Returns:
            删除的文档块数量
        """
        deleted_count = 0

        # 1. 从向量库中删除（按文件名 + owner + visibility 过滤）
        try:
            conditions = []
            if original_filename:
                conditions.append({"original_filename": original_filename})
            else:
                conditions.append({"file_path": file_path})
            if owner:
                conditions.append({"owner": owner})
            if visibility:
                conditions.append({"visibility": visibility})

            if len(conditions) == 0:
                where_clause = None
            elif len(conditions) == 1:
                where_clause = conditions[0]
            else:
                where_clause = {"$and": conditions}

            result = self.vector_store.get(where=where_clause, include=[])
            ids_to_delete = result.get("ids", [])
            if ids_to_delete:
                self.vector_store.delete(ids=ids_to_delete)
                deleted_count = len(ids_to_delete)
                logger.debug(f"[RAG] 从向量库删除 {deleted_count} 个片段: {original_filename or file_path}")
        except Exception as e:
            logger.error(f"[RAG] 向量库删除失败: {e}")

        # 2. 从 BM25 语料库中删除（按 original_filename + owner + visibility 精确匹配）
        original_count = len(self._bm25_corpus)

        def _should_keep_bm25_doc(doc: dict) -> bool:
            """判断 BM25 文档是否应该保留（返回 True=保留，False=删除）"""
            meta = doc.get("metadata", {})
            doc_filename = meta.get("original_filename", "")
            doc_owner = meta.get("owner", "")
            doc_visibility = meta.get("visibility", "")

            # 按 original_filename 匹配
            if original_filename and doc_filename != original_filename:
                return True  # 文件名不匹配，保留
            elif not original_filename and doc.get("file_path") != file_path:
                return True  # file_path 不匹配，保留

            # 按 owner 过滤
            if owner and doc_owner != owner:
                return True  # owner 不匹配，保留

            # 按 visibility 过滤
            if visibility and doc_visibility != visibility:
                return True  # visibility 不匹配，保留

            return False  # 所有条件都匹配，删除
        # 创建了一个全新的语料库列表，只包含那些没有被删的文档
        self._bm25_corpus = [doc for doc in self._bm25_corpus if _should_keep_bm25_doc(doc)]
        bm25_removed = original_count - len(self._bm25_corpus)
        if bm25_removed > 0:
            self._build_bm25_index()
            logger.debug(f"[RAG] 从BM25删除 {bm25_removed} 个片段")

        return deleted_count

    def has_document(self, original_filename: str, owner: str = "", visibility: str = "private",
                     content_hash: str = "") -> bool:
        """
        检查知识库中是否已存在文档（支持内容哈希精确去重）

        去重规则：
        - 私有文档：按 original_filename + owner + visibility='private' 去重
        - 公共文档：按 original_filename + visibility='public' 全局去重（忽略 owner）
        - 若提供 content_hash：额外匹配内容哈希，区分同名但内容不同的文件

        Args:
            original_filename: 原始文件名
            owner: 文档所有者（仅私有文档使用，公共文档忽略）
            visibility: 文档可见性
            content_hash: 文件内容 MD5（可选），用于精确去重

        Returns:
            是否存在匹配文档
        """
        try:
            # 防御性检查：私有文档必须提供 owner
            if visibility == "private" and not owner:
                logger.warning(f"[RAG] has_document: 私有文档缺少 owner，跳过查询: {original_filename}")
                return False

            conditions = [
                {"original_filename": original_filename},
                {"visibility": visibility}
            ]

            # 私有文档按 owner 隔离，公共文档忽略 owner
            if visibility == "private":
                conditions.append({"owner": owner})

            # 内容哈希精确去重（可选）
            if content_hash:
                conditions.append({"content_hash": content_hash})

            where = {"$and": conditions} if len(conditions) > 1 else conditions[0]
            result = self.vector_store.get(where=where, include=[], limit=1)
            ids = result.get("ids", [])
            return len(ids) > 0
        except Exception as e:
            logger.error(f"[RAG] has_document 查询失败: {e}")
            return False

    def is_document_owner(self, original_filename: str, user_id: str, visibility: str = "") -> bool:
        """
        检查指定用户是否为某文档的所有者（用于替换权限判断）

        Args:
            original_filename: 原始文件名
            user_id: 用户ID
            visibility: 文档可见性（可选，为空则不区分可见性）

        Returns:
            是否为文档所有者
        """
        try:
            conditions = [
                {"original_filename": original_filename},
                {"owner": user_id}
            ]
            if visibility:
                conditions.append({"visibility": visibility})
            where = {"$and": conditions} if len(conditions) > 1 else conditions[0]

            result = self.vector_store.get(where=where, include=[], limit=1)
            ids = result.get("ids", [])
            return len(ids) > 0
        except Exception as e:
            logger.error(f"[RAG] is_document_owner 查询失败: {e}")
            return False

    def list_documents(self, owner: str = "", visibility: str = "") -> list:
        """
        列出知识库中的文档（按原始文件名去重）

        Args:
            owner: 过滤指定用户的文档（空字符串不过滤）
            visibility: 过滤可见性 "private"/"public"（空字符串不过滤）

        Returns:
            文档信息列表 [{"original_filename": str, "chunk_count": int, "file_path": str, "visibility": str, "owner": str}]
        """
        try:
            # 构建过滤条件
            where = None
            if owner and visibility:
                where = {"$and": [{"owner": owner}, {"visibility": visibility}]}
            elif owner:
                where = {"owner": owner}
            elif visibility:
                where = {"visibility": visibility}

            if where:
                result = self.vector_store.get(where=where, include=["metadatas"])
            else:
                result = self.vector_store.get(include=["metadatas"])

            metadatas = result.get("metadatas", [])

            # 按 original_filename 分组统计
            doc_map = {}
            for meta in metadatas:
                fname = meta.get("original_filename", "未知文件")
                fpath = meta.get("file_path", "")
                vis = meta.get("visibility", "private")
                own = meta.get("owner", "")
                key = f"{fname}|{vis}|{own}"
                if key not in doc_map:
                    doc_map[key] = {
                        "original_filename": fname,
                        "file_path": fpath,
                        "visibility": vis,
                        "owner": own,
                        "chunk_count": 0
                    }
                doc_map[key]["chunk_count"] += 1

            return list(doc_map.values())
        except Exception as e:
            logger.error(f"[RAG] 列出文档失败: {e}")
            return []

    def _parse_document(self, file_path: str) -> List:
        """解析文档"""
        file_ext = Path(file_path).suffix.lower()

        try:
            if file_ext == ".pdf":
                return self._parse_pdf(file_path)
            elif file_ext in [".docx", ".doc"]:
                return self._parse_docx(file_path)
            elif file_ext == ".txt":
                return self._parse_txt(file_path)
            elif file_ext in [".xlsx", ".xls"]:
                return self._parse_excel(file_path)
            elif file_ext == ".csv":
                return self._parse_csv(file_path)
            else:
                logger.warning(f"不支持的文件格式: {file_ext}")
                return []
        except Exception as e:
            logger.error(f"解析文档失败: {e}")
            return []

    def _parse_pdf(self, file_path: str) -> List:
        """解析PDF文件（检测扫描件并告警）"""
        from langchain_core.documents import Document
        import pypdf

        try:
            reader = pypdf.PdfReader(file_path)
            content = ""
            for page in reader.pages:
                page_text = page.extract_text() or ""
                content += page_text + "\n"

            # 检测扫描件：提取内容过少
            clean_content = content.strip()
            if len(clean_content) < 50:
                page_count = len(reader.pages)
                logger.warning(f"[RAG] PDF 可能为扫描件（{page_count}页，仅提取到 {len(clean_content)} 字符）: {file_path}")

            return [Document(
                page_content=content,
                metadata={"source": file_path, "type": "pdf"}
            )]
        except Exception as e:
            logger.error(f"解析PDF失败: {e}")
            return []

    def _parse_docx(self, file_path: str) -> List:
        """解析Word文件（支持 .docx 和 .doc，使用模块级依赖缓存）"""
        from langchain_core.documents import Document

        # 优先使用 python-docx（更稳定）
        if _HAS_DOCX:
            try:
                from docx import Document as DocxDocument
                doc = DocxDocument(file_path)
                content = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
                if content:
                    return [Document(
                        page_content=content,
                        metadata={"source": file_path, "type": "docx"}
                    )]
                logger.debug(f"[RAG] python-docx 解析为空，尝试 docx2txt: {file_path}")
            except Exception as e:
                logger.warning(f"[RAG] python-docx 解析异常，尝试 docx2txt: {e}")

        # 降级方案：docx2txt
        if _HAS_DOCX2TXT:
            try:
                import docx2txt
                content = docx2txt.process(file_path)
                if content and content.strip():
                    return [Document(
                        page_content=content,
                        metadata={"source": file_path, "type": "docx"}
                    )]
                logger.warning(f"[RAG] docx2txt 解析也为空: {file_path}")
                return []
            except Exception as e:
                logger.error(f"解析Word失败: {e}")
                return []

        # 两个库都不可用
        logger.error("解析Word失败: 需要安装 python-docx 或 docx2txt")
        return []

    def _parse_txt(self, file_path: str) -> List:
        """解析文本文件（支持多种编码）"""
        from langchain_core.documents import Document

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    content = f.read()
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()

        return [Document(
            page_content=content,
            metadata={"source": file_path, "type": "txt"}
        )]

    def _parse_excel(self, file_path: str) -> List:
        """解析Excel文件"""
        import pandas as pd
        from langchain_core.documents import Document

        try:
            df = pd.read_excel(file_path)
            content = df.to_string(index=False)
            return [Document(
                page_content=content,
                metadata={"source": file_path, "type": "excel"}
            )]
        except Exception as e:
            logger.error(f"解析Excel失败: {e}")
            return []

    def _parse_csv(self, file_path: str) -> List:
        """解析CSV文件"""
        import pandas as pd
        from langchain_core.documents import Document

        try:
            df = pd.read_csv(file_path)
            content = df.to_string(index=False)
            return [Document(
                page_content=content,
                metadata={"source": file_path, "type": "csv"}
            )]
        except Exception as e:
            logger.error(f"解析CSV失败: {e}")
            return []

    # ==================== 检索方法 ====================

    def _vector_search(self, query: str, top_k: int = 10, where: Optional[Dict] = None) -> List:
        """
        向量检索（语义相似性）

        Args:
            query: 查询文本
            top_k: 返回结果数量
            where: ChromaDB 元数据过滤条件

        Returns:
            相关文档列表
        """
        try:
            if where:
                results = self.vector_store.similarity_search_with_relevance_scores(
                    query, k=top_k, filter=where
                )
            else:
                results = self.vector_store.similarity_search_with_relevance_scores(query, k=top_k)
            logger.debug(f"[RAG] 向量检索: where={where}, 返回 {len(results)} 条结果")
            return [(doc, score) for doc, score in results]
        except Exception as e:
            logger.error(f"[RAG] 向量检索失败: {e}")
            return []

    def _bm25_search(self, query: str, top_k: int = 10, where: Optional[Dict] = None) -> List:
        """
        BM25稀疏检索（关键词匹配，支持元数据过滤）

        Args:
            query: 查询文本
            top_k: 返回结果数量
            where: 元数据过滤条件（如 {"visibility": "public"}）

        Returns:
            (doc_dict, score) 格式的结果列表
        """
        import jieba

        self._ensure_bm25_ready()

        if self._bm25_retriever is None or not self._bm25_corpus:
            return []

        try:
            # 限制 k 不超过语料库大小
            actual_k = min(top_k, len(self._bm25_corpus))
            query_tokens = [jieba.lcut(query)]
            results, scores = self._bm25_retriever.retrieve(query_tokens, k=actual_k, show_progress=False)
            found = [(results[0, i], scores[0, i]) for i in range(results.shape[1])]

            # 按元数据过滤 BM25 结果
            if where:
                found = [(doc, score) for doc, score in found if self._match_bm25_metadata(doc, where)]

            return found
        except Exception as e:
            logger.error(f"BM25检索失败: {e}")
            return []

    def _match_bm25_metadata(self, doc: dict, where: Dict) -> bool:
        """
        检查 BM25 文档是否匹配元数据过滤条件

        Args:
            doc: BM25 文档字典
            where: 过滤条件（支持简单键值对和 $and 组合）

        Returns:
            是否匹配
        """
        meta = doc.get("metadata", {})

        if "$and" in where:
            return all(self._match_single(meta, cond) for cond in where["$and"])
        return self._match_single(meta, where)

    def _match_single(self, meta: dict, condition: Dict) -> bool:
        """检查单个过滤条件是否匹配"""
        for key, value in condition.items():
            if meta.get(key) != value:
                return False
        return True

    def hybrid_search(self, query: str, top_k: int = 10, rrf_k: int = 60,
                      where: Optional[Dict] = None) -> List:
        """
        混合检索：向量检索 + BM25检索，使用 Reciprocal Rank Fusion (RRF) 融合

        Args:
            query: 查询文本
            top_k: 最终返回的结果数量
            rrf_k: RRF常数，控制排名融合的平滑度
            where: 元数据过滤条件（同时应用于向量检索和BM25检索）

        Returns:
            融合排序后的文档列表 (langchain Document 对象)
        """
        from langchain_core.documents import Document

        # 两路检索，多取一些候选以保证融合质量
        vector_results = self._vector_search(query, top_k=top_k * 2, where=where)
        bm25_results = self._bm25_search(query, top_k=top_k * 2, where=where)

        # 用 RRF 融合
        # key: doc内容的前100字符（用于去重），value: {"doc": Document, "score": float}
        score_map: Dict[str, Dict] = {}

        # 处理向量检索结果
        for rank, (doc, _) in enumerate(vector_results):
            key = doc.page_content[:100]
            if key not in score_map:
                score_map[key] = {"doc": doc, "score": 0.0}
            score_map[key]["score"] += 1.0 / (rrf_k + rank + 1)

        # 处理BM25检索结果
        for rank, (doc_dict, _) in enumerate(bm25_results):
            content = doc_dict.get("content", "")
            key = content[:100]
            if key not in score_map:
                # BM25返回的是dict，恢复保存的元数据
                bm25_meta = doc_dict.get("metadata", {})
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": bm25_meta.get("source", "bm25"),
                        "original_filename": bm25_meta.get("original_filename", ""),
                        "owner": bm25_meta.get("owner", ""),
                        "visibility": bm25_meta.get("visibility", ""),
                        "id": doc_dict.get("id", "")
                    }
                )
                score_map[key] = {"doc": doc, "score": 0.0}
            score_map[key]["score"] += 1.0 / (rrf_k + rank + 1)

        # 按融合分数降序排列
        sorted_results = sorted(score_map.values(), key=lambda x: x["score"], reverse=True)

        # 返回top_k个结果
        final = [item["doc"] for item in sorted_results[:top_k]]
        return final

    def _rerank(self, query: str, documents: List, top_k: int = 5) -> List:
        """
        使用CrossEncoder对检索结果进行重排序

        Args:
            query: 查询文本
            documents: 待重排序的文档列表
            top_k: 返回的结果数量

        Returns:
            重排序后的文档列表
        """
        if not documents:
            return []

        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder
                logger.debug("[Reranker] 加载CrossEncoder模型...")
                self._reranker = CrossEncoder('BAAI/bge-reranker-base')
                logger.debug("[Reranker] 模型加载完成")
            except Exception as e:
                logger.warning(f"[Reranker] 加载失败，跳过重排序: {e}")
                return documents[:top_k]

        try:
            # 构建 query-doc 对
            pairs = [(query, doc.page_content[:512]) for doc in documents]

            # 计算相关性分数
            scores = self._reranker.predict(pairs)

            # 按分数降序排列
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            return [doc for doc, _ in scored_docs[:top_k]]
        except Exception as e:
            logger.error(f"[Reranker] 重排序失败: {e}")
            return documents[:top_k]

    def search(self, query: str, top_k: int = 5, use_hybrid: bool = True,
               where: Optional[Dict] = None) -> List:
        """
        搜索相关文档

        Args:
            query: 查询文本
            top_k: 返回结果数量
            use_hybrid: 是否使用混合检索（默认True）
            where: ChromaDB 元数据过滤条件（如 {"visibility": "public"}）

        Returns:
            相关文档列表
        """
        try:
            if use_hybrid:
                # 混合检索：向量 + BM25 + RRF融合
                candidates = self.hybrid_search(query, top_k=top_k * 3, where=where)

                # 可选：重排序
                if self._reranker_enabled:
                    results = self._rerank(query, candidates, top_k=top_k)
                else:
                    results = candidates[:top_k]

                return results
            else:
                # 纯向量检索
                if where:
                    results = self.vector_store.similarity_search(query, k=top_k, filter=where)
                else:
                    results = self.vector_store.similarity_search(query, k=top_k)
                return results
        except Exception as e:
            logger.error(f"检索失败: {e}")
            return []

    def search_with_scope(self, query: str, user_id: str, scope: str = "all", top_k: int = 5) -> List:
        """
        按范围搜索文档（个人工作区 + 团队共享库）

        Args:
            query: 查询文本
            user_id: 当前用户ID
            scope: 搜索范围 - "personal"(仅个人) / "shared"(仅共享) / "all"(全部)
            top_k: 返回结果数量

        Returns:
            搜索结果列表，每个结果附带 scope 标记
        """
        results = []

        if scope in ("personal", "all"):
            # 搜索个人工作区
            personal_where = {"$and": [{"owner": user_id}, {"visibility": "private"}]}
            personal = self.search(query, top_k=top_k, use_hybrid=True, where=personal_where)
            for doc in personal:
                doc.metadata["scope"] = "personal"
            results.extend(personal)
            logger.debug(f"[RAG] 个人工作区检索到 {len(personal)} 条结果 (user={user_id})")

        if scope in ("shared", "all"):
            # 搜索团队共享库
            shared_where = {"visibility": "public"}
            shared = self.search(query, top_k=top_k, use_hybrid=True, where=shared_where)
            for doc in shared:
                doc.metadata["scope"] = "shared"
            results.extend(shared)
            logger.debug(f"[RAG] 团队共享库检索到 {len(shared)} 条结果")

        # 去重（按内容前100字符）
        seen = set()
        unique = []
        for doc in results:
            key = doc.page_content[:100]
            if key not in seen:
                seen.add(key)
                unique.append(doc)

        logger.debug(f"[RAG] 最终返回 {len(unique[:top_k])} 条结果")
        return unique[:top_k]

    def get_context(self, query: str, top_k: int = 5, user_id: str = "", scope: str = "all") -> str:
        """
        获取上下文信息（纯文本格式）

        Args:
            query: 查询文本
            top_k: 返回结果数量
            user_id: 当前用户ID
            scope: 搜索范围

        Returns:
            格式化的上下文字符串
        """
        result = self.get_context_with_metadata(query, top_k, user_id, scope)
        return result["context"]

    def get_context_with_metadata(self, query: str, top_k: int = 5, user_id: str = "", scope: str = "all") -> dict:
        """
        获取上下文信息（含元数据，用于前端预览高亮）

        Returns:
            {
                "context": str,           # 格式化的上下文字符串
                "sources": list,          # 来源文档列表 [{"filename": str, "chunk_ids": [str]}]
                "chunks": list            # 检索到的片段列表 [{"id": str, "content": str, "filename": str}]
            }
        """
        if user_id:
            docs = self.search_with_scope(query, user_id=user_id, scope=scope, top_k=top_k)
        else:
            docs = self.search(query, top_k)

        logger.debug(f"[RAG] get_context_with_metadata: user_id={user_id}, scope={scope}, 返回 {len(docs)} 条结果")

        if not docs:
            return {"context": "", "sources": [], "chunks": []}

        # 收集来源和片段信息
        source_map = {}  # filename -> chunk contents
        chunks = []
        contexts = []

        for i, doc in enumerate(docs, 1):
            original_filename = doc.metadata.get("original_filename", "")
            source = original_filename or doc.metadata.get("source", "未知来源")
            content = doc.page_content

            # 使用元数据中的 chunk_index（文件内位置），若无则用全局序号
            actual_idx = doc.metadata.get("chunk_index", i)

            # 记录片段
            chunk_id = f"chunk_{actual_idx}"
            chunks.append({
                "id": chunk_id,
                "content": content,
                "filename": source,
                "index": actual_idx
            })

            # 按文件名分组
            if source not in source_map:
                source_map[source] = []
            source_map[source].append(content)

            # 构建上下文（包含文件名和片段编号，便于 LLM 引用）
            contexts.append(f"[{actual_idx}] 来源：{source} — 片段{actual_idx}\n{content[:500]}")

        # 构建来源列表
        sources = [{"filename": fname, "chunk_count": len(cids)} for fname, cids in source_map.items()]

        context_text = "\n\n".join(contexts)

        return {
            "context": context_text,
            "sources": sources,
            "chunks": chunks
        }

    def get_document_count(self) -> int:
        """获取向量存储中的文档数量"""
        try:
            collection = self.vector_store._collection
            return collection.count()
        except Exception:
            return 0

    def clear(self):
        """清空知识库（删除向量库 + BM25索引的所有数据和文件）"""
        import shutil
        import time

        # 1. 清空向量库
        try:
            self.vector_store.delete_collection()
        except Exception:
            pass
        # 关闭 ChromaDB 客户端连接（释放 sqlite3 文件锁）
        if self._vector_store is not None:
            try:
                self._vector_store._client.close()
            except Exception:
                pass
        self._vector_store = None

        # 删除 chroma_db 物理文件（带重试，Windows 文件锁兼容）
        chroma_dir = Path(settings.CHROMA_PERSIST_DIR)
        if chroma_dir.exists():
            for attempt in range(6):
                try:
                    shutil.rmtree(chroma_dir)
                    logger.debug(f"[RAG] 已删除向量库文件: {chroma_dir}")
                    break
                except PermissionError:
                    time.sleep(0.5)
                except Exception as e:
                    logger.error(f"[RAG] 删除向量库文件失败: {e}")
                    break
            os.makedirs(chroma_dir, exist_ok=True)

        # 2. 清空BM25索引（索引文件 + 语料库文件）
        self._bm25_retriever = None
        self._bm25_corpus = []
        self._bm25_dirty = False
        self._bm25_rebuilding = False
        if self._bm25_dir.exists():
            try:
                shutil.rmtree(self._bm25_dir)
                logger.debug(f"[RAG] 已删除BM25索引: {self._bm25_dir}")
            except Exception as e:
                logger.error(f"[RAG] 删除BM25索引失败: {e}")
            os.makedirs(self._bm25_dir, exist_ok=True)

    def clear_user_documents(self, user_id: str) -> int:
        """
        清空指定用户的所有文档（个人 + 共享）

        Args:
            user_id: 用户ID

        Returns:
            删除的文档片段数
        """
        import shutil

        # 1. 从向量库中删除该用户的所有文档
        deleted_count = 0
        try:
            result = self.vector_store.get(
                where={"owner": user_id},
                include=[]
            )
            ids_to_delete = result.get("ids", [])
            if ids_to_delete:
                self.vector_store.delete(ids=ids_to_delete)
                deleted_count = len(ids_to_delete)
                logger.debug(f"[RAG] 已删除用户 {user_id} 的 {deleted_count} 个向量片段")
        except Exception as e:
            logger.error(f"[RAG] 删除用户向量数据失败: {e}")

        # 2. 从 BM25 语料库中删除该用户的文档（跨平台路径匹配）
        user_path_pattern = str(Path(settings.UPLOAD_DIR) / user_id)
        original_count = len(self._bm25_corpus)
        self._bm25_corpus = [
            doc for doc in self._bm25_corpus
            if not doc.get("file_path", "").startswith(user_path_pattern)
        ]
        bm25_removed = original_count - len(self._bm25_corpus)
        if bm25_removed > 0:
            self._build_bm25_index()

        # 3. 删除该用户的上传文件目录
        upload_dir = Path(settings.UPLOAD_DIR) / user_id
        if upload_dir.exists():
            try:
                shutil.rmtree(upload_dir)
                logger.debug(f"[RAG] 已删除用户 {user_id} 的上传文件目录")
            except Exception as e:
                logger.error(f"[RAG] 删除用户上传目录失败: {e}")

        return deleted_count
