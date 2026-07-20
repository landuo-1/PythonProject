"""文档API - 处理文档上传和管理"""

import logging
import re
from typing import List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from pathlib import Path

from backend.models.schemas import DocumentUploadResponse
from backend.utils.file_utils import save_upload_file, is_allowed_file, list_files, delete_file, compute_file_hash
from backend.config.settings import settings
from backend.api.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/documents", tags=["文档管理"])


def _sanitize_user_id(user_id: str) -> str:
    """清洗 user_id，防止路径注入"""
    return re.sub(r'[^\w\-]', '_', user_id)

# Agent 实例管理（与chat模块共享）
from backend.api.chat import get_agent


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    visibility: str = Form("private"),
    replace: bool = Form(False),
    user_id: str = Depends(get_current_user)
):
    """
    上传文档到知识库（自动去重，支持替换，需要登录）

    Args:
        file: 上传的文件
        visibility: 可见性 - "private"(个人工作区) / "public"(团队共享库)
        replace: 是否替换已存在的同名文档（默认 False）
    """
    try:
        user_id = _sanitize_user_id(user_id)

        # 参数校验
        if visibility not in ("private", "public"):
            raise HTTPException(status_code=400, detail="visibility 必须是 'private' 或 'public'")

        logger.info(f"[API] upload: user={user_id}, vis={visibility}, replace={replace}, file={file.filename}")

        # 检查文件格式
        if not is_allowed_file(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件格式。支持的格式: {', '.join(settings.ALLOWED_EXTENSIONS)}"
            )

        # 读取文件内容（用于大小检查）
        file_content = await file.read()
        file_size = len(file_content)

        # 检查文件大小
        if file_size > settings.MAX_FILE_SIZE:
            max_size_mb = settings.MAX_FILE_SIZE / (1024 * 1024)
            raise HTTPException(
                status_code=400,
                detail=f"文件大小超过限制。最大允许: {max_size_mb}MB"
            )

        # 获取Agent实例
        agent = get_agent(user_id)

        # 计算文件内容哈希（用于精确去重）
        content_hash = compute_file_hash(file_content)
        logger.info(f"[API] upload: hash={content_hash[:8]}..., size={file_size}")

        # 两阶段去重检查
        chunks_deleted = 0
        # 阶段1：精确匹配（文件名 + 内容哈希）→ 完全相同的文件，直接跳过
        if agent.rag_engine.has_document(file.filename, owner=user_id, visibility=visibility, content_hash=content_hash):
            scope_name = "个人工作区" if visibility == "private" else "团队共享库"
            return DocumentUploadResponse(
                success=False,
                message=f"文档 '{file.filename}' 已存在于{scope_name}中（内容完全相同），无需重复上传",
                filename=file.filename,
                chunks_added=0,
                total_documents=agent.rag_engine.get_document_count()
            )

        # 阶段2：名称匹配（同名但内容不同）→ 根据 replace 参数决定行为
        if agent.rag_engine.has_document(file.filename, owner=user_id, visibility=visibility):
            if replace:
                # 替换模式：检查是否为文档所有者
                if not agent.rag_engine.is_document_owner(file.filename, user_id, visibility):
                    scope_name = "个人工作区" if visibility == "private" else "团队共享库"
                    return DocumentUploadResponse(
                        success=False,
                        message=f"文档 '{file.filename}' 已被其他用户上传到{scope_name}，无权替换",
                        filename=file.filename,
                        chunks_added=0,
                        total_documents=agent.rag_engine.get_document_count()
                    )
                # 删除旧文档后上传新文档
                logger.info(f"[API] 替换文档: {file.filename} (user={user_id}, vis={visibility})")
                chunks_deleted = agent.rag_engine.delete_document(
                    "", original_filename=file.filename, owner=user_id, visibility=visibility
                )
                # 删除旧的磁盘文件
                try:
                    upload_dir = str(Path(settings.UPLOAD_DIR) / user_id)
                    for f in list_files(upload_dir):
                        if f["name"].endswith(file.filename):
                            delete_file(f["path"])
                except Exception:
                    pass
            else:
                # 非替换模式：返回提示
                scope_name = "个人工作区" if visibility == "private" else "团队共享库"
                return DocumentUploadResponse(
                    success=False,
                    message=f"文档 '{file.filename}' 已存在于{scope_name}中（内容不同）。如需更新，请勾选「替换已有文档」",
                    filename=file.filename,
                    chunks_added=0,
                    total_documents=agent.rag_engine.get_document_count()
                )

        # 保存文件到磁盘
        logger.info(f"[API] 保存文件到磁盘...")
        file.file.seek(0)
        file_path = await save_upload_file(file, sub_dir=user_id)
        logger.info(f"[API] 文件已保存: {file_path}")

        # 添加文档到知识库（附带 visibility、owner、content_hash 元数据）
        logger.info(f"[API] 添加到知识库: {file.filename}, hash={content_hash[:8]}")
        chunks_added = agent.add_document(file_path, metadata={
            "visibility": visibility,
            "owner": user_id
        }, content_hash=content_hash)
        logger.info(f"[API] 知识库添加完成: {chunks_added} 个片段")

        # 获取知识库信息
        logger.info(f"[API] 获取知识库信息...")
        kb_info = agent.get_knowledge_base_info()

        action = "替换" if replace and chunks_deleted else "上传"
        return DocumentUploadResponse(
            success=True,
            message=f"文档{action}成功",
            filename=file.filename,
            chunks_added=chunks_added,
            chunks_deleted=chunks_deleted,
            total_documents=kb_info["document_count"]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] 文档上传失败: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"文档上传失败: {str(e)}")


@router.get("/list")
async def list_documents(
    visibility: str = "",
    user_id: str = Depends(get_current_user)
):
    """
    列出用户知识库中的文档（需要登录）

    Args:
        visibility: 过滤可见性 - "private"(仅自己的私有) / "public"(所有人的公开) / 空(自己的私有+所有公开)
    """
    try:
        agent = get_agent(user_id)

        if visibility == "private":
            # 仅自己的私有文档
            documents = agent.rag_engine.list_documents(owner=user_id, visibility="private")
        elif visibility == "public":
            # 所有人的公开文档
            documents = agent.rag_engine.list_documents(visibility="public")
        else:
            # 自己的私有 + 所有人的公开（按 filename + visibility + owner 去重，保留同名不同空间的文档）
            own_private = agent.rag_engine.list_documents(owner=user_id, visibility="private")
            all_public = agent.rag_engine.list_documents(visibility="public")
            seen = set()
            documents = []
            for doc in own_private + all_public:
                fname = doc.get("original_filename", "")
                vis = doc.get("visibility", "private")
                own = doc.get("owner", "")
                key = f"{fname}|{vis}|{own}"
                if key not in seen:
                    seen.add(key)
                    documents.append(doc)

        return {
            "user_id": user_id,
            "documents": documents,
            "count": len(documents)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")


@router.get("/content/{filename}")
async def get_document_content(filename: str, visibility: str = "", user_id: str = Depends(get_current_user)):
    """
    获取指定文档的内容片段（需要登录）

    Args:
        filename: 文档文件名
        visibility: 可见性过滤 - "private"(仅自己的私有) / "public"(仅公开) / 空(合并两者)
    """
    try:
        agent = get_agent(user_id)
        store = agent.rag_engine.vector_store

        documents = []
        metadatas = []

        if visibility == "private":
            # 仅自己的私有文档
            result = store.get(
                where={"$and": [{"original_filename": filename}, {"owner": user_id}, {"visibility": "private"}]},
                include=["documents", "metadatas"]
            )
            documents = list(result.get("documents", []))
            metadatas = list(result.get("metadatas", []))

        elif visibility == "public":
            # 仅公开文档
            result = store.get(
                where={"$and": [{"original_filename": filename}, {"visibility": "public"}]},
                include=["documents", "metadatas"]
            )
            documents = list(result.get("documents", []))
            metadatas = list(result.get("metadatas", []))

        else:
            # 合并：自己的私有 + 所有公开
            own_result = store.get(
                where={"$and": [{"original_filename": filename}, {"owner": user_id}]},
                include=["documents", "metadatas"]
            )
            public_result = store.get(
                where={"$and": [{"original_filename": filename}, {"visibility": "public"}]},
                include=["documents", "metadatas"]
            )
            documents = list(own_result.get("documents", []))
            metadatas = list(own_result.get("metadatas", []))
            seen_contents = set(documents)
            for doc, meta in zip(public_result.get("documents", []), public_result.get("metadatas", [])):
                if doc not in seen_contents:
                    documents.append(doc)
                    metadatas.append(meta)
                    seen_contents.add(doc)

        if not documents:
            raise HTTPException(status_code=404, detail=f"文档 '{filename}' 不存在")

        # 组装片段内容
        chunks = []
        for i, (content, meta) in enumerate(zip(documents, metadatas), 1):
            chunks.append({
                "index": i,
                "content": content,
                "source": meta.get("source", ""),
                "visibility": meta.get("visibility", "private")
            })

        return {
            "filename": filename,
            "chunks": chunks,
            "total_chunks": len(chunks)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取文档内容失败: {str(e)}")


@router.delete("/{filename}")
async def delete_document(filename: str, visibility: str = "", user_id: str = Depends(get_current_user)):
    """
    删除文档（需要登录）

    Args:
        visibility: 过滤可见性 - "private" / "public" / 空(删所有匹配的)
    """
    try:
        agent = get_agent(user_id)

        # 从向量库查询该文件的所有 file_path（在删除前获取，用于清理磁盘文件）
        conditions = [{"original_filename": filename}, {"owner": user_id}]
        if visibility:
            conditions.append({"visibility": visibility})
        where_clause = {"$and": conditions} if len(conditions) > 1 else conditions[0]

        file_paths_to_delete = set()
        try:
            result = agent.rag_engine.vector_store.get(where=where_clause, include=["metadatas"])
            for meta in result.get("metadatas", []):
                fpath = meta.get("file_path", "")
                if fpath:
                    file_paths_to_delete.add(fpath)
        except Exception:
            pass

        # 从向量库和 BM25 中删除（按 original_filename + owner + visibility 精确过滤）
        deleted_from_store = agent.rag_engine.delete_document(
            "", original_filename=filename, owner=user_id, visibility=visibility
        )

        # 删除磁盘文件（从向量库元数据获取的路径，比遍历磁盘更可靠）
        for fpath in file_paths_to_delete:
            delete_file(fpath)

        # 清理对话摘要，防止 LLM 从摘要中引用已删除的文档
        if deleted_from_store > 0:
            agent.memory_manager.update_conversation_summary("")

        return {
            "message": "文档删除成功",
            "filename": filename,
            "chunks_deleted": deleted_from_store
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除文档失败: {str(e)}")


@router.post("/batch-upload")
async def batch_upload_documents(
    files: List[UploadFile] = File(...),
    visibility: str = Form("private"),
    replace: bool = Form(False),
    user_id: str = Depends(get_current_user)
):
    """批量上传文档（自动去重，支持替换，需要登录）"""
    try:
        user_id = _sanitize_user_id(user_id)
        results = []
        total_chunks = 0

        for file in files:
            if not is_allowed_file(file.filename):
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "message": "不支持的文件格式"
                })
                continue

            try:
                agent = get_agent(user_id)
                chunks_deleted = 0

                # 计算文件内容哈希
                file_content = await file.read()
                content_hash = compute_file_hash(file_content)
                file.file.seek(0)

                # 阶段1：精确匹配（内容完全相同，跳过）
                if agent.rag_engine.has_document(file.filename, owner=user_id, visibility=visibility, content_hash=content_hash):
                    results.append({
                        "filename": file.filename,
                        "success": False,
                        "message": "文档已存在且内容相同，跳过"
                    })
                    continue

                # 阶段2：名称匹配（同名不同内容）
                if agent.rag_engine.has_document(file.filename, owner=user_id, visibility=visibility):
                    if replace:
                        if not agent.rag_engine.is_document_owner(file.filename, user_id, visibility):
                            results.append({
                                "filename": file.filename,
                                "success": False,
                                "message": "文档已被其他用户上传，无权替换"
                            })
                            continue
                        chunks_deleted = agent.rag_engine.delete_document(
                            "", original_filename=file.filename, owner=user_id, visibility=visibility
                        )
                    else:
                        results.append({
                            "filename": file.filename,
                            "success": False,
                            "message": "文档已存在但内容不同，跳过（勾选「替换」可覆盖）"
                        })
                        continue

                file_path = await save_upload_file(file, sub_dir=user_id)
                chunks_added = agent.add_document(file_path, metadata={
                    "visibility": visibility,
                    "owner": user_id
                }, content_hash=content_hash)
                total_chunks += chunks_added

                results.append({
                    "filename": file.filename,
                    "success": True,
                    "chunks_added": chunks_added,
                    "chunks_deleted": chunks_deleted
                })
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "message": str(e)
                })

        agent = get_agent(user_id)
        kb_info = agent.get_knowledge_base_info()

        return {
            "results": results,
            "total_files": len(files),
            "successful_files": sum(1 for r in results if r["success"]),
            "total_chunks_added": total_chunks,
            "total_documents": kb_info["document_count"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量上传失败: {str(e)}")
