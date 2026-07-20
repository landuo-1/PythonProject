"""数据模型定义 - 使用Pydantic进行数据验证"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class ChatRequest(BaseModel):
    """聊天请求模型"""
    message: str = Field(..., description="用户消息", min_length=1, max_length=2000)
    web_search: bool = Field(False, description="是否启用网络搜索")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "请介绍一下公司的产品",
                "web_search": False
            }
        }


class ChatResponse(BaseModel):
    """聊天响应模型"""
    response: str = Field(..., description="Agent响应")
    user_id: str = Field(..., description="用户ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="响应时间")
    sources: Optional[List[Dict[str, Any]]] = Field(None, description="引用来源")
    chunks: Optional[List[Dict[str, Any]]] = Field(None, description="检索片段")

    class Config:
        json_schema_extra = {
            "example": {
                "response": "根据知识库信息，公司的主要产品包括...",
                "user_id": "user123",
                "timestamp": "2024-01-01T12:00:00",
                "sources": [{"filename": "产品手册.pdf", "chunk_count": 3}],
                "chunks": [{"id": "chunk_1", "content": "...", "filename": "产品手册.pdf"}]
            }
        }


class DocumentUploadResponse(BaseModel):
    """文档上传响应模型"""
    success: bool = Field(..., description="上传是否成功")
    message: str = Field(..., description="响应消息")
    filename: str = Field(..., description="文件名")
    chunks_added: int = Field(default=0, description="添加的文档块数量")
    chunks_deleted: int = Field(default=0, description="替换时删除的旧文档块数量")
    total_documents: int = Field(default=0, description="知识库总文档数")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "文档上传成功",
                "filename": "产品手册.pdf",
                "chunks_added": 42,
                "chunks_deleted":5,
                "total_documents": 150
            }
        }


class KnowledgeBaseInfo(BaseModel):
    """知识库信息模型"""
    document_count: int = Field(..., description="文档数量")
    memory_stats: Dict[str, Any] = Field(..., description="记忆统计信息")
    last_updated: Optional[datetime] = Field(None, description="最后更新时间")

    class Config:
        json_schema_extra = {
            "example": {
                "document_count": 150,
                "memory_stats": {
                    "conversation_count": 10,
                    "preference_count": 5,
                    "knowledge_count": 20
                },
                "last_updated": "2024-01-01T12:00:00"
            }
        }


class ConversationHistory(BaseModel):
    """对话历史模型"""
    messages: List[Dict[str, Any]] = Field(..., description="消息列表")
    total_count: int = Field(..., description="消息总数")

    class Config:
        json_schema_extra = {
            "example": {
                "messages": [
                    {
                        "role": "human",
                        "content": "你好",
                        "timestamp": "2024-01-01T12:00:00"
                    },
                    {
                        "role": "assistant",
                        "content": "你好！我是企业知识库助手，有什么可以帮您的吗？",
                        "timestamp": "2024-01-01T12:00:01"
                    }
                ],
                "total_count": 2
            }
        }


class SystemStatus(BaseModel):
    """系统状态模型"""
    status: str = Field(..., description="系统状态")
    version: str = Field(default="1.0.0", description="系统版本")
    uptime: Optional[str] = Field(None, description="运行时间")
    components: Dict[str, str] = Field(..., description="组件状态")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "uptime": "2小时30分钟",
                "components": {
                    "llm": "正常",
                    "rag": "正常",
                    "memory": "正常",
                    "tools": "正常"
                }
            }
        }
