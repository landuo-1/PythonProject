"""认证API - 登录和注册"""

import logging
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Optional

from backend.utils.auth import register_user, authenticate_user, verify_token

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["认证"])


class RegisterRequest(BaseModel):
    """注册请求"""
    user_id: str = Field(..., min_length=2, max_length=20, description="用户ID")
    password: str = Field(..., min_length=4, max_length=50, description="密码")


class LoginRequest(BaseModel):
    """登录请求"""
    user_id: str = Field(..., description="用户ID")
    password: str = Field(..., description="密码")


class AuthResponse(BaseModel):
    """认证响应"""
    success: bool
    message: str
    token: Optional[str] = None
    user_id: Optional[str] = None


@router.post("/register", response_model=AuthResponse)
async def register(request: RegisterRequest):
    """用户注册"""
    success, message = register_user(request.user_id, request.password)

    if not success:
        raise HTTPException(status_code=400, detail=message)

    # 注册成功后自动登录
    token = authenticate_user(request.user_id, request.password)

    return AuthResponse(
        success=True,
        message=message,
        token=token,
        user_id=request.user_id
    )


@router.post("/login", response_model=AuthResponse)
async def login(request: LoginRequest):
    """用户登录"""
    token = authenticate_user(request.user_id, request.password)

    if token is None:
        raise HTTPException(status_code=401, detail="用户ID或密码错误")

    return AuthResponse(
        success=True,
        message="登录成功",
        token=token,
        user_id=request.user_id
    )


# 使用 HTTPBearer 让 Swagger UI 的 Authorize 按钮自动生效
_bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme)
) -> str:
    """
    从请求头提取当前用户（FastAPI 依赖注入）

    优先使用 HTTPBearer 解析（兼容 Swagger UI），
    若 HTTPBearer 未获取到则手动解析 Authorization 头（兼容流式请求等场景）。

    Returns:
        user_id
    """
    token = None

    # 优先从 HTTPBearer 获取
    if credentials is not None:
        token = credentials.credentials

    # 回退：手动解析 Authorization 头
    if not token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]

    if not token:
        logger.warning(f"[AUTH] 401: 未提供认证信息 | path={request.url.path}")
        raise HTTPException(status_code=401, detail="未提供认证信息")

    user_id = verify_token(token)

    if user_id is None:
        logger.warning(f"[AUTH] 401: Token无效或已过期 | path={request.url.path}")
        raise HTTPException(status_code=401, detail="Token无效或已过期")

    return user_id
