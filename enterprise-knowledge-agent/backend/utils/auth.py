"""JWT 认证工具模块"""

import jwt
import bcrypt
import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from backend.config.settings import settings, data_dir

logger = logging.getLogger(__name__)

# JWT 配置（从 settings 读取，必填项）
SECRET_KEY = settings.JWT_SECRET_KEY
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 小时

# 用户数据存储路径（统一使用 backend/data/ 目录）
USERS_FILE = data_dir / "users.json"

# 用户文件写锁（防止并发注册导致数据覆盖）
_users_lock = threading.Lock()


def _load_users() -> dict:
    """加载用户数据"""
    if USERS_FILE.exists():
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_users(users: dict):
    """保存用户数据（带并发锁）"""
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _users_lock:
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(users, f, ensure_ascii=False, indent=2)


def hash_password(password: str) -> str:
    """对密码进行 bcrypt 哈希"""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    """验证密码"""
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


def create_access_token(user_id: str, expires_delta: Optional[timedelta] = None) -> str:
    """
    创建 JWT Access Token

    Args:
        user_id: 用户ID
        expires_delta: 过期时间增量

    Returns:
        JWT Token 字符串
    """
    if expires_delta is None:
        expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    now = datetime.now(timezone.utc)
    expire = now + expires_delta
    payload = {
        "sub": user_id,
        "exp": expire,
        "iat": now
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> Optional[str]:
    """
    验证 JWT Token 并提取 user_id

    Args:
        token: JWT Token 字符串

    Returns:
        user_id 或 None（验证失败时）
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            return None
        return user_id
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def register_user(user_id: str, password: str) -> tuple[bool, str]:
    """
    注册用户

    Args:
        user_id: 用户ID
        password: 密码

    Returns:
        (成功与否, 消息)
    """
    if not user_id or not password:
        return False, "用户ID和密码不能为空"

    if len(user_id) < 2:
        return False, "用户ID至少2个字符"

    if len(password) < 4:
        return False, "密码至少4个字符"

    users = _load_users()

    if user_id in users:
        return False, "用户ID已存在"

    users[user_id] = {
        "password_hash": hash_password(password),
        "created_at": datetime.now().isoformat()
    }
    _save_users(users)

    return True, "注册成功"


def authenticate_user(user_id: str, password: str) -> Optional[str]:
    """
    验证用户凭据并返回 Token

    Args:
        user_id: 用户ID
        password: 密码

    Returns:
        JWT Token 或 None（验证失败时）
    """
    users = _load_users()

    if user_id not in users:
        return None

    if not verify_password(password, users[user_id]["password_hash"]):
        return None

    return create_access_token(user_id)
