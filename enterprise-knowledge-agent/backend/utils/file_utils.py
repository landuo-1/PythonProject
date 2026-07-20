"""文件工具函数 - 处理文件上传和管理"""

import hashlib
import logging
import os
import shutil
from typing import Optional
from pathlib import Path
from datetime import datetime

from fastapi import UploadFile
from backend.config.settings import settings

logger = logging.getLogger(__name__)


def compute_file_hash(data: bytes) -> str:
    """
    计算文件内容的 MD5 哈希值（用于内容级去重）

    Args:
        data: 文件内容字节

    Returns:
        MD5 哈希字符串（32位小写十六进制）
    """
    return hashlib.md5(data).hexdigest()


async def save_upload_file(file: UploadFile, sub_dir: str = "") -> str:
    """
    保存上传的文件

    Args:
        file: 上传的文件
        sub_dir: 子目录

    Returns:
        保存的文件路径
    """
    # 创建上传目录
    upload_dir = Path(settings.UPLOAD_DIR)
    if sub_dir:
        upload_dir = upload_dir / sub_dir
    upload_dir.mkdir(parents=True, exist_ok=True)

    # 生成唯一的文件名（防止路径遍历攻击）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = Path(file.filename).name  # 去掉目录部分，只保留文件名
    file_ext = get_file_extension(safe_name)
    new_filename = f"{timestamp}_{safe_name}"
    file_path = upload_dir / new_filename

    # 保存文件
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return str(file_path)


def get_file_extension(filename: str) -> str:
    """
    获取文件扩展名

    Args:
        filename: 文件名

    Returns:
        文件扩展名（包含点号）
    """
    return Path(filename).suffix.lower()


def is_allowed_file(filename: str) -> bool:
    """
    检查文件是否允许上传

    Args:
        filename: 文件名

    Returns:
        是否允许
    """
    file_ext = get_file_extension(filename)
    return file_ext in settings.ALLOWED_EXTENSIONS


def get_file_size(file_path: str) -> int:
    """
    获取文件大小

    Args:
        file_path: 文件路径

    Returns:
        文件大小（字节）
    """
    return os.path.getsize(file_path)


def delete_file(file_path: str) -> bool:
    """
    删除文件

    Args:
        file_path: 文件路径

    Returns:
        是否删除成功
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False
    except Exception as e:
        logger.error(f"删除文件失败: {e}")
        return False


def list_files(directory: str, extensions: Optional[list] = None) -> list:
    """
    列出目录下的文件

    Args:
        directory: 目录路径
        extensions: 文件扩展名过滤

    Returns:
        文件列表
    """
    files = []
    dir_path = Path(directory)

    if not dir_path.exists():
        return files

    for file_path in dir_path.iterdir():
        if file_path.is_file():
            if extensions is None or file_path.suffix.lower() in extensions:
                files.append({
                    "name": file_path.name,
                    "path": str(file_path),
                    "size": file_path.stat().st_size,
                    "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                })

    return files


def create_directory(directory: str) -> bool:
    """
    创建目录

    Args:
        directory: 目录路径

    Returns:
        是否创建成功
    """
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"创建目录失败: {e}")
        return False


def get_directory_size(directory: str) -> int:
    """
    获取目录大小

    Args:
        directory: 目录路径

    Returns:
        目录大小（字节）
    """
    total_size = 0
    dir_path = Path(directory)

    if not dir_path.exists():
        return total_size

    for file_path in dir_path.rglob("*"):
        if file_path.is_file():
            total_size += file_path.stat().st_size

    return total_size


def format_file_size(size_bytes: int) -> str:
    """
    格式化文件大小

    Args:
        size_bytes: 文件大小（字节）

    Returns:
        格式化的文件大小
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"
