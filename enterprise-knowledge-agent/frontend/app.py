"""Streamlit前端主应用 - 企业智能知识库Agent"""

import json
import os
import queue
import threading
import time
import streamlit as st
import requests
from typing import Dict, Any

# API配置（支持环境变量覆盖，Docker Compose 中使用 http://backend:8000）
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")


# 页面配置
st.set_page_config(
    page_title="企业智能知识库Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS + JS（小标题样式增强）
st.markdown(r"""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stMarkdown a[href^="#"] {
        display: none !important;
    }
    /* 小标题 class 由 JS 动态添加 */
    .chat-subtitle {
        font-size: 1.25em !important;
        color: #1a1a1a !important;
        font-weight: 700 !important;
        display: block;
        margin-top: 0.5em !important;
        margin-bottom: 0.2em !important;
    }
    /* 行内加粗 */
    .stChatMessage .stMarkdown p strong {
        color: #1a1a1a;
        font-weight: 600;
    }
    /* 来源标注 */
    .stChatMessage .stMarkdown em {
        color: #5f6368;
        font-size: 0.9em;
    }
</style>
<script>
// 检测小标题：独占一段且以数字开头的加粗文本
function styleSubtitles() {
    document.querySelectorAll('.stChatMessage .stMarkdown p').forEach(p => {
        const children = p.childNodes;
        // 只有一个子节点且是 <strong>
        if (children.length === 1 && children[0].nodeName === 'STRONG') {
            const text = children[0].textContent.trim();
            // 匹配 "1. xxx" 或 "1、xxx" 格式
            if (/^\d+[\.\、]/.test(text)) {
                children[0].classList.add('chat-subtitle');
            }
        }
    });
}
// MutationObserver 监听 DOM 变化
const observer = new MutationObserver(styleSubtitles);
observer.observe(document.body, { childList: true, subtree: true });
styleSubtitles();
</script>
""", unsafe_allow_html=True)


# ==================== 会话状态 ====================

def init_session_state():
    """初始化会话状态"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_id" not in st.session_state:
        st.session_state.user_id = ""
    if "token" not in st.session_state:
        st.session_state.token = ""
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
    if "_is_chatting" not in st.session_state:
        st.session_state._is_chatting = False
    if "_pending_prompt" not in st.session_state:
        st.session_state._pending_prompt = ""
    if "web_search" not in st.session_state:
        st.session_state.web_search = False


def get_auth_headers() -> dict:
    """获取认证请求头"""
    # 尝试多种方式获取 token，确保兼容性
    token = ""
    try:
        token = st.session_state.token
    except (AttributeError, KeyError):
        try:
            token = st.session_state["token"]
        except (KeyError, TypeError):
            token = ""
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


# ==================== 认证 API ====================

def api_register(user_id: str, password: str) -> Dict[str, Any]:
    """注册"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/auth/register",
            json={"user_id": user_id, "password": password},
            timeout=10
        )
        return response.json()
    except Exception as e:
        return {"success": False, "message": f"注册失败: {str(e)}"}


def api_login(user_id: str, password: str) -> Dict[str, Any]:
    """登录"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/auth/login",
            json={"user_id": user_id, "password": password},
            timeout=10
        )
        return response.json()
    except Exception as e:
        return {"success": False, "message": f"登录失败: {str(e)}"}


# ==================== 业务 API ====================

def send_message_stream(message: str, token: str = "", web_search: bool = False):
    """流式发送消息到后端API（返回结构化数据）

    Args:
        token: JWT token（由调用方在主线程捕获，避免后台线程访问 session_state）
    """
    try:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        response = requests.post(
            f"{API_BASE_URL}/api/chat/stream",
            json={"message": message, "web_search": web_search},
            headers=headers,
            timeout=60,
            stream=True
        )
        if response.status_code == 200:
            for line in response.iter_lines(decode_unicode=True):
                if line and line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        yield data
                    except json.JSONDecodeError:
                        yield {"type": "chunk", "data": data_str}
        elif response.status_code == 401:
            yield {"type": "chunk", "data": "⚠️ 登录已过期，请刷新页面重新登录"}
        else:
            yield {"type": "chunk", "data": f"错误: {response.status_code}"}
    except requests.exceptions.ConnectionError:
        yield {"type": "chunk", "data": "⚠️ 无法连接到后端服务"}
    except Exception as e:
        yield {"type": "chunk", "data": f"❌ 请求失败: {str(e)}"}


def upload_document(file, visibility: str = "private", replace: bool = False) -> Dict[str, Any]:
    """上传文档到知识库"""
    try:
        files = {"file": (file.name, file, file.type)}
        response = requests.post(
            f"{API_BASE_URL}/api/documents/upload",
            files=files,
            data={"visibility": visibility, "replace": str(replace).lower()},
            headers=get_auth_headers(),
            timeout=120
        )
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            return {"success": False, "message": "登录已过期，请重新登录"}
        return {"success": False, "message": f"上传失败: {response.text}"}
    except Exception as e:
        return {"success": False, "message": f"上传错误: {str(e)}"}


def clear_conversation():
    """清空对话历史"""
    try:
        response = requests.delete(
            f"{API_BASE_URL}/api/chat/history",
            headers=get_auth_headers(),
            timeout=10
        )
        if response.status_code == 200:
            st.session_state.messages = []
            # 清除所有预览状态（per-message keys）
            keys_to_remove = [k for k in st.session_state if k.startswith("_preview_")]
            for k in keys_to_remove:
                del st.session_state[k]
            st.session_state._clear_done = True
            st.rerun()
        else:
            st.error("清空失败")
    except Exception as e:
        st.error(f"清空错误: {str(e)}")


def clear_my_documents():
    """清空当前用户的所有文档"""
    try:
        response = requests.delete(
            f"{API_BASE_URL}/api/knowledge-base/my-documents",
            headers=get_auth_headers(),
            timeout=15
        )
        if response.status_code == 200:
            data = response.json()
            st.success(f"已清空 {data.get('deleted_count', 0)} 个文档片段")
        else:
            st.error(f"清空失败: {response.text}")
    except Exception as e:
        st.error(f"清空错误: {str(e)}")


def get_document_list(visibility: str = "") -> list:
    """获取知识库中的文档列表"""
    try:
        params = {}
        if visibility:
            params["visibility"] = visibility
        response = requests.get(
            f"{API_BASE_URL}/api/documents/list",
            headers=get_auth_headers(),
            params=params,
            timeout=5
        )
        if response.status_code == 200:
            return response.json().get("documents", [])
        return []
    except Exception:
        return []


def get_document_content(filename: str, visibility: str = "") -> Dict[str, Any]:
    """获取文档内容"""
    try:
        params = {}
        if visibility:
            params["visibility"] = visibility
        response = requests.get(
            f"{API_BASE_URL}/api/documents/content/{filename}",
            headers=get_auth_headers(),
            params=params,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return {}
    except Exception:
        return {}


def delete_document_api(filename: str, visibility: str = "") -> Dict[str, Any]:
    """删除文档"""
    try:
        params = {}
        if visibility:
            params["visibility"] = visibility
        response = requests.delete(
            f"{API_BASE_URL}/api/documents/{filename}",
            headers=get_auth_headers(),
            params=params,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return {"message": f"删除失败: {response.text}"}
    except Exception as e:
        return {"message": f"删除错误: {str(e)}"}


# ==================== 登录页面 ====================

def render_login_page():
    """渲染登录/注册页面"""
    st.markdown('<h1 class="main-header">🤖 企业智能知识库Agent</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">基于 LangGraph + RAG 混合检索的企业知识问答系统</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        tab_login, tab_register = st.tabs(["🔐 登录", "📝 注册"])

        with tab_login:
            with st.form("login_form"):
                login_user = st.text_input(
                    "用户ID",
                    key="login_user",
                    placeholder="输入你的用户ID"
                )
                login_pass = st.text_input(
                    "密码",
                    type="password",
                    key="login_pass",
                    placeholder="输入你的密码"
                )
                login_submit = st.form_submit_button("登 录", use_container_width=True, type="primary")

                if login_submit:
                    if not login_user or not login_pass:
                        st.warning("⚠️ 请输入用户ID和密码")
                    else:
                        result = api_login(login_user, login_pass)
                        if result.get("success"):
                            st.session_state.token = result["token"]
                            st.session_state.user_id = result["user_id"]
                            st.success("✅ 登录成功，正在跳转...")
                            st.rerun()
                        else:
                            st.error(f"❌ {result.get('message', '登录失败')}")

            st.info("💡 已有账号直接登录，新用户请切换到「注册」标签")

        with tab_register:
            with st.form("register_form"):
                reg_user = st.text_input(
                    "用户ID",
                    key="reg_user",
                    placeholder="2-20个字符，用于标识你的知识库"
                )
                reg_pass = st.text_input(
                    "密码",
                    type="password",
                    key="reg_pass",
                    placeholder="至少4个字符"
                )
                reg_submit = st.form_submit_button("注册并登录", use_container_width=True, type="primary")

                if reg_submit:
                    if not reg_user or not reg_pass:
                        st.warning("⚠️ 请输入用户ID和密码")
                    elif len(reg_user) < 2:
                        st.warning("⚠️ 用户ID至少2个字符")
                    elif len(reg_pass) < 4:
                        st.warning("⚠️ 密码至少4个字符")
                    else:
                        result = api_register(reg_user, reg_pass)
                        if result.get("success"):
                            st.session_state.token = result["token"]
                            st.session_state.user_id = result["user_id"]
                            st.success("✅ 注册成功，正在跳转...")
                            st.rerun()
                        else:
                            st.error(f"❌ {result.get('message', '注册失败')}")

            st.info("💡 注册后自动登录，每个用户拥有独立的个人知识库")


# ==================== 侧边栏 ====================

def render_sidebar():
    """渲染侧边栏（对话进行中按钮禁用 + st.stop 守卫）"""
    with st.sidebar:
        _sidebar_content()


@st.fragment
def _sidebar_content():
    """侧边栏内容（fragment：独立于主脚本，聊天流式输出时仍可操作）"""
    user_id = st.session_state.get("user_id", "")
    is_chatting = st.session_state.get("_is_chatting", False)
    st.markdown(f"## 👤 {user_id}")

    # 网络搜索开关（使用 key 自动同步 session_state）
    st.toggle(
        "🌐 启用网络搜索",
        key="web_search",
        disabled=is_chatting,
        help="启用后，Agent 可以搜索互联网获取实时信息（天气、新闻等）"
    )

    st.markdown("---")

    # 文档上传
    st.markdown("### 📤 上传文档")
    uploaded_files = st.file_uploader(
        "选择文件",
        type=["pdf", "docx", "txt", "xlsx", "csv"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"uploader_{st.session_state.get('uploader_key', 0)}",
        disabled=is_chatting
    )

    if uploaded_files and not is_chatting:
        visibility = st.radio(
            "存储位置",
            ["private", "public"],
            format_func=lambda x: "🔒 个人工作区" if x == "private" else "🌐 团队共享库",
            horizontal=True
        )
        replace_existing = st.checkbox("🔄 替换已有同名文档", help="勾选后，同名文档将被新文件覆盖")

        if st.button("📤 上传到知识库", use_container_width=True):
            if st.session_state.get("_is_chatting", False):
                st.stop()
            progress_bar = st.progress(0)
            status_text = st.empty()
            for i, file in enumerate(uploaded_files):
                status_text.text(f"上传中: {file.name}")
                result = upload_document(file, visibility=visibility, replace=replace_existing)
                progress_bar.progress((i + 1) / len(uploaded_files))
                if result.get("success"):
                    deleted = result.get("chunks_deleted", 0)
                    added = result.get("chunks_added", 0)
                    if deleted:
                        st.success(f"✅ {file.name}（替换：删除{deleted}个旧片段，新增{added}个片段）")
                    else:
                        st.success(f"✅ {file.name}（{added}个片段）")
                else:
                    st.error(f"❌ {file.name}: {result.get('message')}")
            status_text.text("上传完成！")
            st.session_state.uploader_key = st.session_state.get("uploader_key", 0) + 1
            st.rerun()

    st.markdown("---")

    # 个人工作区文档
    st.markdown("### 🔒 个人工作区")
    personal_docs = get_document_list(visibility="private")
    if personal_docs:
        for doc in personal_docs:
            fname = doc.get("original_filename", "未知")
            chunk_count = doc.get("chunk_count", 0)
            col_name, col_view, col_del = st.columns([3, 1, 1])
            with col_name:
                st.caption(f"📄 {fname} ({chunk_count}段)")
            with col_view:
                if st.button("👁️", key=f"view_p_{fname}", help=f"查看 {fname}", disabled=is_chatting):
                    if st.session_state.get("_is_chatting", False):
                        st.stop()
                    st.session_state._sidebar_preview_doc = (fname, "private")
                    st.session_state._preview_toast_doc = None
            with col_del:
                if st.button("🗑️", key=f"del_p_{fname}", help=f"删除 {fname}", disabled=is_chatting):
                    if st.session_state.get("_is_chatting", False):
                        st.stop()
                    with st.spinner("删除中..."):
                        result = delete_document_api(fname, visibility="private")
                    if "chunks_deleted" in result:
                        st.success(f"已删除 {result['chunks_deleted']} 个片段")
                    else:
                        st.error(result.get("message", "删除失败"))
                    st.rerun()
    else:
        st.caption("暂无文档")

    st.markdown("---")

    # 我的共享文档
    st.markdown("### 🌐 我的共享文档")
    shared_docs = get_document_list(visibility="public")
    if shared_docs:
        for doc in shared_docs:
            fname = doc.get("original_filename", "未知")
            chunk_count = doc.get("chunk_count", 0)
            col_name, col_view, col_del = st.columns([3, 1, 1])
            with col_name:
                st.caption(f"📄 {fname} ({chunk_count}段)")
            with col_view:
                if st.button("👁️", key=f"view_s_{fname}", help=f"查看 {fname}", disabled=is_chatting):
                    if st.session_state.get("_is_chatting", False):
                        st.stop()
                    st.session_state._sidebar_preview_doc = (fname, "public")
                    st.session_state._preview_toast_doc = None
            with col_del:
                if st.button("🗑️", key=f"del_s_{fname}", help=f"删除 {fname}", disabled=is_chatting):
                    if st.session_state.get("_is_chatting", False):
                        st.stop()
                    with st.spinner("删除中..."):
                        result = delete_document_api(fname, visibility="public")
                    if "chunks_deleted" in result:
                        st.success(f"已删除 {result['chunks_deleted']} 个片段")
                    else:
                        st.error(result.get("message", "删除失败"))
                    st.rerun()
    else:
        st.caption("暂无共享文档")

    # 侧边栏内联文档预览
    if st.session_state.get("_sidebar_preview_doc") and not is_chatting:
        preview_info = st.session_state._sidebar_preview_doc
        # 兼容旧格式（纯字符串）和新格式（(filename, visibility) 元组）
        if isinstance(preview_info, tuple):
            doc_name, doc_visibility = preview_info
        else:
            doc_name, doc_visibility = preview_info, ""

        st.markdown("---")
        # 只在首次设置预览文档时显示 toast，避免每次重跑都弹出
        if st.session_state.get("_preview_toast_doc") != preview_info:
            st.session_state._preview_toast_doc = preview_info
            vis_label = "个人" if doc_visibility == "private" else "共享" if doc_visibility == "public" else ""
            st.toast(f"📖 文档预览已加载：{doc_name}" + (f"（{vis_label}）" if vis_label else ""), icon="📄")
        st.markdown(f"### 📖 {doc_name}")

        # 关闭按钮在顶部
        if st.button("✖ 关闭预览", key="close_sidebar_preview"):
            if st.session_state.get("_is_chatting", False):
                st.stop()
            st.session_state._sidebar_preview_doc = None
            st.session_state._preview_toast_doc = None
            st.rerun()

        st.markdown("---")

        doc_content = get_document_content(doc_name, visibility=doc_visibility)
        if doc_content and doc_content.get("chunks"):
            for chunk in doc_content["chunks"]:
                st.caption(f"**片段 {chunk['index']}**")
                st.text(chunk["content"][:300])
                st.divider()
        else:
            st.warning("无法获取文档内容")

    st.markdown("---")

    # 操作按钮
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ 清空对话", use_container_width=True, disabled=is_chatting):
            if st.session_state.get("_is_chatting", False):
                st.stop()
            clear_conversation()
    with col2:
        if st.button("🔄 刷新", use_container_width=True, disabled=is_chatting):
            if st.session_state.get("_is_chatting", False):
                st.stop()
            st.rerun()

    st.markdown("---")

    # 清空知识库（带确认）
    st.markdown("### ⚠️ 危险操作")
    if st.button("🗑️ 清空我上传的知识文件", use_container_width=True, type="secondary", disabled=is_chatting):
        if st.session_state.get("_is_chatting", False):
            st.stop()
        st.session_state._confirm_clear = True

    if st.session_state.get("_confirm_clear") and not is_chatting:
        st.warning("⚠️ 确认清空？这将删除你的所有文档（个人+共享文件）！")
        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button("✅ 确认清空", use_container_width=True):
                if st.session_state.get("_is_chatting", False):
                    st.stop()
                clear_my_documents()
                st.session_state._confirm_clear = False
                st.rerun()
        with col_no:
            if st.button("❌ 取消", use_container_width=True):
                if st.session_state.get("_is_chatting", False):
                    st.stop()
                st.session_state._confirm_clear = False
                st.rerun()

    st.markdown("---")

    # 退出登录
    if st.button("🚪 退出登录", use_container_width=True, disabled=is_chatting):
        if st.session_state.get("_is_chatting", False):
            st.stop()
        st.session_state.token = ""
        st.session_state.user_id = ""
        st.session_state.messages = []
        st.session_state._health_retry = 0
        st.rerun()


# ==================== 聊天界面 ====================

def _render_source_buttons(sources_data: list, chunks_data: list, msg_idx: int = 0, disabled: bool = False):
    """渲染来源按钮（每条消息独立预览状态）"""
    if not sources_data:
        return
    st.markdown("---")
    st.markdown("📚 **参考来源：**")

    # 每条消息独立的预览状态
    preview_key = f"_preview_{msg_idx}"
    active_doc = st.session_state.get(preview_key, "")

    for src in sources_data:
        fname = src["filename"]
        chunk_count = src.get("chunk_count", 0)
        is_active = (active_doc == fname)
        label = f"📄 {fname}（{chunk_count}个相关片段）{' ✅' if is_active else ''}"
        if st.button(label, key=f"src_{msg_idx}_{fname}", disabled=disabled):
            if is_active:
                st.session_state[preview_key] = ""
            else:
                st.session_state[preview_key] = fname


def _render_inline_preview(msg_idx: int, chunks_data: list = None, disabled: bool = False):
    """渲染内联预览（每条消息独立）"""
    preview_key = f"_preview_{msg_idx}"
    doc_name = st.session_state.get(preview_key, "")
    if not doc_name or not chunks_data:
        return

    # 从 chunks_data 中筛选当前文件的片段
    preview_chunks = [c for c in chunks_data if c.get("filename") == doc_name]
    if not preview_chunks:
        return

    with st.expander(f"📖 参考片段：{doc_name}", expanded=True):
        # 关闭按钮在顶部
        if st.button("✖ 关闭预览", key=f"close_inline_{msg_idx}", disabled=disabled):
            st.session_state[preview_key] = ""
            st.rerun()

        st.markdown("---")

        for i, chunk in enumerate(preview_chunks):
            actual_idx = chunk.get("index", i + 1)
            st.markdown(f"**片段 {actual_idx}**")
            st.info(chunk.get("content", ""))
            if i < len(preview_chunks) - 1:
                st.divider()


def render_chat_interface():
    """渲染聊天界面（消息历史在上，输入框在下）"""
    # 自动修复异常中断导致的状态卡死：
    # 如果 _is_chatting=True 但没有待处理的 prompt，说明上次对话被刷新/关闭中断
    if st.session_state.get("_is_chatting", False) and not st.session_state.get("_pending_prompt"):
        st.session_state._is_chatting = False

    st.markdown('<h1 class="main-header">🤖 企业智能知识库Agent</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">基于 LangGraph + RAG 混合检索的企业知识问答系统</p>', unsafe_allow_html=True)

    # 清空对话成功提示（由 clear_conversation 设置，显示一次后清除）
    if st.session_state.pop("_clear_done", False):
        st.success("对话已清空")

    # ① 处理待输入：追加用户消息到历史（不渲染，由步骤②统一渲染）
    streaming_ctx = None
    if st.session_state.get("_pending_prompt"):
        prompt = st.session_state._pending_prompt
        st.session_state._pending_prompt = ""
        streaming_ctx = handle_user_input(prompt)

    # ② 显示历史消息（包含刚追加的用户消息）
    is_chatting = st.session_state.get("_is_chatting", False)
    for idx, message in enumerate(st.session_state.messages):
        role = message.get("role", "user")
        with st.chat_message(role):
            st.markdown(message.get("content", ""))
            if role == "assistant" and message.get("sources"):
                _render_source_buttons(
                    message["sources"],
                    message.get("chunks", []),
                    msg_idx=idx,
                    disabled=is_chatting
                )
        if role == "assistant" and message.get("sources"):
            _render_inline_preview(msg_idx=idx, chunks_data=message.get("chunks", []), disabled=is_chatting)

    # ③ 聊天输入框（始终可用，支持流式期间继续输入）
    prompt = st.chat_input("输入您的问题...", disabled=False)
    if prompt:
        st.session_state._is_chatting = True
        st.session_state._pending_prompt = prompt
        st.rerun()

    # ④ 创建流式占位符并执行流式响应
    if streaming_ctx is not None:
        placeholder = st.empty()
        streaming_ctx["placeholder"] = placeholder
        _execute_streaming(streaming_ctx)


def handle_user_input(prompt: str):
    """处理用户输入：追加用户消息到历史，返回流式上下文供调用方执行"""
    # 在主线程中捕获 token 和 web_search，传给后台线程，避免线程访问 session_state
    token = ""
    try:
        token = st.session_state.token
    except (AttributeError, KeyError):
        try:
            token = st.session_state["token"]
        except (KeyError, TypeError):
            token = ""

    web_search = st.session_state.get("web_search", False)

    # 追加用户消息到历史
    st.session_state.messages.append({"role": "user", "content": prompt})

    return {
        "token": token,
        "prompt": prompt,
        "web_search": web_search,
    }


def _execute_streaming(ctx: dict):
    """执行流式响应（带超时保护 + 工具调用状态显示 + 异常安全重置）

    使用 try...finally 确保 _is_chatting 在任何退出路径（正常结束、超时、异常）都被重置，
    防止侧边栏按钮因异常而永久禁用。
    """
    placeholder = ctx["placeholder"]
    token = ctx["token"]
    prompt = ctx["prompt"]
    web_search = ctx.get("web_search", False)

    sources_data = []
    chunks_data = []
    full_response = ""
    tool_calls_display = []
    eval_data = None

    q = queue.Queue()

    def _stream_worker():
        try:
            for item in send_message_stream(prompt, token=token, web_search=web_search):
                q.put(item)
        except Exception as e:
            q.put({"type": "chunk", "data": f"❌ 请求失败: {e}"})
        q.put(None)

    threading.Thread(target=_stream_worker, daemon=True).start()

    STREAM_TIMEOUT = 120
    start_time = time.time()

    try:
        while True:
            try:
                item = q.get(timeout=1.0)
            except queue.Empty:
                if time.time() - start_time > STREAM_TIMEOUT:
                    full_response += "\n\n⚠️ 响应超时，请重试"
                    break
                continue

            if item is None:
                break

            start_time = time.time()

            if isinstance(item, dict):
                event_type = item.get("type", "")

                if event_type == "sources":
                    sources_data = item["data"].get("sources", [])
                    chunks_data = item["data"].get("chunks", [])

                elif event_type == "evaluation":
                    eval_data = item["data"]

                elif event_type == "tool_call":
                    tool_name = item["data"].get("tool", "未知工具")
                    tool_calls_display.append({"tool": tool_name, "status": "calling"})
                    status_text = _build_tool_status_text(tool_calls_display)
                    placeholder.markdown(status_text)

                elif event_type == "tool_result":
                    tool_name = item["data"].get("tool", "未知工具")
                    for tc in tool_calls_display:
                        if tc["tool"] == tool_name and tc["status"] == "calling":
                            tc["status"] = "done"
                            break
                    status_text = _build_tool_status_text(tool_calls_display)
                    placeholder.markdown(status_text)

                elif event_type == "chunk":
                    full_response += item["data"]
                    if tool_calls_display:
                        status_text = _build_tool_status_text(tool_calls_display)
                        placeholder.markdown(status_text + "\n\n" + full_response + "▌")
                    else:
                        placeholder.markdown(full_response + "▌")
            else:
                full_response += str(item)
                placeholder.markdown(full_response + "▌")

    except Exception as e:
        # 捕获任何未预期的异常，避免状态卡死
        full_response += f"\n\n❌ 发生错误: {e}"

    finally:
        # 无论发生什么都执行：最终渲染 + 保存消息 + 重置状态
        try:
            if tool_calls_display and full_response:
                status_text = _build_tool_status_text(tool_calls_display)
                placeholder.markdown(status_text + "\n\n" + full_response)
            else:
                placeholder.markdown(full_response)

            _render_source_buttons(sources_data, chunks_data, msg_idx=len(st.session_state.messages), disabled=True)

            # 显示质量评估结果
            if eval_data:
                _render_evaluation(eval_data)

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": sources_data,
                "chunks": chunks_data,
                "evaluation": eval_data,
            })
        except Exception:
            pass  # 渲染失败不应阻止状态重置

        # 无条件重置聊天状态
        st.session_state._is_chatting = False
        st.rerun()


# 工具名称中文映射
_TOOL_NAME_MAP = {
    "search_web": "🌐 搜索网络",
}


def _render_evaluation(eval_data: dict):
    """渲染质量评估结果"""
    faith = eval_data.get("faithfulness", {})
    relev = eval_data.get("relevancy", {})
    overall = eval_data.get("overall_passed", True)
    strategy = eval_data.get("degradation_strategy", "none")

    faith_score = faith.get("score", 1.0)
    relev_score = relev.get("score", 1.0)

    # 评分颜色
    def score_color(score, threshold):
        if score >= threshold:
            return "🟢"
        elif score >= threshold - 0.1:
            return "🟡"
        return "🔴"

    with st.expander("📊 回答质量评估", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("忠实度", f"{faith_score:.0%}",
                      help="回答中被检索文档支持的声明比例（阈值≥90%）")
        with col2:
            st.metric("答案相关性", f"{relev_score:.0%}",
                      help="回答与用户问题的语义相关度（阈值≥85%）")

        if not overall:
            st.warning(f"⚠️ 质量未达标 | 降级策略: {strategy}")
            if strategy == "fallback" and eval_data.get("fallback_content"):
                st.info(f"📄 Top-1 原文片段：\n\n{eval_data['fallback_content']}")


def _build_tool_status_text(tool_calls: list) -> str:
    """构建工具调用状态的 Markdown 文本"""
    lines = []
    for tc in tool_calls:
        tool_display = _TOOL_NAME_MAP.get(tc["tool"], f"🔧 {tc['tool']}")
        if tc["status"] == "calling":
            lines.append(f"⏳ {tool_display} ...")
        else:
            lines.append(f"✅ {tool_display}")
    return "\n".join(lines)


# ==================== 就绪检查 ====================

@st.cache_data(ttl=3)
def check_backend_ready() -> bool:
    """检查后端是否已加载完核心模块（缓存 3 秒，快速感知就绪状态）"""
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if resp.status_code == 200:
            return resp.json().get("ready", False)
        return False
    except Exception:
        return False


# ==================== 主函数 ====================

def main():
    """主函数"""
    init_session_state()

    # 检查后端是否就绪（最多等待 30 秒，避免死循环）
    if not check_backend_ready():
        retry_count = st.session_state.get("_health_retry", 0)
        if retry_count < 15:  # 最多重试 15 次 × 2 秒 = 30 秒
            st.session_state._health_retry = retry_count + 1
            st.info(f"⏳ 后端核心模块加载中，请稍候... ({retry_count + 1}/15)")
            time.sleep(2)
            st.rerun()
            return
        else:
            st.error("❌ 后端服务连接超时，请确认后端已启动（`python -m backend.main`）")
            if st.button("🔄 重新连接"):
                st.session_state._health_retry = 0
                st.rerun()
            return

    # 后端就绪，重置重试计数
    st.session_state._health_retry = 0

    # 检查是否已登录
    if not st.session_state.token:
        render_login_page()
        return

    render_sidebar()
    render_chat_interface()


if __name__ == "__main__":
    main()
