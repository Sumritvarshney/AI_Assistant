import streamlit as st
import pandas as pd
import io
import json
import base64
import uuid
import os
from datetime import datetime
from pathlib import Path

# Import the updated streaming function and user management functions
from chatbot.agent2 import (
    run_chat_stream, user_exists, save_user_credentials, 
    load_user_credentials, get_user_threads, delete_user_thread_by_id,
    get_user_threads_with_messages, create_user_thread,
    update_chat_metadata, get_user_chat_history, save_chat_messages,
    fetch_and_save_user_profile
)

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="Spog.ai Intelligence", 
    page_icon="2.png", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Default API Base URL (can be overridden per user)
DEFAULT_API_BASE_URL = "https://api.spog.ai"

# Custom CSS for better chat UI (From your earlier design)
st.markdown("""
<style>
    /* Main background */
    .main {
        background-color: #f9f9f9;
    }
    
    /* Login page styling */
    .login-container {
        max-width: 500px;
        margin: 0 auto;
        padding: 40px 20px;
        text-align: center;
    }
    
    .login-header {
        margin-bottom: 30px;
    }
    
    .login-form {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Configuration panel styling */
    .config-status {
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    
    .config-status.success {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 5px solid #1a237e;
    }
    
    /* Header styling */
    .header-container {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 20px;
        padding: 10px;
        background: linear-gradient(90deg, #f9f9f9 0%, #ffffff 100%);
        border-radius: 10px;
    }
    
    /* --- CUSTOM CHAT BUBBLE STYLING --- */
    
    /* Hide default Streamlit chat elements when using custom */
    .stChatMessage {
        display: none !important;
    }
    
    /* Message content styling */
    .chat-message {
        padding: 15px 20px;
        border-radius: 18px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        max-width: 70%;
        word-wrap: break-word;
    }
    
    /* User message container - right aligned */
    .user-message-container {
        display: flex;
        justify-content: flex-end;
        margin: 10px 0;
    }
    
    /* User message bubble - right aligned */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 18px 18px 4px 18px;
        padding: 12px 18px;
        max-width: 80%;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.2);
        text-align: left;
        margin-left: auto;
    }
    
    /* Assistant message container - left aligned */
    .assistant-message-container {
        display: flex;
        justify-content: flex-start;
        margin: 10px 0;
    }
    
    /* Assistant message bubble - left aligned */
    .assistant-message {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 18px 18px 18px 4px;
        padding: 12px 18px;
        max-width: 80%;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        text-align: left;
    }
    
    /* Avatar styling */
    .user-avatar {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 50%;
        width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 16px;
        margin-left: 8px;
        flex-shrink: 0;
    }
    
    .assistant-avatar {
        background: #1a237e;
        border-radius: 50%;
        width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 16px;
        margin-right: 8px;
        flex-shrink: 0;
    }
    
    /* Message row layout */
    .message-row {
        display: flex;
        align-items: flex-end;
        gap: 8px;
        width: 100%;
    }
    
    .user-row {
        flex-direction: row-reverse;
    }
    
    .assistant-row {
        flex-direction: row;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. HELPER FUNCTIONS ---

def get_base64_image(image_path):
    try:
        if os.path.exists(image_path):
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
        return ""
    except Exception:
        return ""

def get_query_params():
    """Get query parameters from URL using the newer API"""
    query_params = st.query_params
    return {
        "user": query_params.get("user", [None])[0] if isinstance(query_params.get("user"), list) else query_params.get("user"),
        "token": query_params.get("token", [None])[0] if isinstance(query_params.get("token"), list) else query_params.get("token")
    }

def set_query_params(user_email=None, token=None):
    """Set query parameters in URL using the newer API"""
    params = {}
    if user_email: params["user"] = user_email
    if token: params["token"] = token
    st.query_params.update(params)

def clear_query_params():
    """Clear all query parameters"""
    st.query_params.clear()

def initialize_session_state():
    """Initialize all session state variables"""
    try:
        query_params = get_query_params()
    except:
        query_params = {"user": None, "token": None}
    
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "user_email" not in st.session_state:
        st.session_state.user_email = query_params.get("user")
    if "auth_token" not in st.session_state:
        st.session_state.auth_token = query_params.get("token")
    if "chats" not in st.session_state:
        st.session_state.chats = {}
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = None
    if "quick_query" not in st.session_state:
        st.session_state.quick_query = None
    if "chat_name_input" not in st.session_state:
        st.session_state.chat_name_input = ""

def load_user_chats(email):
    """Load all chats for a user and restore their messages"""
    threads_data = get_user_threads_with_messages(email)
    st.session_state.chats = threads_data
    
    if threads_data:
        sorted_chats = sorted(
            threads_data.items(), 
            key=lambda x: x[1].get("created", ""), 
            reverse=True
        )
        st.session_state.current_chat_id = sorted_chats[0][0]
        st.session_state.chat_name_input = sorted_chats[0][1]["name"]
    else:
        create_new_chat()

def create_new_chat():
    """Create a new chat for current user"""
    if not st.session_state.logged_in:
        return None
    
    chat_count = len(st.session_state.chats) + 1
    chat_name = f"Chat {chat_count}"
    
    new_chat_id = create_user_thread(st.session_state.user_email, name=chat_name)
    initial_messages = get_user_chat_history(st.session_state.user_email, new_chat_id)
    
    chat_data = {
        "name": chat_name,
        "messages": initial_messages,
        "created": datetime.now().isoformat(),
        "updated": datetime.now().isoformat()
    }
    
    st.session_state.chats[new_chat_id] = chat_data
    st.session_state.current_chat_id = new_chat_id
    st.session_state.chat_name_input = chat_name
    
    return new_chat_id

def delete_chat(chat_id):
    """Delete a chat thread for current user"""
    if not st.session_state.logged_in: return False
    
    if chat_id in st.session_state.chats and len(st.session_state.chats) > 1:
        delete_user_thread_by_id(st.session_state.user_email, chat_id)
        del st.session_state.chats[chat_id]
        
        if st.session_state.current_chat_id == chat_id:
            st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]
            st.session_state.chat_name_input = st.session_state.chats[st.session_state.current_chat_id]["name"]
        return True
    return False

def rename_chat(chat_id, new_name):
    """Rename a chat thread"""
    if chat_id in st.session_state.chats and new_name.strip():
        st.session_state.chats[chat_id]["name"] = new_name.strip()
        update_chat_metadata(st.session_state.user_email, chat_id, {"name": new_name.strip()})
        return True
    return False

def switch_chat(chat_id):
    """Switch to a different chat"""
    if chat_id != st.session_state.current_chat_id and chat_id in st.session_state.chats:
        st.session_state.current_chat_id = chat_id
        st.session_state.chat_name_input = st.session_state.chats[chat_id]["name"]
        return True
    return False

def add_message_to_chat(chat_id, role, content):
    """Add a message to a chat"""
    if chat_id in st.session_state.chats:
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.chats[chat_id]["messages"].append(message)
        save_chat_messages(st.session_state.user_email, chat_id, st.session_state.chats[chat_id]["messages"])
        update_chat_metadata(
            st.session_state.user_email,
            chat_id,
            {
                "updated": datetime.now().isoformat(),
                "last_message": content[:50] if len(content) > 50 else content
            }
        )
        return True
    return False

def login(email, spog_token, api_base_url):
    """Login user and set session state"""
    api_url = api_base_url if api_base_url else DEFAULT_API_BASE_URL
    
    try:
        fetch_and_save_user_profile(email, spog_token, api_url)
    except Exception as e:
        print(f"Profile fetch failed, saving basic credentials: {e}")
        save_user_credentials(email, spog_token, api_url)
    
    st.session_state.logged_in = True
    st.session_state.user_email = email
    st.session_state.auth_token = spog_token
    
    set_query_params(user_email=email, token=spog_token[:8])
    load_user_chats(email)
    return True

def logout():
    """Logout current user"""
    st.session_state.logged_in = False
    st.session_state.user_email = None
    st.session_state.auth_token = None
    st.session_state.chats = {}
    st.session_state.current_chat_id = None
    st.session_state.chat_name_input = ""
    clear_query_params()
    st.rerun()

def validate_existing_login():
    """Validate if the login from URL params is still valid"""
    if st.session_state.user_email and not st.session_state.logged_in:
        if user_exists(st.session_state.user_email):
            creds = load_user_credentials(st.session_state.user_email)
            if creds:
                st.session_state.logged_in = True
                load_user_chats(st.session_state.user_email)
                return True
    return False

# --- 3. LOGIN PAGE ---

def login_page():
    img_b64 = get_base64_image("1.png")
    img_html = f'<img src="data:image/png;base64,{img_b64}" width="120">' if img_b64 else '<h2>🤖</h2>'
    
    st.markdown(f"""
        <div class="login-container">
            <div class="login-header">
                {img_html}
                <h1>Spog.AI Operational Intelligence</h1>
                <p style="color: #666; font-size: 1.1em;">Please login with your credentials to continue</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.form("login_form"):
            st.markdown("### 🔐 Login")
            
            email = st.text_input(
                "Email",
                placeholder="your.email@company.com",
                value=st.session_state.user_email if st.session_state.user_email else "",
            )
            
            spog_token = st.text_input(
                "SPOG Token",
                type="password",
                placeholder="Enter your SPOG token",
            )
            
            api_base_url = st.text_input(
                "API Base URL (optional)",
                placeholder=DEFAULT_API_BASE_URL,
            )
            
            stay_logged_in = st.checkbox("Stay logged in", value=True)
            submitted = st.form_submit_button("Login", type="primary", use_container_width=True)
            
            if submitted:
                if email and spog_token:
                    if login(email, spog_token, api_base_url):
                        st.rerun()
                else:
                    st.error("Please enter both email and SPOG token")
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.9em;">
            Returning user? Your existing chats will be loaded automatically.<br>
            New user? Just enter your email and token above - we'll create your account automatically.
        </div>
        """, unsafe_allow_html=True)

# --- 4. MAIN CHAT INTERFACE ---

def main_chat_interface():
    # Sidebar
    with st.sidebar:
        st.image("1.png", width=150)
        st.title("Spog.AI Pilot")
        st.markdown(f"👤 **User:** {st.session_state.user_email}")
        
        if st.button("🚪 Logout", use_container_width=True):
            logout()
        
        st.markdown("---")
        
        creds = load_user_credentials(st.session_state.user_email)
        if creds:
            st.markdown(f"""
            <div class="config-status success">
                ✅ Connected to: {creds.get('api_base_url', DEFAULT_API_BASE_URL)}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 💬 Chat Threads")
        
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            create_new_chat()
            st.rerun()
        
        if st.session_state.current_chat_id and st.session_state.current_chat_id in st.session_state.chats:
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                st.text_input(
                    "Chat Name",
                    value=st.session_state.chat_name_input,
                    key=f"chat_rename_{st.session_state.current_chat_id}",
                    label_visibility="collapsed",
                    placeholder="Chat name...",
                    on_change=lambda: rename_chat(
                        st.session_state.current_chat_id, 
                        st.session_state[f"chat_rename_{st.session_state.current_chat_id}"]
                    )
                )
            with col2:
                st.markdown(" ")
                st.caption(f"ID: {st.session_state.current_chat_id[:4]}...")
        
        st.markdown("### All Chats")
        for chat_id, chat_info in st.session_state.chats.items():
            col1, col2, col3 = st.columns([0.7, 0.15, 0.15])
            with col1:
                msg_count = len(chat_info.get("messages", []))
                if st.button(
                    f"💬 {chat_info['name']}",
                    key=f"chat_{chat_id}",
                    use_container_width=True,
                    help=f"Messages: {msg_count}"
                ):
                    if switch_chat(chat_id):
                        st.rerun()
            with col2:
                if st.button("✏️", key=f"edit_{chat_id}", help="Rename"):
                    st.session_state.chat_name_input = chat_info['name']
                    st.rerun()
            with col3:
                if len(st.session_state.chats) > 1:
                    if st.button("🗑️", key=f"delete_{chat_id}", help="Delete chat"):
                        if delete_chat(chat_id):
                            st.rerun()
            
            if chat_info.get("messages") and len(chat_info["messages"]) > 0:
                last_msg = chat_info["messages"][-1]["content"]
                preview = last_msg[:30] + "..." if len(last_msg) > 30 else last_msg
                st.caption(f"  {preview}")
        
        st.markdown("---")
        st.markdown("### ⚡ Quick Actions")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 All Tickets", use_container_width=True):
                st.session_state.quick_query = "List all tickets"
        with col2:
            if st.button("👤 Find User", use_container_width=True):
                st.session_state.quick_query = "Show me the tenant for Janit"
        col3, col4 = st.columns(2)
        with col3:
            if st.button("☁️ Services", use_container_width=True):
                st.session_state.quick_query = "List all services"
        with col4:
            if st.button("📊 Analytics", use_container_width=True):
                st.session_state.quick_query = "Show me analytics dashboard"
        
        st.markdown("---")
        st.markdown("**Session Metrics**")
        total_messages = sum(len(c.get("messages", [])) for c in st.session_state.chats.values())
        st.metric("Active Chats", len(st.session_state.chats))
        st.metric("Total Messages", total_messages)

    # Main chat area header
    img_b64 = get_base64_image("2.png")
    img_html = f'<img src="data:image/png;base64,{img_b64}" width="48">' if img_b64 else '<h2>🤖</h2>'
    st.markdown(
        f"""
        <div class="header-container">
            {img_html}
            <h1>Spog.AI Operational Intelligence</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("ASK about **Infrastructure, Users, Tickets, or Assets**.")
    
    current_chat = st.session_state.chats.get(st.session_state.current_chat_id, {"messages": []})
    
    if st.session_state.quick_query:
        user_input = st.session_state.quick_query
        st.session_state.quick_query = None
    else:
        user_input = st.chat_input("Ex: 'Show me all high priority tickets'")
    
    # --- RENDER EXISTING CHAT HISTORY USING CUSTOM CSS ---
    for msg in current_chat.get("messages", []):
        if msg["role"] == "user":
            st.markdown(f"""
                <div class="user-message-container">
                    <div class="message-row user-row">
                        <div class="user-message">{msg['content']}</div>
                        <div class="user-avatar">👤</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="assistant-message-container">
                    <div class="message-row assistant-row">
                        <div class="assistant-avatar">🤖</div>
                        <div class="assistant-message">{msg['content']}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    # Process new user input
    if user_input:
        # Add to state and save
        add_message_to_chat(st.session_state.current_chat_id, "user", user_input)
        
        # Render the user message immediately using custom CSS
        st.markdown(f"""
            <div class="user-message-container">
                <div class="message-row user-row">
                    <div class="user-message">{user_input}</div>
                    <div class="user-avatar">👤</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Generate and stream Bot Response
        status_container = st.empty()
        response_placeholder = st.empty()
        raw_response = ""
        table_data = [] # 🔴 Capture the raw dataframe JSON
        
        try:
            with status_container.status("🔄 Processing request...", expanded=True) as status_ui:
                for chunk in run_chat_stream(
                    user_input,
                    user_email=st.session_state.user_email,
                    thread_id=st.session_state.current_chat_id,
                    persist_memory=True
                ):
                    chunk_data = json.loads(chunk)
                    
                    # 🔴 Handle the new high-speed streaming chunk!
                    if chunk_data["status"] == "streaming":
                        response_placeholder.markdown(f"""
                            <div class="assistant-message-container">
                                <div class="message-row assistant-row">
                                    <div class="assistant-avatar">🤖</div>
                                    <div class="assistant-message">{chunk_data["chunk"]}▌</div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                    elif chunk_data["status"] != "done":
                        status_ui.update(label=chunk_data["status"], state="running")
                    else:
                        raw_response = chunk_data["response"]
                        table_data = chunk_data.get("table_data", []) # 🔴 Grab Native DF
                        status_ui.update(label="✅ Complete!", state="complete", expanded=False)
            
            # Clear the loading status
            status_container.empty()
            
            # Final stream render
            response_placeholder.markdown(f"""
                <div class="assistant-message-container">
                    <div class="message-row assistant-row">
                        <div class="assistant-avatar">🤖</div>
                        <div class="assistant-message">{raw_response}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # 🔴 Render the Native Streamlit Interactive Table if data exists!
            # 🔴 Render the Native Streamlit Interactive Table if data exists!
            if table_data:
                # 1. Convert to DataFrame
                df = pd.DataFrame(table_data)
                
                # 2. Force all columns to strings to prevent PyArrow mixed-type crashes
                df = df.astype(str) 
                
                # 3. Render using the new syntax to fix the deprecation warning
                st.dataframe(df, width='stretch')
            # Save the final message to history
            add_message_to_chat(st.session_state.current_chat_id, "assistant", raw_response)
            
            if raw_response:
                st.download_button(
                    label="📥 Download Response",
                    data=raw_response,
                    file_name=f'response_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt',
                    mime='text/plain',
                    key=f"download_{len(current_chat['messages'])}"
                )
                
        except Exception as e:
            status_container.empty()
            st.error(f"❌ Error: {str(e)}")
            error_msg = f"I encountered an error: {str(e)}"
            add_message_to_chat(st.session_state.current_chat_id, "assistant", error_msg)
            response_placeholder.markdown(f"""
                <div class="assistant-message-container">
                    <div class="message-row assistant-row">
                        <div class="assistant-avatar">🤖</div>
                        <div class="assistant-message">{error_msg}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

# --- 5. MAIN APP ---

def main():
    initialize_session_state()
    
    if not st.session_state.logged_in:
        if validate_existing_login():
            st.rerun()
    
    if not st.session_state.logged_in:
        login_page()
    else:
        main_chat_interface()

if __name__ == "__main__":
    main()