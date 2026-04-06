import os
import json
import re
import pickle
import time
import httpx
import faiss
import concurrent.futures
import uuid
import urllib.parse
import base64
from typing import List, Dict, Any, TypedDict, Annotated, Literal, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END, add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from sentence_transformers import SentenceTransformer
from thefuzz import fuzz
from datetime import datetime, timezone
from pathlib import Path
from celery_app import send_response_email # type: ignore
import hashlib


# --- NEW MONGODB IMPORTS ---
from pymongo import MongoClient
from langgraph.checkpoint.mongodb import MongoDBSaver

# --- 1. CONFIGURATION ---

LLAMA_URL      = os.environ.get("LLAMA_URL")
METADATA_FILE = "apis.json"
INDEX_FILE = "apis.faiss"
INDEX_META_FILE = "apis_metadata.pkl"

# Base directory for all user data (static assets like creds and json history)
USERS_BASE_DIR = os.environ.get("USERS_BASE_DIR", "data/users")
os.makedirs(USERS_BASE_DIR, exist_ok=True)

# How many recent messages to include for context
CONVERSATION_CONTEXT_WINDOW = 6

# --- MONGODB SETUP ---
MONGO_URI    = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
mongo_client = MongoClient(MONGO_URI)

checkpointer = MongoDBSaver(mongo_client, db_name="spog_ai_agent")

# --- USER MANAGEMENT FUNCTIONS ---

def get_user_id(email: str) -> str:
    """Generate a unique user ID from email"""
    return hashlib.sha256(email.lower().encode()).hexdigest()[:16]

def get_user_paths(email: str):
    """Get all paths for a specific user"""
    user_id = get_user_id(email)
    base = Path(USERS_BASE_DIR) / user_id
    return {
        "user_id": user_id,
        "base": base,
        "credentials": base / "credentials.json",
        "chat_metadata": base / "chat_metadata",
        "chat_messages": base / "chat_messages",
    }

def save_user_credentials(email: str, spog_token: str, api_base_url: str = None):
    """Save credentials for a user"""
    paths = get_user_paths(email)
    paths["base"].mkdir(parents=True, exist_ok=True)
    paths["chat_metadata"].mkdir(exist_ok=True)
    paths["chat_messages"].mkdir(exist_ok=True)

    creds = {
        "email": email,
        "spog_token": spog_token,
        "api_base_url": api_base_url,
        "created_at": datetime.now().isoformat(),
        "last_login": datetime.now().isoformat()
    }

    with open(paths["credentials"], 'w') as f:
        json.dump(creds, f, indent=2)

    return paths

def fetch_and_save_user_profile(email: str, spog_token: str, api_base_url: str):
    """Fetch and store user_context directly by decoding the JWT token."""
    user_context = {
        "email": email,
        "name": "Unknown",
        "title": "Unknown",
        "mobile": "Unknown",
        "teams": [],
        "roles": "Unknown",
        "provider": [],
    }

    try:
        if spog_token and "." in spog_token:
            payload_b64 = spog_token.split('.')[1]
            payload_b64 += "=" * ((4 - len(payload_b64) % 4) % 4)
            decoded_payload_str = base64.urlsafe_b64decode(payload_b64).decode('utf-8')
            token_data = json.loads(decoded_payload_str)

            user_context["email"] = token_data.get("email") or email
            user_context["name"] = token_data.get("fullname") or token_data.get("displayName") or "Unknown"
            user_context["title"] = token_data.get("title") or "Unknown"
            user_context["mobile"] = str(token_data.get("mobile")) if token_data.get("mobile") else "Unknown"
            user_context["roles"] = "Super Admin" if token_data.get("is_super_admin") else "User"

            print(f"SUCCESS: Extracted profile for {user_context['name']} directly from token.")
        else:
            print("WARNING: Invalid or missing token provided.")
    except Exception as e:
        print(f"ERROR: Could not decode local token: {e}")

    paths = get_user_paths(email)
    paths["base"].mkdir(parents=True, exist_ok=True)
    paths["chat_metadata"].mkdir(exist_ok=True)
    paths["chat_messages"].mkdir(exist_ok=True)

    creds_file = paths["credentials"]
    if creds_file.exists():
        with open(creds_file, "r") as f:
            creds = json.load(f)
    else:
        creds = {"email": email}

    creds["user_context"] = user_context
    creds["spog_token"] = spog_token
    creds["api_base_url"] = api_base_url

    with open(creds_file, "w") as f:
        json.dump(creds, f, indent=2)

    return user_context

def load_user_credentials(email: str):
    """Load credentials for a user"""
    paths = get_user_paths(email)
    creds_file = paths["credentials"]
    paths["base"].mkdir(parents=True, exist_ok=True)
    paths["chat_metadata"].mkdir(exist_ok=True)
    paths["chat_messages"].mkdir(exist_ok=True)

    if creds_file.exists():
        with open(creds_file, 'r') as f:
            return json.load(f)
    return None

def user_exists(email: str) -> bool:
    """Check if user exists"""
    paths = get_user_paths(email)
    return paths["credentials"].exists()

def get_user_threads(email: str) -> List[str]:
    """Get all chat threads for a user"""
    paths = get_user_paths(email)
    threads = []
    try:
        if paths["chat_metadata"].exists():
            for filename in os.listdir(paths["chat_metadata"]):
                if filename.endswith('.json'):
                    thread_id = filename[:-5]
                    threads.append(thread_id)
    except Exception as e:
        print(f"Error listing threads for user {email}: {e}")
    return threads

def save_chat_messages(user_email: str, thread_id: str, messages: list):
    """Save chat messages for a user's thread"""
    try:
        paths = get_user_paths(user_email)
        messages_file = paths["chat_messages"] / f"{thread_id}.json"
        paths["chat_messages"].mkdir(exist_ok=True)

        with open(messages_file, 'w') as f:
            json.dump(messages, f, default=str, indent=2)
        return True
    except Exception as e:
        print(f"Error saving chat messages: {e}")
        return False

def load_chat_messages(user_email: str, thread_id: str):
    """Load chat messages for a user's thread"""
    try:
        paths = get_user_paths(user_email)
        messages_file = paths["chat_messages"] / f"{thread_id}.json"

        if messages_file.exists():
            with open(messages_file, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"Error loading chat messages: {e}")
        return []

def save_chat_metadata(user_email: str, thread_id: str, metadata: dict):
    """Save chat metadata for a user"""
    try:
        paths = get_user_paths(user_email)
        metadata_file = paths["chat_metadata"] / f"{thread_id}.json"
        paths["chat_metadata"].mkdir(exist_ok=True)

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, default=str, indent=2)
        return True
    except Exception as e:
        print(f"Error saving chat metadata: {e}")
        return False

def load_chat_metadata(user_email: str, thread_id: str):
    """Load chat metadata for a user"""
    try:
        paths = get_user_paths(user_email)
        metadata_file = paths["chat_metadata"] / f"{thread_id}.json"

        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        print(f"Error loading chat metadata: {e}")
        return None

def get_user_threads_with_messages(email: str) -> Dict:
    """Get all chat threads with their messages for a user"""
    paths = get_user_paths(email)
    threads = {}

    try:
        if paths["chat_metadata"].exists():
            for filename in os.listdir(paths["chat_metadata"]):
                if filename.endswith('.json'):
                    thread_id = filename[:-5]
                    metadata_file = paths["chat_metadata"] / filename
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)

                    messages = load_chat_messages(email, thread_id)

                    threads[thread_id] = {
                        "name": metadata.get("name", f"Chat {len(threads) + 1}"),
                        "messages": messages,
                        "created": metadata.get("created", datetime.now().isoformat()),
                        "updated": metadata.get("updated", datetime.now().isoformat())
                    }
    except Exception as e:
        print(f"Error loading threads for user {email}: {e}")

    return threads

def create_user_thread(email: str, thread_id: str = None, name: str = None):
    """Create a new thread for a user"""
    if not thread_id:
        thread_id = str(uuid.uuid4())

    paths = get_user_paths(email)
    paths["chat_metadata"].mkdir(exist_ok=True)
    paths["chat_messages"].mkdir(exist_ok=True)

    welcome_message = generate_welcome_brief(email)

    metadata = {
        "thread_id": thread_id,
        "name": name or f"Chat {len(get_user_threads(email)) + 1}",
        "created": datetime.now().isoformat(),
        "updated": datetime.now().isoformat(),
        "user_email": email,
        "last_message": welcome_message[:50]
    }

    metadata_file = paths["chat_metadata"] / f"{thread_id}.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    initial_chat_history = [
        {
            "role": "assistant",
            "content": welcome_message,
            "timestamp": datetime.now().isoformat()
        }
    ]

    messages_file = paths["chat_messages"] / f"{thread_id}.json"
    with open(messages_file, 'w') as f:
        json.dump(initial_chat_history, f, indent=2)

    return thread_id

def update_chat_metadata(user_email: str, thread_id: str, updates: dict):
    """Update chat metadata for a user"""
    metadata = load_chat_metadata(user_email, thread_id) or {}
    metadata.update(updates)
    metadata["updated"] = datetime.now().isoformat()
    return save_chat_metadata(user_email, thread_id, metadata)

def delete_user_thread(email: str, thread_id: str) -> bool:
    """Delete a chat thread for a user"""
    paths = get_user_paths(email)

    metadata_file = paths["chat_metadata"] / f"{thread_id}.json"
    if metadata_file.exists():
        metadata_file.unlink()

    messages_file = paths["chat_messages"] / f"{thread_id}.json"
    if messages_file.exists():
        messages_file.unlink()

    return True

# --- 2. STATE DEFINITION ---

class AgentState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    intent: str
    selected_api: str
    extracted_filters: Dict
    requested_display_fields: List[str]
    pagination: Dict
    api_data: List[Dict]
    recursion_count: int
    mode: Literal["single_page", "deep_scan"]
    analysis_ready: bool
    needs_refetch: bool
    user_email: str
    user_context: Dict
    table_data: List[Dict]

# --- 3. INTENT TYPE CONSTANTS ---

class IntentType:
    CHITCHAT = "chitchat"
    API_QUERY = "api_query"
    CLARIFICATION = "clarification"
    OUT_OF_SCOPE = "out_of_scope"
    ANALYSIS = "analysis"
    EMAIL_REQUEST = "email_request"

# --- 4. METADATA REGISTRY ---

class MetadataRegistry:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        if not os.path.exists(self.filepath):
            print(f"WARNING: {self.filepath} not found. Using empty registry.")
            return {}
        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)
                print(f"SUCCESS: Loaded {len(data)} API Configs from {self.filepath}")
                return data
        except json.JSONDecodeError as e:
            print(f"CRITICAL: Failed to parse {self.filepath}: {e}")
            return {}

    def get_api_config(self, api_key: str) -> Dict[str, Any]:
        return self.registry.get(api_key, {})

    def get_all_apis(self) -> Dict[str, Any]:
        return self.registry

registry_loader = MetadataRegistry(METADATA_FILE)

# --- 5. PRE-BUILT FAISS ROUTER ---

class PreBuiltFaissRouter:
    def __init__(self, index_path: str, meta_path: str, registry: MetadataRegistry):
        print("DEBUG: Initializing Pre-Built FAISS Router...")
        self.registry = registry
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        if os.path.exists(index_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(index_path)
            with open(meta_path, 'rb') as f:
                self.api_keys_list = pickle.load(f)
            print(f"SUCCESS: Loaded FAISS Index with {self.index.ntotal} vectors.")
        else:
            print("CRITICAL: Index files not found. Please run 'build_index.py' first.")
            self.index = None
            self.api_keys_list = []

    def get_top_candidates(self, query: str, k: int = 15) -> List[Dict]:
        if not self.index:
            return []
        query_vec = self.model.encode([query]).astype('float32')
        distances, indices = self.index.search(query_vec, k)
        candidates = []
        seen = set()
        for idx in indices[0]:
            if idx == -1:
                continue
            if idx < len(self.api_keys_list):
                api_key = self.api_keys_list[idx]
                if api_key not in seen:
                    seen.add(api_key)
                    config = self.registry.get_api_config(api_key)
                    description = config.get("description", "No description available.")
                    candidates.append({"key": api_key, "description": description})
        return candidates

api_router = PreBuiltFaissRouter(INDEX_FILE, INDEX_META_FILE, registry_loader)

# --- 6. HELPERS ---

def _authenticated_client(user_email: str) -> httpx.Client:
    """Create an authenticated client using user-specific credentials"""
    creds = load_user_credentials(user_email)
    if not creds or not creds.get("spog_token"):
        raise ValueError(f"API credentials not set for user {user_email}.")

    headers = {
        "x-token": creds["spog_token"],
        "Content-Type": "application/json",
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0"
    }
    return httpx.Client(headers=headers, timeout=15.0, follow_redirects=True)

def generate_welcome_brief(user_email: str) -> str:
    """Silently fetches user's tickets and generates a personalized greeting."""
    creds = load_user_credentials(user_email)
    if not creds:
        return "Welcome to Spog.ai! How can I help you today?"

    user_context = creds.get("user_context", {})
    full_name = user_context.get("name", "").strip()

    if full_name == "Unknown" or not full_name:
        return "Welcome to Spog.ai! How can I help you today?"

    first_name = full_name.split(" ")[0]

    ticket_api_key = "tickets"
    config = registry_loader.get_api_config(ticket_api_key)

    if not config:
        return f"Hi {first_name}, welcome back! What would you like to look up today?"

    try:
        with _authenticated_client(user_email) as client:
            url = f"{creds.get('api_base_url', '')}{config.get('endpoint', '')}"

            filter_payload = {"assignee": full_name}
            filter_key_name = config.get("filter_param", "filters")
            params = {filter_key_name: json.dumps(filter_payload), "limit": 100}

            r = client.get(url, params=params)
            if r.status_code == 200:
                resp_json = r.json()
                raw_data = resp_json.get("data", []) if isinstance(resp_json, dict) else (resp_json if isinstance(resp_json, list) else [])

                data = generic_filter_tool(raw_data, filter_payload, config)

                if not data:
                    return f"Hi {first_name}, you have no active tickets assigned to you right now. What else can I help you with?"

                high_count = 0
                medium_count = 0
                low_count = 0
                unassigned_count = 0

                for t in data:
                    priority = str(t.get("priority", "")).lower().strip()
                    if priority in ["high", "critical", "p1", "urgent"]:
                        high_count += 1
                    elif priority in ["medium", "p2", "normal"]:
                        medium_count += 1
                    elif priority in ["low", "p3"]:
                        low_count += 1
                    else:
                        unassigned_count += 1

                total = len(data)
                greeting = f"Hi {first_name}, welcome back! I found **{total} tickets** currently assigned to you.\n\n"

                if high_count > 0:
                    greeting += f"🚨 **{high_count} High/Critical** "
                if medium_count > 0:
                    greeting += f"⚠️ **{medium_count} Medium** "
                if low_count > 0:
                    greeting += f"🔽 **{low_count} Low** "
                if unassigned_count > 0:
                    greeting += f"⚪ **{unassigned_count} Uncategorized**"

                greeting += "\n\nWould you like me to pull up the high-priority ones first, or do you need something else?"
                return greeting

    except Exception as e:
        print(f"DEBUG BRIEF: Failed to generate welcome brief: {e}")

    return f"Hi {first_name}, welcome back! How can I assist you today?"

def call_llama(messages: List[Dict[str, str]], max_retries=3) -> str:
    """Calls Llama API with automatic retries and exponential backoff for network resilience."""
    llama_url = LLAMA_URL
    payload = {"messages": messages, "model": "meta-llama/Llama-3.1-8B-Instruct", "temperature": 0.0}

    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=60.0) as client:
                r = client.post(llama_url, json=payload, headers={"Content-Type": "application/json"})

                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]["content"]
                else:
                    print(f"Llama Error {r.status_code}: {r.text}")

        except httpx.TimeoutException:
            print(f"Llama Timeout (Attempt {attempt + 1}/{max_retries}). Retrying...")
            time.sleep(2 ** attempt)

        except Exception as e:
            print(f"Llama Connection Error: {e}")
            break

    print("CRITICAL: Llama API failed after all retries.")
    return "{}"

def get_nested_value(data: Dict, path: str) -> str:
    keys = path.split('.')
    curr = data
    try:
        for k in keys:
            if k.isdigit():
                curr = curr[int(k)]
            else:
                curr = curr[k]
        return str(curr)
    except Exception:
        return ""

def _messages_to_context(messages: List[BaseMessage], max_messages: int = CONVERSATION_CONTEXT_WINDOW) -> str:
    recent = messages[-max_messages:] if len(messages) > max_messages else messages
    parts = []
    for m in recent:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        content = m.content

        if role == "Assistant" and len(content) > 500:
            if "Found" in content[:200] or "|" in content[:200]:
                content = content[:300] + "\n... [DATA TRUNCATED FOR CONTEXT WINDOW] ..."

        parts.append(f"{role}: {content}")
    return "\n".join(parts) if parts else "(no prior messages)"

def get_capabilities_context() -> str:
    apis = registry_loader.get_all_apis()
    if not apis:
        return "I can help you with various business queries."
    api_descriptions = []
    for k, v in apis.items():
        desc = v.get('description', f'{k} data')
        api_descriptions.append(f"{k} ({desc})")
    return f"I can help you query: {', '.join(api_descriptions)}. Just ask naturally!"

def get_api_names_context() -> str:
    apis = registry_loader.get_all_apis()
    if not apis:
        return "No APIs available"
    return ", ".join(apis.keys())

def is_gratitude_or_acknowledgment(message: str) -> bool:
    patterns = [
        r"^th(anks|ank you|x)\.?$",
        r"^(ok|okay|got it|cool|great|awesome|perfect)\.?$",
        r"^(i see|understood|makes sense)\.?$",
        r"^(nice|good|excellent) (job|work|thanks)?\.?$",
        r"^appreciate it\.?$"
    ]
    msg_clean = message.lower().strip()
    return any(re.search(p, msg_clean) for p in patterns)

def detect_email_request(message: str) -> bool:
    """Detect if the user wants the response sent to their email."""
    patterns = [
        r'\b(send|mail|email|forward)\b.{0,30}\b(me|this|response|result|report|it)\b.{0,20}\b(mail|email|inbox)\b',
        r'\b(send|mail|email|forward)\b.{0,20}\b(to my|via|through)\b.{0,10}\b(mail|email)\b',
        r'\bsend (this |it )?(to |via )?(my )?(email|mail|inbox)\b',
        r'\bemail (me|this|the response|the result)\b',
        r'\bmail (me|this|the response|the result)\b',
        r'\bforward (this |it )?to (my )?(email|mail|inbox)\b',
    ]
    msg_lower = message.lower().strip()
    return any(re.search(p, msg_lower) for p in patterns)

def generic_filter_tool(data: List[Dict], filters: Dict, api_config: Dict) -> List[Dict]:
    if not filters:
        return data
    filtered_results = []
    mapping = api_config.get("filter_mapping", {})
    FUZZY_THRESHOLD = 85
    print(f"DEBUG FILTER: Filtering {len(data)} items against {filters}")

    generic_terms = ["tickets", "users", "items", "records", "data", "information", "frameworks", "applications", "software", "lists", "list"]
    cleaned_filters = {}
    for key, value in filters.items():
        if value and isinstance(value, str) and value.lower() not in generic_terms:
            cleaned_filters[key] = value
        elif value and not isinstance(value, str):
            cleaned_filters[key] = value

    if len(cleaned_filters) != len(filters):
        print(f"DEBUG FILTER: Removed generic terms: {set(filters.keys()) - set(cleaned_filters.keys())}")

    for item in data:
        is_match = True
        for key, target_val in cleaned_filters.items():
            if not target_val:
                continue
            target_str = str(target_val).lower().strip()
            possible_paths = mapping.get(key, [key])
            path_match = False

            use_exact_match = key.lower() == "id" or key.lower().endswith("_id")

            for path in possible_paths:
                actual_val = str(get_nested_value(item, path)).lower().strip()

                if actual_val:
                    if target_str == actual_val or target_str in actual_val or actual_val in target_str:
                        path_match = True
                        break

                    if not use_exact_match:
                        score = fuzz.partial_ratio(target_str, actual_val)
                        if score >= FUZZY_THRESHOLD and len(target_str) > 2:
                            print(f"DEBUG FUZZY: '{target_str}' matched '{actual_val}' (Score: {score})")
                            path_match = True
                            break

            if not path_match:
                is_match = False
                break
        if is_match:
            filtered_results.append(item)
    return filtered_results

def _fetch_single_page(client, url, params) -> List[Dict]:
    try:
        r = client.get(url, params=params)
        if r.status_code == 200:
            json_resp = r.json()
            return json_resp.get("data", []) if isinstance(json_resp, dict) else (json_resp if isinstance(json_resp, list) else [])
                                                                                 
        else:
            print(f"API returned status {r.status_code} - {r.url}")
            return []
    except Exception as e:
        print(f"API Network Error: {e}")
        return []

# --- 8. INTENT CLASSIFICATION ---

def classify_intent(state: AgentState) -> Dict:
    messages = state["messages"]
    last_msg = messages[-1].content

    conversation_context = _messages_to_context(messages[:-1], max_messages=4)
    current_api = state.get("selected_api", "none")
    available_apis = get_api_names_context()

    has_data = bool(state.get("api_data", []))
    data_status = "DATA IS CURRENTLY LOADED" if has_data else "NO DATA IS CURRENTLY LOADED"

    classification_prompt = f"""You are the Master Intent Classifier for Spog.ai.
Your job is to read the user's message and categorize it perfectly.

SYSTEM STATE:
- AVAILABLE API DOMAINS: {available_apis}
- CURRENTLY LOADED TOPIC: '{current_api}'
- DATA STATUS: {data_status}

INTENT DEFINITIONS:
1. API_QUERY: The user wants to fetch, list, or look up a dataset. 
2. ANALYSIS: The user is asking a follow-up question about the CURRENTLY LOADED data.
3. CHITCHAT: The user is greeting you, thanking you, OR asking about your capabilities, available topics, or what you can do.
4. CLARIFICATION: The user's message is gibberish, too short to understand, or explicitly asks for clarification.
5. OUT_OF_SCOPE: The user wants to perform an action (create/update/delete) or asks about non-business things (weather, sports, travel).
6. EMAIL_REQUEST: The user wants to send the current response or data to their email.

CRITICAL RULES:
- 🔴 Asking for "my teams", "my details", "my tickets", or "my profile" is a valid API_QUERY. Do NOT mark work-related self-queries as OUT_OF_SCOPE!

CRITICAL TRAINING EXAMPLES:

[Category 1: Direct Data Requests]
User: "show all tickets in our system"
Intent: API_QUERY
User: "fetch me the user list"
Intent: API_QUERY

[Category 2: Follow-ups & Contextual Analysis]
User: "how many of these are still open?"
Intent: ANALYSIS
User: "can you summarize their descriptions?"
Intent: ANALYSIS

[Category 3: Greetings & Capabilities]
User: "hi, good morning"
Intent: CHITCHAT
User: "what topics can you tell me about?"
Intent: CHITCHAT

[Category 4: Out of Scope (Actions & Unrelated Topics)]
User: "book a flight ticket for me from mumbai to delhi"
Intent: OUT_OF_SCOPE
User: "can you delete the last ticket?"
Intent: OUT_OF_SCOPE

[Category 5: Confusion, Vagueness & Typos]
User: "what do you mean by that?"
Intent: CLARIFICATION
User: "asdfghjkl"
Intent: CLARIFICATION

[Category 6: Gratitude]
User: "okay"
Intent: CHITCHAT

[Category 7: Email Requests — ALWAYS this intent]
User: "send this to my email"
Intent: EMAIL_REQUEST
User: "email me this response"
Intent: EMAIL_REQUEST
User: "send this response to my mail"
Intent: EMAIL_REQUEST
User: "mail me the results"
Intent: EMAIL_REQUEST
User: "forward this to my inbox"
Intent: EMAIL_REQUEST
User: "can you send this to me via email"
Intent: EMAIL_REQUEST


CONVERSATION HISTORY:
{conversation_context}

USER MESSAGE: "{last_msg}"

Analyze the USER MESSAGE. Respond with ONLY ONE of the following exact words: 
CHITCHAT | API_QUERY | ANALYSIS | CLARIFICATION | OUT_OF_SCOPE | EMAIL_REQUEST"""

    result = call_llama([{"role": "system", "content": classification_prompt}]).strip().upper()

    intent_map = {
        "CHITCHAT": IntentType.CHITCHAT,
        "API_QUERY": IntentType.API_QUERY,
        "ANALYSIS": IntentType.ANALYSIS,
        "CLARIFICATION": IntentType.CLARIFICATION,
        "OUT_OF_SCOPE": IntentType.OUT_OF_SCOPE,
         "EMAIL_REQUEST":  IntentType.EMAIL_REQUEST
    }

    detected_intent = IntentType.API_QUERY
    for key, val in intent_map.items():
        if key in result:
            detected_intent = val
            break

    print(f"DEBUG INTENT: LLM logically classified '{last_msg}' -> {detected_intent}")

    return {
        "intent": detected_intent,
        "user_email": state.get("user_email"),
        "analysis_ready": False,
        "needs_refetch": False,
        "recursion_count": 0,
    }

# --- 9. CHITCHAT NODE ---

def chitchat_node(state: AgentState) -> Dict:
    messages = state["messages"]
    last_msg = messages[-1].content
    conversation_context = _messages_to_context(messages[:-1], max_messages=6)

    user_email = state.get("user_email")
    user_context = state.get("user_context", {})
    actual_user_name = user_context.get("name", "User")

    is_capabilities_question = any(kw in last_msg.lower() for kw in [
        "what can you do", "help me", "capabilities", "features", "how do you work", "topics"
    ])

    capabilities_info = get_capabilities_context() if is_capabilities_question else ""

    system_prompt = f"""You are a helpful, friendly assistant for Spog.ai.
You are currently talking to: {actual_user_name}

{capabilities_info}

RULES:
- Be warm and conversational.
- ALWAYS remember the user's name is {actual_user_name}. Never confuse them with names found in the data history.
- If user asks what you can do or what topics you have, clearly explain your capabilities based on the info provided above.
- If they just said thanks, respond warmly using their name.
- Keep responses brief.
- Do not make up data.
"""

    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"History:\n{conversation_context}\n\nUser: {last_msg}\n\nResponse:"}
    ]

    response = call_llama(prompt)

    response = re.sub(r'```.*?```', '', response, flags=re.DOTALL)
    response = response.replace("json", "").replace("{", "").replace("}", "").strip()

    if not response:
        response = f"I'm here to help, {actual_user_name.split(' ')[0]}! What would you like to know?"

    return {"messages": [AIMessage(content=response)], "user_email": user_email}

# --- 10. CLARIFICATION NODE ---

def clarification_node(state: AgentState) -> Dict:
    messages = state["messages"]
    last_msg = messages[-1].content
    conversation_context = _messages_to_context(messages[:-1], max_messages=4)
    user_email = state.get("user_email")
    user_context = state.get("user_context", {})
    user_name = user_context.get("name", "User")

    system_prompt = f"""You are a helpful assistant for Spog.ai. 
The user has said something unclear or ambiguous. 
Your job is to ask a clarifying question to understand what they need.

Current user: {user_name}
You have the conversation history below. 
Write a short, friendly message asking for clarification. 
Be polite and helpful, and use the user's name if appropriate.
Do not repeat the last user message verbatim; instead, ask for what's missing.
Keep it to one sentence or a short question.
"""

    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"History:\n{conversation_context}\n\nUser: {last_msg}\n\nClarification:"}
    ]
    response = call_llama(prompt)
    # Ensure response is not empty
    if not response or len(response.strip()) < 3:
        response = f"Sorry {user_name}, could you please clarify what you meant? I can help with {get_capabilities_context()}."

    return {
        "messages": [AIMessage(content=response)],
        "user_email": user_email
    }

# --- 11. GUARDRAIL NODE ---

def guardrail_node(state: AgentState) -> Dict:
    messages = state["messages"]
    last_msg = messages[-1].content
    capabilities = get_capabilities_context()
    user_email = state.get("user_email")

    response = f"""I can't help with "{last_msg}". 

As a Spog.ai assistant, I don't have the ability to perform actions (like updating, creating, or deleting records) or answer questions outside of your business data.

**Here is what I CAN do:**
{capabilities}

Would you like me to look up any of this data for you?"""

    return {
        "messages": [AIMessage(content=response)],
        "user_email": user_email
    }
    
def email_request_node(state: AgentState) -> Dict:
    """
    Handles email requests. Finds the last data response in history,
    queues it via Celery, and replies naturally to the user.
    """
    messages    = state["messages"]
    user_email  = state.get("user_email")
    user_context = state.get("user_context", {})
    user_name   = user_context.get("name", "User")
    first_name  = user_name.split()[0] if user_name and user_name != "User" else "there"
    to_email    = user_email  # use the logged-in email directly

    # Find the last assistant data response and the user query that produced it
    email_content      = None
    user_query_for_email = ""

    # Walk backwards through saved messages to find last assistant response
    # that is NOT itself an email confirmation or guardrail message
    skip_phrases = ["📧", "I can't help with", "I don't have the ability", "sent this response"]

    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        # Handle both dict (from saved history) and LangChain message objects
        role    = msg.get("role") if isinstance(msg, dict) else ("assistant" if isinstance(msg, AIMessage) else "user")
        content = msg.get("content") if isinstance(msg, dict) else msg.content

        if role == "assistant" and content:
            if not any(phrase in content for phrase in skip_phrases):
                email_content = content
                # Find the user query just before this response
                if i > 0:
                    prev = messages[i - 1]
                    prev_role    = prev.get("role") if isinstance(prev, dict) else ("user" if isinstance(prev, HumanMessage) else "assistant")
                    prev_content = prev.get("content") if isinstance(prev, dict) else prev.content
                    if prev_role == "user":
                        user_query_for_email = prev_content
                break

    if not email_content:
        response = f"I don't have a previous response to send yet, {first_name}. Ask me something first and then I'll email you the results!"
        return {"messages": [AIMessage(content=response)], "user_email": user_email}

    # Queue the Celery task
    try:
        send_response_email.delay(
            to_email=to_email,
            user_name=user_name,
            query=user_query_for_email,
            response_text=email_content
        )
        print(f"[EMAIL] Queued email to {to_email} via Celery")
        response = f"Done, {first_name}! I've sent that response to **{to_email}**. It should arrive in your inbox within a minute."
    except Exception as e:
        print(f"[EMAIL] Failed to queue email: {e}")
        response = f"Sorry {first_name}, I wasn't able to send the email right now. Please try again in a moment."

    return {"messages": [AIMessage(content=response)], "user_email": user_email}

# --- 12. ROUTER NODE ---

def router_node(state: AgentState):
    messages = state["messages"]
    last_msg = messages[-1].content
    intent = state.get("intent")
    needs_refetch = state.get("needs_refetch", False)
    user_email = state.get("user_email")

    recursion = state.get("recursion_count", 0)
    if recursion >= 2:
        print("DEBUG ROUTER: Max recursion reached. Breaking infinite loop.")
        return {"selected_api": "none", "analysis_ready": False, "needs_refetch": False, "recursion_count": 0}

    conversation_history = []
    for msg in messages[-6:]:
        role = "USER" if isinstance(msg, HumanMessage) else "ASSISTANT"
        content = msg.content
        if role == "ASSISTANT" and len(content) > 500:
            content = content[:300] + "\n... [DATA TRUNCATED FOR CONTEXT WINDOW] ..."
        conversation_history.append(f"{role}: {content}")

    context_str = "\n".join(conversation_history)

    current_data = state.get("api_data", [])
    data_loaded_count = len(current_data)
    current_api = state.get("selected_api", "none")
    current_api_desc = ""
    if current_api != "none":
        current_api_desc = registry_loader.get_api_config(current_api).get(
            "description", f"Data about {current_api}"
    )

    is_follow_up = False
    if intent == IntentType.ANALYSIS:
        is_follow_up = True
    elif intent != IntentType.API_QUERY:
        is_follow_up = any([
            re.search(r'\b(it|they|them|those|these|that|the one)\b', last_msg.lower()),
            re.search(r'\b(first|second|last|next|previous)\b.*\b(one|item|result)\b', last_msg.lower())
        ])

    # Trap-Breaker: Broaden scope if requested
    if re.search(r'\b(all|every|entire|clear|remove|any|different|topic|new|change|another|stop)\b', last_msg.lower()):
        print("DEBUG ROUTER: User asked to broaden scope. Forcing FETCH_NEW bypass.")
        is_follow_up = False

    if is_follow_up and current_data and not needs_refetch:
        print(f"DEBUG ROUTER: Detected follow-up question with existing data -> ANALYSIS")

        current_api_keys = []
        if current_api and current_api != "none":
            config = registry_loader.get_api_config(current_api)
            current_api_keys = list(config.get("filter_mapping", {}).keys())

        extraction_prompt = f"""Analyze this follow-up question.
Context - Previous data was about: {current_api}
Valid filter keys for this data: {current_api_keys}
Question: "{last_msg}"

1. If the user wants to filter the EXISTING data further, extract the specific filters.
2. If the user wants to BROADEN the search, change the person, or asks for "all" records, set "requires_new_fetch" to true.

Output JSON ONLY:
{{"filters": {{}}, "requires_new_fetch": false}}"""

        ext_response = call_llama([{"role": "user", "content": extraction_prompt}])
        try:
            clean_ext = ext_response.replace("```json", "").replace("```", "").strip()
            if "{" in clean_ext:
                clean_ext = clean_ext[clean_ext.find("{"):clean_ext.rfind("}")+1]
            extracted = json.loads(clean_ext)
            filters = extracted.get("filters", {})
            requires_new = extracted.get("requires_new_fetch", False)
        except:
            filters = {}
            requires_new = False

        if requires_new:
            print("DEBUG ROUTER: Follow-up requires broader data. Bypassing Analysis memory.")
            is_follow_up = False

    search_query = last_msg.strip() or "help"
    candidates = api_router.get_top_candidates(search_query, k=15)
    candidates_text = "\n".join([f"- API: '{c['key']}' | PURPOSE: {c['description']}" for c in candidates]) if candidates else "No APIs found."

    selection_prompt = f"""You are the Intelligent Router for Spog.ai.

CONTEXT:
- Currently Loaded Data: {data_loaded_count} records from API '{current_api}'.
- Domain of Loaded Data: {current_api_desc}
- User Query: "{last_msg}"

CANDIDATE APIS:
{candidates_text}

YOUR GOAL: Determine if this request can be handled by any available API.

CRITICAL DECISION RULES (follow strictly):
1. **Domain Match Check**: First, decide what the user is asking about (e.g., users, tickets, etc.). Compare this to the domain of the currently loaded data.
   - If the user is asking about a **different domain** than the loaded data (e.g., loaded data is tickets but user asks about a user), you MUST decide **FETCH_NEW** and pick the correct API for that domain.
2. **Data Refresh Triggers**: FETCH_NEW is also required if:
   - The user asks for "all" records, or wants to clear previous filters.
   - The user mentions a new person/entity that wasn’t in the loaded data.
   - The user explicitly says something like "show me something else".
3. **Follow‑up Analysis**: Only use ANALYZE_EXISTING if:
   - The user’s question is a **strict follow‑up** about the **exact same data** that is already loaded (e.g., "how many of these are open?", "tell me more about that ticket").
   - The domain matches and no new fetch is needed.
4. If the request cannot be satisfied by any API, output **NONE**.

OUTPUT JSON ONLY:
{{
    "thought_process": "Step‑by‑step reasoning",
    "decision": "FETCH_NEW" | "ANALYZE_EXISTING" | "NONE",
    "api_key": "selected_key_or_null"
}}"""

    prompt = [
        {"role": "system", "content": selection_prompt},
        {"role": "user", "content": f"History:\n{context_str}\n\nMake your decision."}
    ]

    response = call_llama(prompt)

    parsed = {}
    try:
        clean = response.replace("```json", "").replace("```", "").strip()
        if "{" in clean:
            clean = clean[clean.find("{"):clean.rfind("}")+1]
        parsed = json.loads(clean)
    except Exception as e:
        print(f"DEBUG ROUTER: Selection JSON Failed: {e}")
        if data_loaded_count > 0 and not needs_refetch and not is_follow_up:
            parsed = {"decision": "ANALYZE_EXISTING", "api_key": current_api}
        else:
            parsed = {"decision": "FETCH_NEW", "api_key": current_api if current_api != "none" else None}

    decision = parsed.get("decision", "NONE")
    selected_key_from_llm = parsed.get("api_key")
    thought = parsed.get("thought_process", "No thought provided")

    # Python Override: broaden keywords always force FETCH_NEW
    is_broaden = bool(re.search(r'\b(all|every|entire|clear|remove|any|different|topic|new|change|another|stop)\b', last_msg.lower()))
    if is_broaden and selected_key_from_llm and selected_key_from_llm != "none":
        decision = "FETCH_NEW"
        print("DEBUG SMART ROUTER: Python Override applied -> Forced FETCH_NEW due to keyword.")

    print(f"DEBUG SMART ROUTER: Thought='{thought}'")
    print(f"DEBUG SMART ROUTER: Decision='{decision}'")

    if decision == "ANALYZE_EXISTING" and (not current_data or needs_refetch):
        decision = "FETCH_NEW"
        print("DEBUG SMART ROUTER: Forced FETCH_NEW because data was missing or needs refetch.")

    if decision == "NONE" or not selected_key_from_llm or selected_key_from_llm == "none":
        return {"selected_api": "none", "analysis_ready": False, "needs_refetch": False, "user_email": user_email}

    final_api_key = current_api if (decision == "ANALYZE_EXISTING" and current_api != "none") else selected_key_from_llm

    config = registry_loader.get_api_config(final_api_key)
    valid_keys = list(config.get("filter_mapping", {}).keys())

    user_context = state.get("user_context", {})
    teams_str = ", ".join(user_context.get('teams', [])) if user_context.get('teams') else "None"

    extraction_system_prompt = f"""You are an API Extraction Engine for '{final_api_key}'.
    
ALLOWED FILTER KEYS: {json.dumps(valid_keys)}

CURRENT USER IDENTITY:
- Name: {user_context.get('name')}
- Email: {user_context.get('email')}
- Teams: {teams_str}

EXTRACTION RULES (CRITICAL):
1. EXTRACT ONLY SPECIFIC VALUES MENTIONED OR IMPLIED FOR FILTERING.

2. 🔴 EXPLICIT NAMES & SEARCHES:
   - If the user asks for a specific person or item by name (e.g., "Sumit Monga", "Laptop"), you MUST extract it into the most logical search filter (like 'name', 'title', or 'query').
   - Example: User says "details for sumit monga" -> Output {{"name": "Sumit Monga"}}
   
3. 🔴 FIRST-PERSON PRONOUNS ONLY ("my", "mine", "I", "me"):
   - ONLY filter by the Current User Identity if the user EXPLICITLY uses first-person pronouns (e.g., "my tickets").
   - Example: User says "my tickets" -> Output {{"assignee": "{user_context.get('name')}"}}
   - CRITICAL: If the user says "their", "all", "our", or names someone else, DO NOT use the current user's name as a filter!

4. 🔴 THIRD-PERSON PRONOUNS ("he", "she", "him", "her", "them", "this person"):
   - If the user uses a third-person pronoun, you MUST look at the History to figure out the actual name of the person they are talking about.
   - Example: History says "tickets for Shreya", User says "open tickets for her" -> Output {{"assignee": "Shreya", "status": "open"}}
   - NEVER use the literal words "he", "she", "him", or "her" as filter values.

5. 🔴 DISPLAY FIELDS vs FILTERS:
   - If the user asks to SHOW or DISPLAY a specific field (e.g., "with their assignee", "show status"), put it in `display_fields`.
   - DO NOT put it in `filters` unless they are asking to restrict the search to a specific value.

Output JSON ONLY: 
{{
    "filters": {{}},
    "display_fields": [],
    "target_page": null
}}"""

    ext_prompt = [
        {"role": "system", "content": extraction_system_prompt},
        {"role": "user", "content": f"History:\n{context_str}\n\nRequest: {last_msg}"}
    ]

    ext_response = call_llama(ext_prompt)

    extracted_data = {"filters": {}, "display_fields": [], "target_page": None}
    try:
        clean_ext = ext_response.replace("```json", "").replace("```", "").strip()
        if "{" in clean_ext:
            clean_ext = clean_ext[clean_ext.find("{"):clean_ext.rfind("}")+1]
        extracted_data = json.loads(clean_ext)
    except Exception as e:
        print(f"DEBUG ROUTER: Extraction failed: {e}")

    raw_filters = extracted_data.get("filters", {})
    if final_api_key in ["risk_fail", "risk_pass","risk_history"] and "check_id" not in raw_filters:
        # Looks for patterns like CHK-123, RISK-456, etc.
        id_match = re.search(r'\b[A-Z]+-\d+\b', last_msg.upper())
        if id_match:
            raw_filters["check_id"] = id_match.group(0)
            print(f"DEBUG ROUTER: Regex fallback caught missed check_id: {raw_filters['check_id']}")
    generic_terms = ["tickets", "users", "items", "records", "data", "information", "frameworks", "applications", "software", "lists", "list", "our", "system", "my", "employees", "employee", "all", "any"]
    final_filters = {}
    for k, v in raw_filters.items():
        if k in valid_keys:
            if isinstance(v, str) and v.lower() not in generic_terms:
                final_filters[k] = v
            elif not isinstance(v, str):
                final_filters[k] = v

    target_page = extracted_data.get("target_page")
    offset = 0
    mode = "deep_scan"

    if re.search(r'\b(all|everything|entire)\b', last_msg.lower()):
        mode = "deep_scan"
        target_page = None
    elif isinstance(target_page, int) and target_page > 0:
        offset = (target_page - 1) * 20
        mode = "single_page"
    elif re.search(r'\b(page|limit|top)\b', last_msg.lower()):
        mode = "single_page"

    output_state = {
        "selected_api": final_api_key,
        "extracted_filters": final_filters,
        "requested_display_fields": extracted_data.get("display_fields", []),
        "pagination": {"limit": 20, "offset": offset},
        "recursion_count": recursion,
        "mode": mode,
        "analysis_ready": (decision == "ANALYZE_EXISTING" and not needs_refetch),
        "needs_refetch": False,
        "user_email": user_email
    }

    if decision != "ANALYZE_EXISTING":
        output_state["api_data"] = []
    if decision == "ANALYZE_EXISTING" and current_data and final_filters:
        config  = registry_loader.get_api_config(final_api_key)
        mapping = config.get("filter_mapping", {})
        
        for key, target_val in final_filters.items():
            target_str = str(target_val).lower().strip()
            paths      = mapping.get(key, [key])
            
            value_found = False
            for item in current_data:
                for path in paths:
                    actual_val = get_nested_value(item, path).lower().strip()
                    if actual_val and (target_str in actual_val or actual_val in target_str):
                        value_found = True
                        break
                if value_found:
                    break
            
            if not value_found:
                print(f"DEBUG ROUTER: Filter '{key}={target_val}' not found in loaded data -> forcing FETCH_NEW")
                decision = "FETCH_NEW"
                output_state["api_data"] = []
                output_state["analysis_ready"] = False
                break

    return output_state
# ---HELPER DISPLAY ID TO MONGO ID FUNCTION FOR RISHK/HISTORY API
def _resolve_check_id_to_object_id(display_id: str, user_email: str) -> str:
    """
    Secretly hits the 'checks' API to look up the MongoDB ObjectId 
    for a given display ID (like 'CHK-10').
    """
    creds = load_user_credentials(user_email)
    config = registry_loader.get_api_config("checks")
    
    if not creds or not config:
        return display_id  # Fallback to the original string if we can't look it up

    url = f"{creds.get('api_base_url', '')}{config.get('endpoint', '')}"
    filter_key_name = config.get("filter_param", "filters")
    
    # We search the checks API for the specific CHK-10 ID
    search_params = {filter_key_name: json.dumps({"search": display_id}), "limit": 5}

    try:
        print(f"DEBUG RESOLVER: Looking up MongoDB ObjectId for {display_id}...")
        with _authenticated_client(user_email) as client:
            r = client.get(url, params=search_params)
            
            if r.status_code == 200:
                # The API usually returns {"status": true, "data": [...]}
                resp_json = r.json()
                data = resp_json.get("data", []) if isinstance(resp_json, dict) else (resp_json if isinstance(resp_json, list) else [])
                
                if data:
                    # Grab the first matching check
                    first_match = data[0]
                    
                    # The ObjectId is usually '_id', but sometimes it's nested under 'check_detail' depending on your backend
                    obj_id = first_match.get("_id") 
                    if not obj_id and "check_detail" in first_match:
                        obj_id = first_match["check_detail"].get("_id")
                    
                    if obj_id:
                        print(f"DEBUG RESOLVER: Success! Mapped {display_id} -> {obj_id}")
                        return str(obj_id)
                        
    except Exception as e:
        print(f"DEBUG RESOLVER: Network/Lookup error: {e}")

    print(f"DEBUG RESOLVER: Could not find mapping for {display_id}, falling back to original.")
    return display_id
# --- 13. FETCHER NODE ---

def generic_fetcher_node(state: AgentState):
    api_key = state["selected_api"]
    if api_key == "none":
        return {"api_data": [], "needs_refetch": False, "user_email": state.get("user_email")}

    config = registry_loader.get_api_config(api_key)
    is_nclc = config.get("is_nclc", False)

    filters = state.get("extracted_filters") or {}
    user_email = state.get("user_email")
    user_context = state.get("user_context", {})
    true_api_name = user_context.get("name", "Unknown")

    print(f"DEBUG FETCHER: Fetching data for {api_key} with filters: {filters}")
    print(f"DEBUG FETCHER: Local Session Folder: {user_email} | True API Identity: {true_api_name}")

    # Merge default filters with extracted ones
    if state.get("ignore_defaults", False):
        server_filters = {}
    else:
        server_filters = config.get("default_filters", {}).copy()
    server_filters.update(filters)

    # ------------------------------------------------------------------
    # Special handling for risk_fail, risk_pass, risk_history
    # ------------------------------------------------------------------
    if api_key in ["risk_fail", "risk_pass", "risk_history"]:

        # 1. Separate top‑level parameters from nested filters
        top_level_keys = config.get("top_level_params", [])
        top_params = {}
        nested_filters = {}

        for k, v in server_filters.items():
            if k in top_level_keys:
                top_params[k] = v
            else:
                nested_filters[k] = v

        # 2. Ensure check_id is present
        if "check_id" not in nested_filters:
            print("DEBUG FETCHER: Missing required 'check_id'. Aborting API call.")
            return {"api_data": [], "needs_refetch": False, "user_email": user_email}

        val = nested_filters["check_id"]
        extracted_id = val[0] if isinstance(val, list) else str(val)

        # 3. Map check_id to the correct top‑level parameter
        if api_key == "risk_history":
            # Convert display ID (e.g., CHK-67) to MongoDB ObjectId
            resolved_object_id = _resolve_check_id_to_object_id(extracted_id, user_email)
            top_params["risk_id"] = resolved_object_id
            # Remove check_id from nested filters – not needed because risk_id already pins the check
            del nested_filters["check_id"]
        else:
            # For risk_fail / risk_pass
            top_params["test_id"] = extracted_id
            # Keep check_id as an array in nested filters (as required by the backend)
            nested_filters["check_id"] = [extracted_id]

        # 4. Ensure array filters (like check_id) are lists
        array_keys = config.get("array_filters", [])
        for key in array_keys:
            if key in nested_filters and not isinstance(nested_filters[key], list):
                nested_filters[key] = [nested_filters[key]]

        # 5. Paginated fetch (same loop for all three risk APIs)
        target_page = (state["pagination"]["offset"] // 20) + 1
        pages_to_fetch = [target_page] if state.get("mode") == "single_page" else [1, 2, 3, 4, 5]
        all_raw_data = []
        results_by_page = {}

        creds = load_user_credentials(user_email)
        if not creds:
            print(f"DEBUG FETCHER: No credentials found for folder {user_email}")
            return {"api_data": [], "needs_refetch": False, "user_email": user_email}

        with _authenticated_client(user_email) as client:
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_page = {}
                for page in pages_to_fetch:
                    # Build parameters – all risk APIs use query parameters
                    filter_key_name = config.get("filter_param", "filters")
                    params = {}

                    # Add top_params (test_id or risk_id) as top‑level query params
                    for tk, tv in top_params.items():
                        if isinstance(tv, bool):
                            params[tk] = "true" if tv else "false"
                        elif isinstance(tv, (list, dict)):
                            params[tk] = json.dumps(tv)
                        else:
                            params[tk] = tv

                    # Add nested filters as JSON string
                    if nested_filters:
                        params[filter_key_name] = json.dumps(nested_filters)

                    # Add pagination
                    if config.get("pagination", True):
                        params["page"] = page
                        params["limit"] = 20

                    # For NCLC endpoints, wrap in a different structure (if needed)
                    if is_nclc:
                        inner_params = params
                        params = {
                            "endpoint": config.get("nclc_endpoint", "/"),
                            "params": json.dumps(inner_params)
                        }

                    url = f"{creds.get('api_base_url', '')}{config.get('endpoint', '')}"
                    future = executor.submit(_fetch_single_page, client, url, params)
                    future_to_page[future] = page

                for future in concurrent.futures.as_completed(future_to_page):
                    page_num = future_to_page[future]
                    try:
                        data = future.result()
                        if data:
                            # For risk_history, data is a dict (timestamp -> stats); convert to list
                            if api_key == "risk_history" and isinstance(data, dict):
                                formatted_list = []
                                for ts, stats in data.items():
                                    record = {"timestamp": ts}
                                    record.update(stats)
                                    if "check_id" in filters:
                                        record["check_id"] = filters["check_id"]
                                    formatted_list.append(record)
                                results_by_page[page_num] = formatted_list
                            else:
                                # For risk_fail/pass, data is already a list
                                results_by_page[page_num] = data
                            print(f"DEBUG FETCHER: Successfully fetched page {page_num} ({len(results_by_page[page_num])} items)")
                    except Exception as exc:
                        print(f"DEBUG FETCHER: Page {page_num} generated an exception: {exc}")

        for page_num in sorted(results_by_page.keys()):
            all_raw_data.extend(results_by_page[page_num])

        # ✅ Return raw data – the backend already filtered by test_id/risk_id
        return {
            "api_data": all_raw_data,
            "needs_refetch": False,
            "user_email": user_email,
            "interpreter_code": ""
        }

    # ------------------------------------------------------------------
    # All other APIs (original logic)
    # ------------------------------------------------------------------
    target_page = (state["pagination"]["offset"] // 20) + 1
    pages_to_fetch = [target_page] if state.get("mode") == "single_page" else [1, 2, 3, 4, 5]
    all_raw_data = []
    results_by_page = {}

    creds = load_user_credentials(user_email)
    if not creds:
        print(f"DEBUG FETCHER: No credentials found for folder {user_email}")
        return {"api_data": [], "needs_refetch": False, "user_email": user_email}

    with _authenticated_client(user_email) as client:
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_page = {}
            for page in pages_to_fetch:
                if is_nclc:
                    inner_params = {"filters": json.dumps(server_filters), "page": page, "limit": 20}
                    params = {"endpoint": config.get("nclc_endpoint", "/"), "params": json.dumps(inner_params)}
                else:
                    filter_key_name = config.get("filter_param", "filters")
                    params = {filter_key_name: json.dumps(server_filters), "page": page, "limit": 20}
                    for extra_key, extra_val in config.get("extra_params", {}).items():
                        params[extra_key] = json.dumps(extra_val) if isinstance(extra_val, dict) else extra_val

                if not config.get("pagination", True):
                    params.pop("page", None)
                    params.pop("limit", None)

                url = f"{creds.get('api_base_url', '')}{config.get('endpoint', '')}"
                future = executor.submit(_fetch_single_page, client, url, params)
                future_to_page[future] = page

            for future in concurrent.futures.as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    data = future.result()
                    if data:
                        results_by_page[page_num] = data
                        print(f"DEBUG FETCHER: Successfully fetched page {page_num} ({len(data)} items)")
                except Exception as exc:
                    print(f"DEBUG FETCHER: Page {page_num} generated an exception: {exc}")

    for page_num in sorted(results_by_page.keys()):
        all_raw_data.extend(results_by_page[page_num])

    # Apply local filtering for other APIs
    filtered_data = generic_filter_tool(all_raw_data, filters, config)
    return {"api_data": filtered_data, "needs_refetch": False, "user_email": user_email, "interpreter_code": ""}

# --- 14. FORMATTER NODE ---

def formatter_node(state: AgentState):
    api_key      = state["selected_api"]
    user_email   = state.get("user_email")
    full_data    = state.get("api_data", [])
    filters      = state.get("extracted_filters", {})
    messages     = state["messages"]
    user_context = state.get("user_context", {})

    # Guard: nothing to talk about
    if api_key == "none" or not full_data:
        subject = api_key.replace("_", " ") if api_key and api_key != "none" else "that"
        if filters:
            filter_desc = ", ".join(f"{k} = {v}" for k, v in filters.items())
            msg = (
                f"I couldn't find any {subject} matching {filter_desc}. "
                "Want me to try a broader search or check a different filter?"
            )
        else:
            msg = f"No {subject} were returned. The data source may be empty or temporarily unavailable."
        return {
            "messages": [AIMessage(content=msg)],
            "user_email": user_email,
            "table_data": []
        }

    # Recover the user's original question
    user_question = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_question = msg.content
            break
    if api_key == "risk_history" and full_data:
        from datetime import datetime as dt

        sorted_records = sorted(full_data, key=lambda x: int(x.get("timestamp", 0)))
        readable = []
        for r in sorted_records:
            ts   = int(r.get("timestamp", 0))
            date = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%d/%b/%y")
            readable.append({
                "date":       date,
                "pass":       r.get("pass", 0),
                "fail":       r.get("fail", 0),
                "remediated": r.get("remediated", 0),
                "total":      r.get("total", 0)
            })

        records_str = json.dumps(readable, indent=2)

        risk_history_prompt = f"""You are Spog.ai, a smart security analyst assistant.

The user asked: "{user_question}"

Here is the historical risk data sorted oldest to newest:
{records_str}

YOUR JOB:
First, flag the conclusion with coloured statement if the data contains problem with coloured statement red for problem and green for correct, then Write a short, natural analyst briefing. Keep it only to the bullet points, CONCISE, SHORT and to the point, with bullet points only. SHORT

STRICT RULES:
1. Data contains problem if there is large gaps between dates, passed, failed and remediated. incosistency should not happen.
2. START WITH THE CONCLUSION then, Briefly summarize the data — how many snapshots, what date range. (keep it only to the point)
3. Show the latest numbers (pass, fail, remediated).(keep it only to the point)
4. Look at the dates between consecutive records. If there is a large gap between any two dates, flag it — explain that monitoring data may be missing for that period.(keep it only to the point)
5. Look at the pass/fail numbers across records. If there is a major jump or drop between any two consecutive records, flag it and mention the dates.(keep it only to the point)
6.  Look at the remediated numbers across records. If there is a major jump or drop between any two consecutive records, flag it and mention the dates.(keep it only to the point)
7. Only if everything looks consistent, say so.
8. End with one helpful follow-up suggestion. (keep it only to the point)

OUTPUT FORMAT — follow this structure exactly:
<span style="color:red">🔴 INCONSISTENCIES DETECTED</span>  (or <span style="color:green">🟢 ALL CLEAR</span> if no problems)

- **Summary:** [rule 2]

- **Latest:** [rule 3]

- **Date gaps:** [rule 4]

- **Pass/Fail jumps:** [rule 5]

-**Remediated records:** [rule 6]

- **Suggestion:** [rule 7]

FORMATTING RULES:
- Each bullet on its OWN line with a BLANK LINE between bullets
- NEVER place two bullets on the same line
- ONLY use `-` dash bullets, NEVER `•`
- Bold all key numbers and dates
- Max 20 words per bullet
"""

        prompt = [
            {"role": "system", "content": risk_history_prompt},
            {"role": "user",   "content": "Please give me your briefing now."}
        ]
        response = call_llama(prompt)
        response = re.sub(r'```.*?```', '', response, flags=re.DOTALL).strip()
        if not response:
            response = f"Risk history loaded with {len(readable)} snapshots."

        return {
            "messages":   [AIMessage(content=response)],
            "user_email": user_email,
            "table_data": full_data
        }

    # Build a lean JSON preview for the LLM
    config         = registry_loader.get_api_config(api_key)
    mapping        = config.get("filter_mapping", {})
    display_fields = list(config.get("display_fields", []))
    requested      = state.get("requested_display_fields", [])

    for field in requested:
        clean = field.lower().strip()
        if clean in mapping and clean not in display_fields:
            display_fields.append(clean)

    if not display_fields:
        display_fields = list(mapping.keys())

    total         = len(full_data)
    PREVIEW_LIMIT = 25
    lean_preview  = []

    for item in full_data[:PREVIEW_LIMIT]:
        lean_item = {}
        for field in display_fields:
            paths = mapping.get(field, [field])
            val   = ""
            for path in paths:
                val = get_nested_value(item, path)
                if val and val not in ("None", "[]", ""):
                    break
            if val and val not in ("None", "[]", ""):
                if val.startswith("[") and val.endswith("]"):
                    try:
                        val = val.strip("[]").replace("'", "").replace('"', "").strip()
                    except Exception:
                        pass
                lean_item[field] = val
        if lean_item:
            lean_preview.append(lean_item)

    preview_json   = json.dumps(lean_preview, indent=2, default=str)
    truncated      = total > PREVIEW_LIMIT
    scan_note      = "(Scanned pages 1-5)" if state.get("mode") != "single_page" else "(Single page scan)"
    filter_note    = f"Active filters: {json.dumps(filters)}" if filters else "No filters applied (full dataset)."
    user_name      = user_context.get("name", "the user")
    truncation_note = (
        f"\nNOTE: The dataset has {total} records in total. "
        f"You are only seeing the first {PREVIEW_LIMIT} here. "
        "Mention this to the user and tell them they can ask you to analyse or filter further."
    ) if truncated else ""

    system_prompt = f"""You are Spog.ai — a smart, conversational AI assistant. Talk like a helpful colleague who explains things clearly.

**What just happened:** You fetched data from '{api_key}' for {user_name}
**What they asked:** "{user_question}"
**What filters are active:** {filter_note}
**Scan mode:** {scan_note}

⚠️ **CRITICAL — TOTAL RECORD COUNT: {total}**
The JSON below is a PREVIEW only ({min(total, PREVIEW_LIMIT)} of {total} total records).
**ALWAYS use {total} as the count** when describing results to the user.
**NEVER count the JSON items below** and report that number as the total.

**DATA PREVIEW (JSON — just a preview, NOT the full dataset):**
{preview_json}

---

**HOW TO RESPOND (ChatGPT-style guidelines):**

**1. Lead with the correct total count (from above, NOT from counting JSON):**
- "You have **{total}** tickets..." 
- "I found **{total}** users matching..."
- Never say "I see 5 records" when total is 50

**2. Match your response length to the situation:**
- 0 results → 2-3 sentences (apology + suggestion)
- 1-3 results → Describe each briefly (3-5 sentences)
- 4-10 results → Summarize + highlight standouts (4-6 sentences)  
- 10+ results → Group by category, don't list (3-5 sentences)

**3. Write naturally, like texting a colleague:**
- Short paragraphs (2-3 sentences each)
- **Bold** only numbers and key findings
- Bullet points ONLY for 3+ distinct items
- Never say "based on the data" or mention JSON

**4. Use natural language, not field names:**
- ❌ "issue_id: TASK-001 has status: open"
- ✅ "Ticket TASK-001 is still open"
- ❌ "created_at: 2024-01-15"  
- ✅ "submitted on January 15th"

**5. End with a helpful nudge (one sentence max):**
- "Want me to show you the high-priority ones first?"
- "Should I filter these by status?"

**6. If data is empty:**
"I couldn't find any matching records. Want me to try a broader search?"

---

**Now respond naturally (your response only, no explanations):**"""

    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": "Please give me your response now."}
    ]

    response = call_llama(prompt)

    response = re.sub(r'```json.*?```', '', response, flags=re.DOTALL).strip()
    response = re.sub(r'```.*?```',    '', response, flags=re.DOTALL).strip()

    if not response:
        response = f"I found **{total} {api_key.replace('_', ' ')}** records. Want me to analyse or filter them?"

    return {
        "messages":   [AIMessage(content=response)],
        "user_email": user_email,
        "table_data": full_data
    }

# --- 15. ANALYSIS NODE ---

def analysis_node(state: AgentState):
    full_data  = state.get("api_data", [])
    filters    = state.get("extracted_filters", {})
    api_key    = state.get("selected_api", "")
    user_email = state.get("user_email")
    recursion  = state.get("recursion_count", 0)
    messages   = state["messages"]

    # Recover the original user question
    user_question = ""
    for msg in reversed(messages):
        if getattr(msg, "type", "") == "human" or isinstance(msg, HumanMessage):
            user_question = msg.content
            break
    if not user_question:
        user_question = messages[-1].content

    if not full_data:
        return {
            "messages": [AIMessage(content="I don't have any data loaded to analyse. What would you like to search for?")],
            "user_email": user_email
        }

    config = registry_loader.get_api_config(api_key)

    # Apply local filters before analysing
    if filters:
        print(f"DEBUG ANALYSIS: Pre-filtering data with: {filters}")
        filtered_data = generic_filter_tool(full_data, filters, config)
        if not filtered_data:
            print("DEBUG ANALYSIS: Local filter yielded 0 results. Triggering API Fallback.")
            return {"needs_refetch": True, "user_email": user_email, "recursion_count": recursion + 1}
        target_data = filtered_data
    else:
        target_data = full_data

    # Build a lean preview (key fields only, capped at 20 rows)
    lean_data = []
    key_fields = [
        "id", "issue_id", "summary", "description", "name", "email",
        "status", "priority", "assignee", "title", "subject", "type",
        "category", "created_at", "updated_at"
    ]
    limit = min(len(target_data), 20)

    for item in target_data[:limit]:
        lean_item = {}
        for field in key_fields:
            if field in item:
                lean_item[field] = item[field]
        for k, v in item.items():
            if k not in lean_item and k not in ["_links", "meta", "metadata"]:
                if isinstance(v, (str, int, float, bool)) and (not isinstance(v, str) or len(v) < 200):
                    lean_item[k] = v
                elif isinstance(v, dict) and len(str(v)) < 300:
                    lean_item[k] = str(v)[:200]
        lean_data.append(lean_item)

    data_str             = json.dumps(lean_data, indent=1, default=str)
    conversation_context = _messages_to_context(messages[:-1], max_messages=4)

    system_prompt = f"""You are a Senior Data Analyst for Spog.ai.

CONVERSATION CONTEXT:
{conversation_context}

CURRENT LOADED DATA DOMAIN: '{api_key}'
DATA PREVIEW ({len(lean_data)} of {len(target_data)} records shown):
{data_str}

USER QUESTION: "{user_question}"

INSTRUCTIONS:
1. Answer the user's question naturally and conversationally using the data above.
2. For counts or totals, derive them carefully by reading the data — do not guess.
3. If the user asks to "summarise" or "explain" a ticket, read 'summary' and 'description' and explain in plain English BULLET POINTS.
4. Do NOT mention JSON, field names, or your analysis process. Just answer like a smart colleague.
5. If the specific person, ID, or item the user asked about is NOT present in the data, output exactly:
   [DATA_MISSING]

Your response:"""

    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Please analyse based on the instructions provided."}
    ]

    response = call_llama(prompt)

    if "[DATA_MISSING]" in response:
        print("DEBUG ANALYSIS: LLM reported missing data. Triggering API Fallback.")
        return {"needs_refetch": True, "user_email": user_email, "recursion_count": recursion + 1}

    return {
        "messages": [AIMessage(content=response)],
        "needs_refetch": False,
        "user_email": user_email,
        "recursion_count": 0,
        "table_data": []
    }

# --- 16. ROUTING FUNCTIONS ---

def route_by_intent(state):
    intent = state.get("intent", IntentType.API_QUERY)

    if intent == IntentType.CHITCHAT:
        return "chitchat"
    elif intent == IntentType.CLARIFICATION:
        return "clarification"
    elif intent == IntentType.OUT_OF_SCOPE:
        return "guardrail"
    elif intent== IntentType.EMAIL_REQUEST:
        return "email_request"
    else:
        return "router"

def route_router_output(state):
    if state.get("analysis_ready"):
        print(f"DEBUG ROUTING: analysis_ready flag -> analysis node")
        return "analysis"

    if state["selected_api"] == "none":
        print(f"DEBUG ROUTING: No API selected -> guardrail")
        return "guardrail"

    print(f"DEBUG ROUTER OUTPUT: Fetching new data -> fetcher node")
    return "fetcher"

def route_fetcher_output(state):
    last_msg = state["messages"][-1].content.lower()
    analysis_keywords = [
        "analyze", "summary", "details", "tell me", "what about",
        "explain", "break down", "how many", "count", "percentage"
    ]
    if any(kw in last_msg for kw in analysis_keywords):
        return "analysis"
    return "formatter"

def route_analysis_output(state):
    if state.get("needs_refetch"):
        return "router"
    return "end"

# --- 17. GRAPH BUILDER ---

def _build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("chitchat",        chitchat_node)
    workflow.add_node("clarification",   clarification_node)
    workflow.add_node("guardrail",       guardrail_node)
    workflow.add_node("email_request", email_request_node)  # NEW
    workflow.add_node("router",          router_node)
    workflow.add_node("fetcher",         generic_fetcher_node)
    workflow.add_node("formatter",       formatter_node)
    workflow.add_node("analysis",        analysis_node)

    workflow.set_entry_point("classify_intent")

    workflow.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "chitchat":      "chitchat",
            "clarification": "clarification",
            "guardrail":     "guardrail",
            "email_request": "email_request",
            "router":        "router"
        }
    )

    workflow.add_edge("chitchat",      END)
    workflow.add_edge("clarification", END)
    workflow.add_edge("guardrail",     END)
    workflow.add_edge("email_request", END)  # NEW

    workflow.add_conditional_edges(
        "router",
        route_router_output,
        {
            "analysis": "analysis",
            "guardrail": "guardrail",
            "fetcher":   "fetcher"
        }
    )

    workflow.add_conditional_edges(
        "fetcher",
        route_fetcher_output,
        {
            "analysis":  "analysis",
            "formatter": "formatter"
        }
    )

    workflow.add_conditional_edges(
        "analysis",
        route_analysis_output,
        {
            "router": "router",
            "end":    END
        }
    )

    workflow.add_edge("formatter", END)

    return workflow.compile(checkpointer=checkpointer)

_agent = _build_graph()

# --- 18. RUNNER FUNCTIONS WITH TRUE STREAMING ---

def run_chat_stream(message: str, user_email: str, thread_id: str = "default", persist_memory: bool = True):
    print(f"--- Processing Stream (user={user_email}, thread={thread_id}): {message} ---")

    if not user_exists(user_email):
        error_msg = f"User {user_email} not found. Please login first."
        yield json.dumps({"status": "done", "response": error_msg}) + "\n"
        return

    creds = load_user_credentials(user_email)
    if not creds or not creds.get("spog_token"):
        error_msg = f"SPOG token not configured for user {user_email}. Please set it in the UI."
        yield json.dumps({"status": "done", "response": error_msg}) + "\n"
        return

    user_context = creds.get("user_context", {"email": "Unknown"})

    messages = load_chat_messages(user_email, thread_id)

    messages.append({
        "role": "user",
        "content": message,
        "timestamp": datetime.now().isoformat()
    })
    save_chat_messages(user_email, thread_id, messages)

    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "messages": [HumanMessage(content=message)],
        "user_email": user_email,
        "user_context": user_context,
        "table_data": [],
        "recursion_count": 0
    }

    final_response = None

    for event in _agent.stream(initial_state, config=config, stream_mode="updates"):
        for node_name, node_state in event.items():

            if node_name == "classify_intent":
                yield '{"status": "💭 Understanding request..."}\n'
            elif node_name == "router":
                yield '{"status": "🔍 Locating best data source..."}\n'
            elif node_name == "fetcher":
                yield '{"status": "⚡ Fetching data from APIs in parallel..."}\n'
            elif node_name == "analysis":
                if node_state.get("needs_refetch"):
                    yield '{"status": "🔄 Data missing locally. Re-evaluating routing and hitting APIs..."}\n'
                else:
                    yield '{"status": "🧠 Finalizing analysis..."}\n'

            if node_name in ["formatter", "analysis", "chitchat", "guardrail", "clarification","email_request"]:
                if "messages" in node_state and not node_state.get("needs_refetch"):
                    last_msg = node_state["messages"][-1].content

                    if not last_msg:
                        continue

                    final_response = last_msg
                    table_data = node_state.get("table_data", [])

                    words = final_response.split(" ")
                    current_stream = ""
                    for word in words:
                        current_stream += word + " "
                        yield json.dumps({"status": "streaming", "chunk": current_stream}) + "\n"
                        time.sleep(0.015)

                    messages.append({
                        "role": "assistant",
                        "content": final_response,
                        "timestamp": datetime.now().isoformat()
                    })
                    save_chat_messages(user_email, thread_id, messages)

                    update_chat_metadata(user_email, thread_id, {"last_message": final_response[:50]})

                    

                    yield json.dumps({"status": "done", "response": final_response, "table_data": table_data}) + "\n"

def run_chat(message: str, user_email: str, thread_id: str = "default", persist_memory: bool = True) -> str:
    print(f"--- Processing (user={user_email}, thread={thread_id}): {message} ---")

    if not user_exists(user_email):
        return f"User {user_email} not found. Please login first."

    creds = load_user_credentials(user_email)
    if not creds or not creds.get("spog_token"):
        return f"SPOG token not configured for user {user_email}. Please set it in the UI."

    user_context = creds.get("user_context", {"email": "Unknown"})

    messages = load_chat_messages(user_email, thread_id)

    messages.append({
        "role": "user",
        "content": message,
        "timestamp": datetime.now().isoformat()
    })
    save_chat_messages(user_email, thread_id, messages)

    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "messages": [HumanMessage(content=message)],
        "user_email": user_email,
        "user_context": user_context,
        "table_data": [],
        "recursion_count": 0
    }

    result = _agent.invoke(initial_state, config=config)

    final_response = result["messages"][-1].content

    messages.append({
        "role": "assistant",
        "content": final_response,
        "timestamp": datetime.now().isoformat()
    })
    save_chat_messages(user_email, thread_id, messages)

    update_chat_metadata(user_email, thread_id, {"last_message": final_response[:50]})

    return final_response

# --- 19. UTILITY FUNCTIONS ---

def get_user_threads_list(user_email: str) -> List[str]:
    """Get all threads for a user"""
    return get_user_threads(user_email)

def delete_user_thread_by_id(user_email: str, thread_id: str) -> bool:
    """Delete a specific thread for a user"""
    return delete_user_thread(user_email, thread_id)

def get_user_chat_history(user_email: str, thread_id: str) -> List[Dict]:
    """Get chat history for a specific thread"""
    return load_chat_messages(user_email, thread_id)

# --- 20. MAIN ENTRY POINT ---

if __name__ == "__main__":
    print("Spog.ai Chat Assistant (type 'exit' to quit)")
    print("-" * 50)

    test_user = "test@example.com"

    if not user_exists(test_user):
        print(f"Creating test user: {test_user}")
        save_user_credentials(test_user, "test_token", "https://api.example.com")

    thread_id = "cli-session"

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break

        print("\nAssistant: ", end="", flush=True)
        for chunk in run_chat_stream(user_input, test_user, thread_id, persist_memory=True):
            try:
                data = json.loads(chunk)
                if data.get("status") == "done":
                    print(data.get("response", ""))
            except:
                print(chunk, end="", flush=True)
        print()