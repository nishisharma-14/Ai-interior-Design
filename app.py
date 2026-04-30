import streamlit as st
import httpx
import re
import os
from io import BytesIO
from PIL import Image
from huggingface_hub import InferenceClient

st.set_page_config(page_title="Interior Design AI", page_icon="🛋️", layout="wide", initial_sidebar_state="collapsed")

# Inject Custom CSS for Dark Mode Chat Widget Look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Dark Theme Background */
    .stApp {
        background: radial-gradient(circle at 15% 50%, #1e1b4b, #09090b);
        color: #f8fafc;
        font-family: 'Outfit', sans-serif;
    }
    
    h1 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700 !important;
        font-size: 3.5rem !important;
        background: linear-gradient(90deg, #d8b4fe, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding-bottom: 2rem;
        margin-top: -2rem;
    }
    
    /* Scrollable Chat Box Container */
    [data-testid="stVerticalBlock"] > div[data-testid="stScrollableContainer"] {
        background: rgba(15, 23, 42, 0.4) !important;
        border-left: 2px solid rgba(255,255,255,0.2) !important;
        border-right: 2px solid rgba(255,255,255,0.2) !important;
        padding: 1.5rem !important;
    }
    
    /* Style Visualizer Column to look like a Card */
    [data-testid="column"]:nth-of-type(2) {
        background: rgba(15, 23, 42, 0.4);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        border: 2px solid rgba(255,255,255,0.2);
    }
    
    /* AI Message Bubble */
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 18px;
        border-bottom-left-radius: 4px;
        padding: 0.8rem 1.2rem;
        color: #f8fafc;
        margin-right: 15%;
        margin-bottom: 1rem;
    }
    
    /* User Message Bubble */
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        flex-direction: row-reverse;
        background: rgba(129, 140, 248, 0.15);
        border: 1px solid rgba(129, 140, 248, 0.3);
        border-radius: 18px;
        border-bottom-right-radius: 4px;
        padding: 0.8rem 1.2rem;
        color: #f8fafc !important;
        margin-left: 15%;
        margin-bottom: 1rem;
    }
    
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) p {
        color: #f8fafc !important;
    }
    
    /* Input field styling */
    div[data-testid="stTextInput"] input {
        border-radius: 0px 0px 16px 16px !important;
        border: 2px solid rgba(255,255,255,0.2);
        border-top: none;
        padding: 1.2rem;
        background: rgba(15, 23, 42, 0.6);
        color: white;
        font-size: 1rem;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #818cf8;
        background: rgba(15, 23, 42, 0.9);
        color: white;
        box-shadow: none;
    }
    
    /* Hide label on input */
    label[data-testid="stWidgetLabel"] {
        display: none;
    }
    
    /* Scrollbar for chat box */
    [data-testid="stScrollableContainer"]::-webkit-scrollbar {
        width: 6px;
    }
    [data-testid="stScrollableContainer"]::-webkit-scrollbar-track {
        background: transparent;
    }
    [data-testid="stScrollableContainer"]::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

HF_TOKEN = os.getenv("HF_TOKEN", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

SYSTEM_PROMPT = """You are an expert interior designer AI assistant. Your goal is to guide the user to design their perfect room through short, conversational questions, and finally provide a summary and image preview.

CRITICAL RULES:
1. BE CONCISE: Keep your responses to 2-3 short sentences. NEVER output large paragraphs.
2. ASK MINIMAL QUESTIONS: Ask only 1 or 2 essential questions at a time (e.g., "What is the room type?", "Do you have a specific style or color in mind?"). Do not overwhelm the user.
3. FINAL SUMMARY: ONLY once you have enough details (room type, style, and colors), provide a brief final design summary including:
   - A color palette with EXACT HEX CODES (e.g., #2E3440).
   - A short bulleted list of recommended furniture and materials.
4. IMAGE GENERATION: ONLY in your FINAL message (when the design is complete), include an image prompt at the very end of your message formatted EXACTLY like this:
   IMAGE_PROMPT: A beautiful modern living room with large windows, dark grey sofa...
5. DO NOT include the "IMAGE_PROMPT:" tag until the final design plan is ready and all questions have been answered.
"""

def extract_hex_colors(text):
    return list(set(re.findall(r'(#[A-Fa-f0-9]{6})\b', text)))

def extract_image_prompt(text):
    match = re.search(r'IMAGE_PROMPT:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

def clean_message(text):
    return re.sub(r'IMAGE_PROMPT:\s*(.+?)(?:\n|$)', '', text, flags=re.IGNORECASE).strip()

st.title("✨ Interior Design AI")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "generated_image" not in st.session_state:
    st.session_state.generated_image = None
if "palette" not in st.session_state:
    st.session_state.palette = []
if "trigger_api" not in st.session_state:
    st.session_state.trigger_api = False
if "chat_input_widget" not in st.session_state:
    st.session_state.chat_input_widget = ""

def submit_chat():
    val = st.session_state.chat_input_widget
    if val and val.strip():
        st.session_state.messages.append({"role": "user", "content": val})
        st.session_state.trigger_api = True
        st.session_state.chat_input_widget = "" # clear input

col_chat, col_vis = st.columns([1.1, 1], gap="large")

with col_chat:
    # 1. Widget Header (Dark, Outlined)
    st.markdown("""
    <div style="background: rgba(30, 41, 59, 0.9); color: white; padding: 1.2rem; border-top-left-radius: 16px; border-top-right-radius: 16px; border: 2px solid rgba(255,255,255,0.2); border-bottom: none; display: flex; align-items: center; gap: 12px; font-weight: 600; font-size: 1.2rem; margin-bottom: -1rem; position: relative; z-index: 10;">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#d8b4fe" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>
        <span style="letter-spacing: 0.5px;">Design Chat</span>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. Chat Box
    chat_box = st.container(height=480, border=False)
    
    with chat_box:
        if len(st.session_state.messages) == 0:
            with st.chat_message("assistant"):
                st.markdown("Hi! Welcome to Interior AI. I'll be assisting you today.\n\n**How can I help you design your dream room?**")
                
        for msg in st.session_state.messages:
            if msg["role"] != "system":
                with st.chat_message(msg["role"]):
                    st.markdown(msg["display_content"] if "display_content" in msg else msg["content"])
                    
        # API Logic execution inside chat box
        if st.session_state.trigger_api:
            st.session_state.trigger_api = False
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("*(Typing...)*")
                
                api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                for m in st.session_state.messages:
                    api_messages.append({"role": m["role"], "content": m["content"]})
                
                if not GROQ_API_KEY:
                    message_placeholder.error("🚨 **Configuration Error:** `GROQ_API_KEY` is missing! \n\nPlease go to your Hugging Face Space **Settings** -> **Variables and secrets**, add `GROQ_API_KEY`, and then click **Factory Reboot** to restart the space.")
                    st.stop()
                    
                try:
                    headers = {
                        "Authorization": f"Bearer {GROQ_API_KEY}",
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "model": "llama-3.3-70b-versatile", 
                        "messages": api_messages,
                        "temperature": 0.7,
                        "top_p": 0.9
                    }
                    
                    with httpx.Client(timeout=60.0) as client:
                        response = client.post(
                            "https://api.groq.com/openai/v1/chat/completions",
                            headers=headers,
                            json=payload
                        )
                        response.raise_for_status()
                        data = response.json()
                        full_reply = data["choices"][0]["message"]["content"]
                        
                        img_prompt = extract_image_prompt(full_reply)
                        hex_colors = extract_hex_colors(full_reply)
                        clean_reply = clean_message(full_reply)
                        
                        message_placeholder.markdown(clean_reply)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": full_reply,
                            "display_content": clean_reply
                        })
                        
                        if hex_colors:
                            st.session_state.palette = hex_colors
                        
                        if img_prompt:
                            with st.spinner("✨ Design finalized! Generating your stunning 3D room preview..."):
                                if not HF_TOKEN:
                                    st.error("🚨 `HF_TOKEN` is missing! Please add it in Space Settings -> Variables and secrets.")
                                else:
                                    hf_client = InferenceClient(token=HF_TOKEN)
                                    image = hf_client.text_to_image(img_prompt, model="black-forest-labs/FLUX.1-schnell")
                                    buf = BytesIO()
                                    image.save(buf, format="PNG")
                                    st.session_state.generated_image = buf.getvalue()
                    
                    st.rerun()
                except Exception as e:
                    message_placeholder.markdown(f"**Error:** {str(e)}")

    # 3. Input Box directly attached to the bottom
    st.text_input("Type a message...", key="chat_input_widget", on_change=submit_chat, placeholder="Write a reply...")

with col_vis:
    st.markdown("<h3 style='color: #e2e8f0; font-weight: 500; margin-bottom: 1.5rem; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 0.5rem;'>🖼️ Live Visualizer</h3>", unsafe_allow_html=True)
    
    if st.session_state.generated_image:
        try:
            image = Image.open(BytesIO(st.session_state.generated_image))
            st.image(image, width="stretch")
        except Exception:
            st.error("Failed to decode the image from the API.")
    else:
        st.markdown("""
        <div style="text-align:center; padding: 5rem 1rem; background: rgba(0,0,0,0.2); border-radius: 12px; border: 2px dashed rgba(255,255,255,0.1); margin-bottom: 2rem;">
            <div style="font-size: 4rem; color: #94a3b8; margin-bottom: 1rem; background: linear-gradient(90deg, #d8b4fe, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">📸</div>
            <p style="font-size: 1.1rem; color: #cbd5e1; font-weight: 300;">Chat with the AI to refine your design.<br>Your premium room render will appear here.</p>
        </div>
        """, unsafe_allow_html=True)
        
    if st.session_state.palette:
        st.markdown("<h4 style='text-align:center; color:#d8b4fe; font-weight:500; margin-bottom: 1rem;'>Selected Color Palette</h4>", unsafe_allow_html=True)
        cols = st.columns(len(st.session_state.palette))
        for idx, color in enumerate(st.session_state.palette):
            with cols[idx]:
                st.markdown(
                    f'''<div style="
                        background-color:{color}; 
                        width:100%; 
                        height:60px; 
                        border-radius:8px; 
                        border: 1px solid rgba(255,255,255,0.1);
                        box-shadow: 0 4px 6px rgba(0,0,0,0.2);">
                    </div>''',
                    unsafe_allow_html=True
                )
                st.markdown(f"<p style='text-align:center; font-family:monospace; font-size:0.85rem; color:#cbd5e1; margin-top:8px; font-weight:400;'>{color}</p>", unsafe_allow_html=True)
