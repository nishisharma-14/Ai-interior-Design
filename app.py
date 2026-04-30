import streamlit as st
import httpx
import re
import os
from io import BytesIO
from PIL import Image
from huggingface_hub import InferenceClient

st.set_page_config(page_title="Interior Design AI", page_icon="🛋️", layout="wide", initial_sidebar_state="collapsed")

# Inject Custom CSS for Widget Look
st.markdown("""
<style>
    /* Bright Clean Background */
    .stApp {
        background-color: #f4f7fb;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    h1 {
        color: #0f172a !important;
        font-weight: 800 !important;
        text-align: center;
        padding-bottom: 2rem;
        letter-spacing: -1px;
    }
    
    /* Remove padding around the scrollable box */
    [data-testid="stVerticalBlock"] > div[data-testid="stScrollableContainer"] {
        background-color: #ffffff;
        border-left: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        padding: 1.5rem !important;
    }
    
    /* Style Visualizer Column to look like a Card */
    [data-testid="column"]:nth-of-type(2) {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
    }
    
    /* AI Message Bubble */
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background-color: #f1f5f9;
        border-radius: 18px;
        border-bottom-left-radius: 4px;
        padding: 0.8rem 1.2rem;
        color: #1e293b;
        border: none;
        margin-right: 15%;
        margin-bottom: 1rem;
    }
    
    /* User Message Bubble */
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        flex-direction: row-reverse;
        background-color: #2563eb;
        border-radius: 18px;
        border-bottom-right-radius: 4px;
        padding: 0.8rem 1.2rem;
        color: white !important;
        border: none;
        margin-left: 15%;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.2);
    }
    
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) p {
        color: white !important;
    }
    
    /* Input field styling to attach seamlessly to the chat box */
    div[data-testid="stTextInput"] input {
        border-radius: 0px 0px 16px 16px !important;
        border: 1px solid #e2e8f0;
        border-top: none;
        padding: 1.2rem;
        background-color: #ffffff;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        font-size: 1rem;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #2563eb;
        box-shadow: none;
    }
    
    /* Hide label on input */
    label[data-testid="stWidgetLabel"] {
        display: none;
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
    # 1. Widget Header (BotPenguin style)
    st.markdown("""
    <div style="background-color: #2563eb; color: white; padding: 1.2rem; border-top-left-radius: 16px; border-top-right-radius: 16px; display: flex; align-items: center; gap: 12px; font-weight: 600; font-size: 1.2rem; margin-bottom: -1rem; position: relative; z-index: 10; box-shadow: 0 4px 10px rgba(0,0,0,0.1);">
        <span style="font-size: 1.5rem; background: white; border-radius: 50%; padding: 4px;">🤖</span> 
        <span style="letter-spacing: 0.5px;">Interior AI Assistant</span>
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

    # 3. Input Box directly attached to the bottom (using standard text input to bypass Streamlit chat limitations)
    st.text_input("Type a message...", key="chat_input_widget", on_change=submit_chat, placeholder="Write a reply...")

with col_vis:
    st.markdown("<h3 style='color: #0f172a; font-weight: 700; margin-bottom: 1.5rem; border-bottom: 2px solid #f1f5f9; padding-bottom: 0.5rem;'>🖼️ Live Visualizer</h3>", unsafe_allow_html=True)
    
    if st.session_state.generated_image:
        try:
            image = Image.open(BytesIO(st.session_state.generated_image))
            st.image(image, width="stretch")
        except Exception:
            st.error("Failed to decode the image from the API.")
    else:
        st.markdown("""
        <div style="text-align:center; padding: 5rem 1rem; background-color: #f8fafc; border-radius: 12px; border: 2px dashed #cbd5e1; margin-bottom: 2rem;">
            <div style="font-size: 4rem; color: #94a3b8; margin-bottom: 1rem;">📸</div>
            <p style="font-size: 1.1rem; color: #64748b; font-weight: 500;">Chat with the AI to refine your design.<br>Your premium render will appear here.</p>
        </div>
        """, unsafe_allow_html=True)
        
    if st.session_state.palette:
        st.markdown("<h4 style='text-align:center; color:#334155; font-weight:700; margin-bottom: 1rem;'>Selected Color Palette</h4>", unsafe_allow_html=True)
        cols = st.columns(len(st.session_state.palette))
        for idx, color in enumerate(st.session_state.palette):
            with cols[idx]:
                st.markdown(
                    f'''<div style="
                        background-color:{color}; 
                        width:100%; 
                        height:60px; 
                        border-radius:8px; 
                        border: 1px solid rgba(0,0,0,0.1);
                        box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                    </div>''',
                    unsafe_allow_html=True
                )
                st.markdown(f"<p style='text-align:center; font-family:monospace; font-size:0.85rem; color:#475569; margin-top:8px; font-weight:600;'>{color}</p>", unsafe_allow_html=True)
