import streamlit as st
import httpx
import re
import os
from io import BytesIO
from PIL import Image
from huggingface_hub import InferenceClient

st.set_page_config(page_title="Interior Design AI", page_icon="🛋️", layout="wide", initial_sidebar_state="collapsed")

# Inject Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
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
    
    /* Both columns get the card treatment */
    [data-testid="column"] {
        background: rgba(15, 23, 42, 0.4);
        padding: 0rem;
        border-radius: 16px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        border: 2px solid rgba(255,255,255,0.15);
        overflow: hidden;
    }
    
    /* Scrollable Chat Box Container */
    [data-testid="stVerticalBlock"] > div[data-testid="stScrollableContainer"] {
        background: transparent !important;
        border: none !important;
        padding: 1.2rem !important;
    }
    
    /* AI Message Bubble */
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: rgba(0, 0, 0, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        border-bottom-left-radius: 4px;
        padding: 0.8rem 1.2rem;
        color: #f8fafc;
        margin-right: 10%;
        margin-bottom: 0.8rem;
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
        margin-left: 10%;
        margin-bottom: 0.8rem;
    }
    
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) p {
        color: #f8fafc !important;
    }
    
    /* Input field */
    div[data-testid="stTextInput"] input {
        border-radius: 0px 0px 14px 14px !important;
        border: none;
        border-top: 1px solid rgba(255,255,255,0.1);
        padding: 1.2rem;
        background: rgba(0, 0, 0, 0.3);
        color: white;
        font-size: 1rem;
    }
    div[data-testid="stTextInput"] input::placeholder {
        color: rgba(255,255,255,0.3);
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #818cf8;
        background: rgba(0, 0, 0, 0.5);
        color: white;
        box-shadow: none;
    }
    
    label[data-testid="stWidgetLabel"] {
        display: none;
    }
    
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
4. IMAGE GENERATION: This is MANDATORY. Every time you provide a final design summary with HEX color codes, you MUST also include an image prompt at the VERY END of your message formatted EXACTLY like this on its own line:
   IMAGE_PROMPT: A photorealistic interior design of a modern living room with large windows, dark grey sofa, wooden floor, ambient lighting, 8k quality
5. NEVER forget the IMAGE_PROMPT line. If your message contains HEX codes and furniture suggestions, it MUST end with IMAGE_PROMPT.
6. DO NOT include the "IMAGE_PROMPT:" tag in early conversational messages where you are still asking questions.
"""

def build_fallback_image_prompt(messages):
    """Build an image prompt from conversation history when the AI forgets to include one."""
    conversation_text = " ".join([m["content"] for m in messages if m["role"] != "system"])
    # Extract key design details
    room_types = re.findall(r'(bedroom|living room|kitchen|bathroom|office|dining room|studio)', conversation_text, re.IGNORECASE)
    styles = re.findall(r'(modern|minimalist|rustic|bohemian|scandinavian|industrial|traditional|contemporary|cozy|luxury)', conversation_text, re.IGNORECASE)
    room = room_types[-1] if room_types else "room"
    style = styles[-1] if styles else "modern"
    return f"A photorealistic interior design of a {style} {room}, beautifully decorated, professional interior photography, ambient lighting, 8k quality"

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
        st.session_state.chat_input_widget = ""

col_chat, col_vis = st.columns([1.1, 1], gap="large")

with col_chat:
    # Header bar inside the card
    st.markdown("""
    <div style="background: rgba(0,0,0,0.3); padding: 1rem 1.5rem; display: flex; align-items: center; gap: 10px; border-bottom: 1px solid rgba(255,255,255,0.1);">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#d8b4fe" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>
        <span style="font-weight: 600; font-size: 1.1rem; color: #e2e8f0;">Design Chat</span>
    </div>
    """, unsafe_allow_html=True)
    
    chat_box = st.container(height=450, border=False)
    
    with chat_box:
        if len(st.session_state.messages) == 0:
            with st.chat_message("assistant"):
                st.markdown("Hi! Welcome to Interior AI. I'll be assisting you today.\n\n**What type of room would you like to design?**")
                
        for msg in st.session_state.messages:
            if msg["role"] != "system":
                with st.chat_message(msg["role"]):
                    st.markdown(msg["display_content"] if "display_content" in msg else msg["content"])
                    
        if st.session_state.trigger_api:
            st.session_state.trigger_api = False
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("*(Typing...)*")
                
                api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                for m in st.session_state.messages:
                    api_messages.append({"role": m["role"], "content": m["content"]})
                
                if not GROQ_API_KEY:
                    message_placeholder.error("🚨 `GROQ_API_KEY` is missing! Add it in Space Settings -> Secrets, then Factory Reboot.")
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
                        
                        # Fallback: if AI gave colors but forgot IMAGE_PROMPT, auto-generate one
                        if not img_prompt and hex_colors and len(hex_colors) >= 2:
                            img_prompt = build_fallback_image_prompt(st.session_state.messages)
                        
                        if img_prompt:
                            st.session_state["img_status"] = f"🔄 Generating image..."
                            if not HF_TOKEN:
                                st.session_state["img_status"] = "🚨 HF_TOKEN is missing! Add it in Space Settings -> Secrets, then Factory Reboot."
                            else:
                                try:
                                    hf_client = InferenceClient(token=HF_TOKEN)
                                    image = hf_client.text_to_image(img_prompt, model="black-forest-labs/FLUX.1-schnell")
                                    buf = BytesIO()
                                    image.save(buf, format="PNG")
                                    buf.seek(0)
                                    st.session_state.generated_image = buf.getvalue()
                                    st.session_state["img_status"] = "✅ Image generated!"
                                except Exception as img_err:
                                    st.session_state["img_status"] = f"❌ Image failed: {str(img_err)}"
                        
                    st.rerun()
                except Exception as e:
                    message_placeholder.markdown(f"**Error:** {str(e)}")

    # Input attached to bottom of the card
    st.text_input("Type a message...", key="chat_input_widget", on_change=submit_chat, placeholder="Type your message and press Enter...")

with col_vis:
    # Header bar inside the card
    st.markdown("""
    <div style="background: rgba(0,0,0,0.3); padding: 1rem 1.5rem; display: flex; align-items: center; gap: 10px; border-bottom: 1px solid rgba(255,255,255,0.1);">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#d8b4fe" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>
        <span style="font-weight: 600; font-size: 1.1rem; color: #e2e8f0;">Live Visualizer</span>
    </div>
    """, unsafe_allow_html=True)
    
    vis_container = st.container()
    with vis_container:
        # Show status message (persists across reruns)
        if "img_status" in st.session_state and st.session_state["img_status"]:
            status = st.session_state["img_status"]
            if "🚨" in status or "❌" in status:
                st.error(status)
            elif "✅" in status:
                st.success(status)
            else:
                st.info(status)
        
        if st.session_state.generated_image:
            try:
                st.image(st.session_state.generated_image, use_column_width=True)
            except Exception as display_err:
                st.error(f"Failed to display the image: {str(display_err)}")
        else:
            st.markdown("""
            <div style="text-align:center; padding: 6rem 1rem;">
                <div style="font-size: 4rem; margin-bottom: 1rem; opacity: 0.4;">🏠</div>
                <p style="font-size: 1.05rem; color: #94a3b8; font-weight: 400;">Chat with the AI to refine your design.<br>Your room render will appear here.</p>
            </div>
            """, unsafe_allow_html=True)
            
        if st.session_state.palette:
            st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 1.5rem 1rem;'>", unsafe_allow_html=True)
            st.markdown("<p style='text-align:center; color:#d8b4fe; font-weight:600; font-size:1rem; margin-bottom: 1rem;'>Color Palette</p>", unsafe_allow_html=True)
            cols = st.columns(len(st.session_state.palette))
            for idx, color in enumerate(st.session_state.palette):
                with cols[idx]:
                    st.markdown(
                        f'''<div style="
                            background-color:{color}; 
                            width:100%; 
                            height:50px; 
                            border-radius:8px; 
                            border: 1px solid rgba(255,255,255,0.1);
                            box-shadow: 0 4px 6px rgba(0,0,0,0.2);">
                        </div>''',
                        unsafe_allow_html=True
                    )
                    st.markdown(f"<p style='text-align:center; font-family:monospace; font-size:0.8rem; color:#cbd5e1; margin-top:6px;'>{color}</p>", unsafe_allow_html=True)
