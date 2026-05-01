import streamlit as st
import httpx
import re
import os
from io import BytesIO
from PIL import Image
from huggingface_hub import InferenceClient

st.set_page_config(page_title="Interior Design AI", page_icon="✨", layout="wide", initial_sidebar_state="expanded")

# Light Theme CSS based on Veritas
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Top Logo text in sidebar */
    .logo-text {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e3a8a;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }
    
    /* Main Suggestion Buttons (White cards) */
    .block-container div.stButton > button {
        background-color: #ffffff;
        color: #4b5563;
        border-radius: 8px;
        padding: 1rem;
        font-weight: 400;
        border: 1px solid #e5e7eb;
        width: 100%;
        transition: all 0.2s;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .block-container div.stButton > button:hover {
        border-color: #93c5fd;
        background-color: #f8fafc;
        color: #1e40af;
    }
    .block-container div.stButton > button p {
        font-size: 0.95rem;
    }

    /* Sidebar "New Design" Button (Blue) */
    [data-testid="stSidebar"] div.stButton > button {
        background-color: #1e40af;
        color: #ffffff;
        border-radius: 8px;
        font-weight: 500;
        border: none;
        transition: all 0.2s;
    }
    [data-testid="stSidebar"] div.stButton > button:hover {
        background-color: #1e3a8a;
        color: #ffffff;
    }

    /* Chat Messages */
    div[data-testid="stChatMessage"] {
        background: transparent;
        padding: 1rem 0;
    }
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background-color: #f8fafc;
        border-top: 1px solid #f1f5f9;
        border-bottom: 1px solid #f1f5f9;
        padding: 1.5rem 2rem;
        border-radius: 0;
        margin: 0 -2rem;
    }
    
    div[data-testid="stChatMessage"] p {
        color: #1f2937;
    }

    /* Center column width */
    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 5rem;
    }

    /* Hero */
    .hero-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin-top: 10vh;
        margin-bottom: 3rem;
    }
    .hero-icon {
        font-size: 3rem;
        color: #60a5fa;
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 1.5rem;
        color: #374151;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

HF_TOKEN = os.getenv("HF_TOKEN", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

SYSTEM_PROMPT = """You are an expert interior designer AI assistant. Your goal is to guide the user to design their perfect room through short, conversational questions, and finally provide a summary and image preview.

CRITICAL RULES:
1. BE CONCISE: Keep your responses to 2-3 short sentences. NEVER output large paragraphs.
2. ASK MINIMAL QUESTIONS: Ask only 1 or 2 essential questions at a time.
3. FINAL SUMMARY: ONLY once you have enough details, provide a brief final design summary including:
   - A color palette with EXACT HEX CODES.
   - A short bulleted list of recommended furniture and materials.
4. IMAGE GENERATION: Every time you provide a final design summary with HEX color codes, you MUST also include an image prompt at the VERY END of your message formatted EXACTLY like this on its own line:
   IMAGE_PROMPT: A photorealistic interior design of a modern living room...
5. NEVER forget the IMAGE_PROMPT line when giving final colors/summary.
"""

def build_fallback_image_prompt(messages):
    conversation_text = " ".join([m["content"] for m in messages if m["role"] != "system"])
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

if "messages" not in st.session_state:
    st.session_state.messages = []
if "trigger_api" not in st.session_state:
    st.session_state.trigger_api = False
if "pending_input" not in st.session_state:
    st.session_state.pending_input = ""

# Sidebar
with st.sidebar:
    st.markdown("<div class='logo-text'>✨ Interior AI</div>", unsafe_allow_html=True)
    if st.button("New Design", use_container_width=True):
        st.session_state.messages = []
        st.session_state.trigger_api = False
        st.rerun()
    
    st.markdown("<div style='margin-top: 2rem; font-size: 0.8rem; font-weight: 600; color: #9ca3af; letter-spacing: 1px;'>SYSTEM: CONNECTED</div>", unsafe_allow_html=True)
    st.markdown("<hr style='margin: 1rem 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.8rem; font-weight: 600; color: #6b7280; margin-bottom: 1rem;'>🕒 DESIGN HISTORY</div>", unsafe_allow_html=True)
    if len(st.session_state.messages) > 0:
        first_msg = next((m["content"] for m in st.session_state.messages if m["role"] == "user"), "Current Session")
        st.markdown(f"<div style='background: #f3f4f6; padding: 0.75rem; border-radius: 8px; font-size: 0.85rem; color: #4b5563; border: 1px solid #e5e7eb;'>{first_msg[:50]}...</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='font-size: 0.85rem; color: #9ca3af;'>No history yet.</div>", unsafe_allow_html=True)

# Main Area
if len(st.session_state.messages) == 0:
    st.markdown("""
        <div class="hero-container">
            <div class="hero-icon">✨</div>
            <div class="hero-title">What would you like to design today?</div>
        </div>
    """, unsafe_allow_html=True)
    
    # 2x2 Grid of Suggestions
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Modern minimalist living room", use_container_width=True):
            st.session_state.pending_input = "I want to design a modern minimalist living room."
            st.session_state.trigger_api = True
        if st.button("Cozy scandinavian bedroom", use_container_width=True):
            st.session_state.pending_input = "I want to design a cozy scandinavian bedroom."
            st.session_state.trigger_api = True
    with col2:
        if st.button("Rustic industrial kitchen", use_container_width=True):
            st.session_state.pending_input = "I want to design a rustic industrial kitchen."
            st.session_state.trigger_api = True
        if st.button("Bohemian productive home office", use_container_width=True):
            st.session_state.pending_input = "I want to design a bohemian productive home office."
            st.session_state.trigger_api = True

# Chat Interface
for idx, msg in enumerate(st.session_state.messages):
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg.get("display_content", msg["content"]))
            
            # Display generated image and palette if it exists for this assistant message
            if msg["role"] == "assistant":
                if "palette" in msg and msg["palette"]:
                    st.markdown("<div style='margin-top: 1rem;'><strong>Color Palette:</strong></div>", unsafe_allow_html=True)
                    cols = st.columns(len(msg["palette"]))
                    for c_idx, color in enumerate(msg["palette"]):
                        with cols[c_idx]:
                            st.markdown(f'''<div style="background-color:{color}; width:100%; height:40px; border-radius:6px; border: 1px solid #e5e7eb; box-shadow: 0 1px 2px rgba(0,0,0,0.05);"></div>
                                        <p style='text-align:center; font-family:monospace; font-size:0.75rem; color:#6b7280; margin-top:4px;'>{color}</p>''', unsafe_allow_html=True)
                
                if "image" in msg and msg["image"]:
                    st.image(msg["image"], use_container_width=True, caption="Generated Concept")

# Input Handling
user_input = st.chat_input("Enter a statement to design...")

if user_input or st.session_state.pending_input:
    val = user_input if user_input else st.session_state.pending_input
    st.session_state.pending_input = ""
    
    st.session_state.messages.append({"role": "user", "content": val})
    st.session_state.trigger_api = True
    st.rerun()

# API Trigger
if st.session_state.trigger_api:
    st.session_state.trigger_api = False
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("*(Thinking...)*")
        
        api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in st.session_state.messages:
            api_messages.append({"role": m["role"], "content": m["content"]})
        
        if not GROQ_API_KEY:
            message_placeholder.error("🚨 `GROQ_API_KEY` is missing!")
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
                
                new_msg = {
                    "role": "assistant", 
                    "content": full_reply,
                    "display_content": clean_reply,
                    "palette": hex_colors,
                    "image": None
                }
                
                # Image Generation
                if not img_prompt and hex_colors and len(hex_colors) >= 2:
                    img_prompt = build_fallback_image_prompt(st.session_state.messages)
                
                if img_prompt:
                    img_status = st.empty()
                    img_status.info("🔄 Generating image...")
                    if HF_TOKEN:
                        try:
                            hf_client = InferenceClient(token=HF_TOKEN)
                            image = hf_client.text_to_image(img_prompt, model="black-forest-labs/FLUX.1-schnell")
                            buf = BytesIO()
                            image.save(buf, format="PNG")
                            buf.seek(0)
                            new_msg["image"] = buf.getvalue()
                            img_status.empty()
                            st.image(new_msg["image"], use_container_width=True, caption="Generated Concept")
                        except Exception as img_err:
                            img_status.error(f"❌ Image failed: {str(img_err)}")
                    else:
                        img_status.warning("🚨 HF_TOKEN missing, cannot generate image.")
                
                st.session_state.messages.append(new_msg)
                
                # Rerun to render palettes correctly if needed
                if hex_colors:
                    st.rerun()
                    
        except Exception as e:
            message_placeholder.markdown(f"**Error:** {str(e)}")
