import streamlit as st
import httpx
import re
import os
from io import BytesIO
from PIL import Image
from huggingface_hub import InferenceClient

st.set_page_config(page_title="Interior Design AI", page_icon="🛋️", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for Absolute Premium UI
st.markdown("""
<style>
    /* Global Reset & Font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: radial-gradient(circle at 15% 50%, #1e1b4b, #09090b);
        color: #f8fafc;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Top Header */
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
    
    /* Elegant Subheadings */
    h3 {
        color: #e2e8f0 !important;
        font-family: 'Outfit', sans-serif;
        font-weight: 500 !important;
        letter-spacing: 0.5px;
    }
    
    /* Chat Container Box */
    [data-testid="stVerticalBlock"] > div[data-testid="stScrollableContainer"] {
        background: rgba(15, 23, 42, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        padding: 1.5rem;
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
    
    /* Chat Bubbles Styling */
    div[data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 1.2rem;
        margin-bottom: 1.2rem;
        backdrop-filter: blur(8px);
        animation: fadeIn 0.4s ease-out;
    }
    
    /* Chat Input */
    div[data-testid="stChatInput"] {
        border-radius: 25px !important;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        background: rgba(15, 23, 42, 0.8) !important;
    }
    
    /* Visualizer Box styling */
    .visualizer-box {
        background: rgba(15, 23, 42, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        height: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    
    /* Swatch Container */
    .palette-box {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 1.5rem;
        border-radius: 16px;
        margin-top: 2rem;
        width: 100%;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
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

# Move chat input OUTSIDE of the columns to fix Streamlit SDK constraints
prompt = st.chat_input("E.g. A cozy modern bedroom with lots of natural light...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

col_chat, col_vis = st.columns([1.2, 1], gap="large")

with col_chat:
    st.markdown("### 💬 Design Chat")
    
    # Scrollable Chat Box Container
    chat_box = st.container(height=580, border=False)
    
    with chat_box:
        # Welcome message
        if len(st.session_state.messages) == 0:
            with st.chat_message("assistant"):
                st.markdown("Hello! I'm your personal AI Interior Designer. Let's create your perfect space. \n\n**What type of room are we working on today?**")
                
        # Display chat messages
        for msg in st.session_state.messages:
            if msg["role"] != "system":
                with st.chat_message(msg["role"]):
                    st.markdown(msg["display_content"] if "display_content" in msg else msg["content"])

        if prompt:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("*(Thinking...)*")
                
                # Real API Call
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

with col_vis:
    st.markdown("### 🖼️ Visualizer")
    
    st.markdown("<div class='visualizer-box'>", unsafe_allow_html=True)
    if st.session_state.generated_image:
        try:
            image = Image.open(BytesIO(st.session_state.generated_image))
            st.image(image, width="stretch")
        except Exception:
            st.error("Failed to decode the image from the API.")
    else:
        st.markdown("""
        <div style="text-align:center; padding: 4rem 1rem; opacity: 0.6;">
            <div style="font-size: 5rem; margin-bottom: 1rem; background: linear-gradient(90deg, #d8b4fe, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">📸</div>
            <p style="font-size: 1.2rem; color: #cbd5e1; font-weight: 300;">Chat with the AI to refine your design.<br>Your premium room render will appear here.</p>
        </div>
        """, unsafe_allow_html=True)
        
    if st.session_state.palette:
        st.markdown("<div class='palette-box'>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align:center; margin-bottom:1rem; color:#d8b4fe; font-weight:500;'>Selected Color Palette</h4>", unsafe_allow_html=True)
        cols = st.columns(len(st.session_state.palette))
        for idx, color in enumerate(st.session_state.palette):
            with cols[idx]:
                st.markdown(
                    f'''<div style="
                        background-color:{color}; 
                        width:100%; 
                        height:70px; 
                        border-radius:12px; 
                        border: 2px solid rgba(255,255,255,0.15);
                        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
                        transition: transform 0.2s;">
                    </div>''',
                    unsafe_allow_html=True
                )
                st.markdown(f"<p style='text-align:center; font-family:monospace; font-size:0.9rem; color:#cbd5e1; margin-top:10px;'>{color}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
