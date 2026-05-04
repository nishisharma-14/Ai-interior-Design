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

    /* App Background */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 15% 50%, #FAF9F6 0%, #F5F4F0 50%, #E8E6E1 100%);
    }

    /* Hero Text */
    .hero-title {
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #4A4A48 0%, #8B7355 50%, #A39171 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding-top: 1rem;
        margin-bottom: 2.5rem;
        animation: gradient-shift 6s ease infinite;
        background-size: 200% 200%;
        line-height: 1.2;
    }
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .hero-icon {
        font-size: 4rem;
        margin-bottom: 0.5rem;
        animation: float 3s ease-in-out infinite;
        text-align: center;
    }
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }

    /* Top Logo text in sidebar */
    .logo-text {
        font-size: 1.5rem;
        font-weight: 700;
        color: #4A4A48;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }
    
    /* Main Suggestion Buttons (Glassy cards) */
    .block-container div.stButton > button {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        color: #1f2937;
        border-radius: 16px;
        padding: 1.5rem;
        font-weight: 600;
        border: 1px solid rgba(255,255,255,0.5);
        width: 100%;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 6px rgba(0,0,0,0.02), 0 10px 15px rgba(0,0,0,0.03);
    }
    .block-container div.stButton > button:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 10px 25px rgba(139, 115, 85, 0.15);
        border-color: rgba(139, 115, 85, 0.3);
        background: rgba(255, 255, 255, 0.9);
        color: #8B7355;
    }
    .block-container div.stButton > button p {
        font-size: 1.05rem;
    }

    /* Sidebar "New Design" Button (Taupe) */
    [data-testid="stSidebar"] div.stButton > button {
        background-color: #8B7355;
        color: #ffffff;
        border-radius: 8px;
        font-weight: 500;
        border: none;
        transition: all 0.2s;
    }
    [data-testid="stSidebar"] div.stButton > button:hover {
        background-color: #6d5a42;
        color: #ffffff;
    }

    /* iMessage-Style Chat Messages */
    div[data-testid="stChatMessage"] {
        background: transparent !important;
        padding: 0.5rem 0;
        border: none !important;
    }
    
    /* Assistant Bubbles */
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) > div:nth-child(2) {
        background-color: #f1f5f9;
        color: #1f2937;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 4px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        margin-left: 0.5rem;
        width: fit-content;
        max-width: 85%;
    }
    
    /* User Bubbles */
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        flex-direction: row-reverse;
    }
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) > div:nth-child(2) {
        background-color: #8B7355;
        color: #ffffff;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 4px 20px;
        box-shadow: 0 2px 4px rgba(139,115,85,0.2);
        margin-right: 0.5rem;
        width: fit-content;
        max-width: 85%;
    }
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) > div:nth-child(2) p {
        color: #ffffff;
    }
    
    /* Enhanced Color Palette */
    .color-palette-box {
        position: relative;
        width: 100%;
        height: 50px;
        border-radius: 12px;
        border: 1px solid rgba(0,0,0,0.05);
        box-shadow: 0 4px 6px rgba(0,0,0,0.05), inset 0 2px 4px rgba(255,255,255,0.4);
        transition: transform 0.2s, box-shadow 0.2s;
        overflow: hidden;
        cursor: pointer;
    }
    .color-palette-box::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 50%;
        background: linear-gradient(to bottom, rgba(255,255,255,0.5) 0%, rgba(255,255,255,0) 100%);
        border-radius: 12px 12px 0 0;
    }
    .color-palette-box:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.1), inset 0 2px 4px rgba(255,255,255,0.5);
    }
    .color-palette-text {
        text-align: center;
        font-family: monospace;
        font-size: 0.75rem;
        color: #6b7280;
        margin-top: 6px;
        font-weight: 600;
        transition: color 0.2s;
    }

    /* Pill-Shaped Chat Input */
    [data-testid="stChatInput"] {
        border-radius: 9999px !important;
        border: 1px solid #e5e7eb !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05) !important;
        background-color: #ffffff;
        overflow: hidden;
    }
    [data-testid="stChatInput"] > div {
        border: none !important;
        box-shadow: none !important;
        background-color: transparent !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #8B7355 !important;
        box-shadow: 0 4px 12px rgba(139,115,85,0.15) !important;
    }

    /* Slider Thumbs */
    .stSlider [role="slider"] {
        border-radius: 9999px !important;
        background-color: #8B7355 !important;
        border: 2px solid #FAFAF7 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    .stSlider [role="slider"]:hover {
        transform: scale(1.1);
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

def build_fallback_image_prompt(messages, vibe="Balanced", mats=[]):
    conversation_text = " ".join([m["content"] for m in messages if m["role"] != "system"])
    room_types = re.findall(r'(bedroom|living room|kitchen|bathroom|office|dining room|studio)', conversation_text, re.IGNORECASE)
    styles = re.findall(r'(modern|minimalist|rustic|bohemian|scandinavian|industrial|traditional|contemporary|cozy|luxury)', conversation_text, re.IGNORECASE)
    room = room_types[-1] if room_types else "room"
    style = styles[-1] if styles else "modern"
    mat_str = f", featuring {', '.join(mats)}" if mats else ""
    return f"A photorealistic {vibe.lower()} interior design of a {style} {room}{mat_str}, beautifully decorated, professional interior photography, ambient lighting, 8k quality"

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
if "saved_chats" not in st.session_state:
    st.session_state.saved_chats = []
if "trigger_api" not in st.session_state:
    st.session_state.trigger_api = False
if "pending_input" not in st.session_state:
    st.session_state.pending_input = ""
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# Sidebar
with st.sidebar:
    st.markdown("<div class='logo-text'>✨ Interior AI</div>", unsafe_allow_html=True)
    
    # 1. Dark Mode Toggle
    dark_toggle = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    if dark_toggle != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_toggle
        st.rerun()
        
    st.markdown("<hr style='margin: 1rem 0;'>", unsafe_allow_html=True)
    
    # 1.5 Visual Controls
    st.markdown("<div style='font-size: 0.8rem; font-weight: 600; color: #6b7280; margin-bottom: 0.5rem;'>🎛️ DESIGN VIBE</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.75rem; color: #4b5563;'>Complexity</div>", unsafe_allow_html=True)
    vibe_level = st.select_slider("Vibe", options=["Minimalist", "Balanced", "Maximalist"], value="Balanced", label_visibility="collapsed")
    
    st.markdown("<div style='font-size: 0.75rem; color: #4b5563; margin-top: 0.5rem;'>Budget Target</div>", unsafe_allow_html=True)
    budget_level = st.select_slider("Budget", options=["$ IKEA", "$$ Mid-Range", "$$$ Designer"], value="$$ Mid-Range", label_visibility="collapsed")
    
    st.markdown("<div style='font-size: 0.75rem; color: #4b5563; margin-top: 0.5rem;'>Primary Materials</div>", unsafe_allow_html=True)
    materials = st.multiselect("Key Materials", ["🪵 Wood", "🪨 Marble", "⛓️ Metal", "🌿 Plants", "🧶 Velvet", "🧱 Brick"], default=["🪵 Wood", "🌿 Plants"], label_visibility="collapsed")
    
    st.markdown("<hr style='margin: 1rem 0;'>", unsafe_allow_html=True)
    
    # 2. History Section
    st.markdown("<div style='font-size: 0.8rem; font-weight: 600; color: #6b7280; margin-bottom: 1rem;'>🕒 DESIGN HISTORY</div>", unsafe_allow_html=True)
    
    if len(st.session_state.saved_chats) == 0 and len(st.session_state.messages) == 0:
        st.markdown("<div style='font-size: 0.85rem; color: #9ca3af;'>No history yet.</div>", unsafe_allow_html=True)
    else:
        for i, chat in enumerate(st.session_state.saved_chats):
            first_msg = next((m["content"] for m in chat if m["role"] == "user"), f"Saved Session {i+1}")
            if st.button(f"📜 {first_msg[:25]}...", key=f"saved_{i}", use_container_width=True):
                st.session_state.messages = list(chat)
                st.session_state.trigger_api = False
                st.rerun()
                
        if len(st.session_state.messages) > 0:
            current_first_msg = next((m["content"] for m in st.session_state.messages if m["role"] == "user"), "Current Session")
            st.markdown(f"<div style='background: #f3f4f6; padding: 0.75rem; border-radius: 8px; font-size: 0.85rem; color: #4b5563; border: 1px solid #e5e7eb; margin-top: 0.5rem; margin-bottom: 0.5rem;'>🟢 Current: {current_first_msg[:20]}...</div>", unsafe_allow_html=True)

    st.markdown("<hr style='margin: 1rem 0;'>", unsafe_allow_html=True)
    
    # 3. Save Chat and New Design Buttons
    if len(st.session_state.messages) > 0:
        if st.button("💾 Save Chat to Dashboard", use_container_width=True):
            st.session_state.saved_chats.append(list(st.session_state.messages))
            st.success("Chat saved!")

    if st.button("✨ New Design", use_container_width=True):
        st.session_state.messages = []
        st.session_state.trigger_api = False
        st.rerun()
    
    st.markdown("<div style='margin-top: 2rem; font-size: 0.8rem; font-weight: 600; color: #9ca3af; letter-spacing: 1px;'>SYSTEM: CONNECTED</div>", unsafe_allow_html=True)
    


# Apply Dark Mode CSS if enabled
# Apply Dark Mode CSS if enabled
if st.session_state.dark_mode:
    st.markdown("""
    <style>
        html { filter: invert(1) hue-rotate(180deg); }
        img, picture, video, svg, iframe, .color-palette-box { filter: invert(1) hue-rotate(180deg) !important; }
    </style>
    """, unsafe_allow_html=True)

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
                            st.markdown(f'''<div class="color-palette-box" style="background-color:{color};" title="{color}"></div>
                                        <p class="color-palette-text">{color}</p>''', unsafe_allow_html=True)
                
                if "image" in msg and msg["image"]:
                    st.image(msg["image"], use_column_width=True, caption="Generated Concept")

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
        
        api_messages = []
        dynamic_sys_prompt = SYSTEM_PROMPT + f"\n\nUSER CONSTRAINTS:\n- Style/Complexity: {vibe_level}\n- Budget: {budget_level}\n- Preferred Materials: {', '.join(materials)}\nEnsure your recommendations respect these constraints!"
        api_messages.append({"role": "system", "content": dynamic_sys_prompt})
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
                    img_prompt = build_fallback_image_prompt(st.session_state.messages, vibe_level, materials)
                
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
                            st.image(new_msg["image"], use_column_width=True, caption="Generated Concept")
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
