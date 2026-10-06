from RAG_ChatBot import ChatBot
import streamlit as st
from PIL import Image

# 1. LOAD GAMBAR DAN SET PAGE CONFIG (Wajib jadi perintah st pertama)
logo_image = Image.open("logo_ermai.png")
st.set_page_config(
    page_title="ERM Tutor AI",
    page_icon=logo_image,
    layout="centered"
)

# 2. INISIASI BOT DENGAN CACHE (Ditaruh setelah page config)
@st.cache_resource
def get_bot():
    return ChatBot()

bot = get_bot()

# Custom CSS (Diperbaiki agar warna teks chat jelas dan kontras)
st.markdown("""
    <style>
        body {
            background-color: #e6f0ff;
        }
        .stApp {
            background-color: #e6f0ff;
            color: #003366;
        }
        .stChatMessage {
            background-color: #f0f8ff;
            border-radius: 10px;
            margin-bottom: 10px;
            padding: 10px;
        }
        /* Memaksa semua teks di dalam chat box berwarna biru tua pekat */
        .stChatMessage p, .stChatMessage div, .stChatMessage span {
            color: #003366 !important;
        }
        .stTextInput>div>div>input {
            background-color: #ffffff;
            color: #003366;
            border: 1px solid #cce0ff;
        }
        .stMarkdown h1 {
            color: #003366;
        }
        .source-info {
            font-size: 0.85em;
            color: #555555;
            margin-top: 10px;
        }
        .word-badge {
            display: inline-block;
            background-color: #E0F7FA;
            color: #00796B;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.85em;
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Display logo
st.image(logo_image, width=150)

# Copyright directly below logo, in one line
st.markdown(
    """
    <div style='width: 150px; text-align: center; font-size: 0.63em; color: #777777; white-space: nowrap; margin-top: 5px; margin-bottom: 20px;'>
        © Developed by Amrina as part of a thesis project.
    </div>
    """,
    unsafe_allow_html=True
)

# Title and description
st.title("Hello! I'm your AI tutor for Empirical Research Methods (ERM) course.")
st.markdown(
    "Ask me anything about the **Empirical Research Methods (ERM)** course. "
    "I’ll give short, reliable answers based on your course materials, and not just that. "
    "As your AI tutor, I’m also here to guide you, reflect on your questions, and support your learning journey like a real mentor would. 😊"
)

# Chat history setup
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("source_info"):
            st.markdown(f"<div class='source-info'>{msg['source_info']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='word-badge'>📝 {msg['word_count']} words</div>", unsafe_allow_html=True)

# User input
user_input = st.chat_input("Type your question here...")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("🧑‍🏫 *'Great question — let's break it down together!'*"):
            try:
                # Get response from chatbot
                result = bot.rag_chain.invoke(user_input)
                answer = result["result"]
                sources = result.get("source_documents", [])
                word_count = len(answer.split())

                # Cek apakah ini pertanyaan umum/identitas (sapaan)
                is_general_greeting = any(keyword in user_input.lower() for keyword in ["who are you", "siapa kamu", "hello", "hi", "halo", "selamat pagi", "selamat sore"])

                source_info = None
                if is_general_greeting:
                    source_info = "🤖 <i>System Identity</i>"
                elif sources:
                    # Jika ada dokumen materi yang cocok
                    meta = sources[0].metadata
                    file_path = meta.get("source", "")
                    import os
                    file_name = os.path.basename(file_path)
                    clean_name = os.path.splitext(file_name)[0].replace("_", " ")
                    if clean_name:
                        source_info = f"📘 <i>Course Materials: {clean_name}</i>"
                    else:
                        source_info = "📘 <i>Course Materials</i>"
                else:
                    # Jika pertanyaan ERM dijawab di luar dokumen lokal
                    source_info = "🌐 <i>General Academic & Research Methodology Knowledge</i>"

                # Show answer
                st.markdown(answer)

                # Tampilkan sumber jika ada info sumbernya
                if source_info:
                    st.markdown(f"<div class='source-info'>{source_info}</div>", unsafe_allow_html=True)
                
                # Show word count badge
                st.markdown(f"<div class='word-badge'>📝 {word_count} words</div>", unsafe_allow_html=True)

                # Save to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "source_info": source_info,
                    "word_count": word_count
                })

            except Exception as e:
                st.error("An error occurred while processing your request.")
                st.exception(e)
