from RAG_ChatBot import ChatBot
import streamlit as st
from PIL import Image

# Initialize chatbot
bot = ChatBot()

# Load logo image
logo_image = Image.open("logo_ermai.png")

# Page config with custom icon
st.set_page_config(
    page_title="ERM Tutor AI",
    page_icon=logo_image,
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
        body {
            background-color: #e6f0ff;
        }
        .stApp {
            background-color: #e6f0ff;
            color: #003366;
        }
        .css-18ni7ap.e8zbici2 {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 10px;
            border: 1px solid #cce0ff;
        }
        .stChatMessage {
            background-color: #f0f8ff;
            border-radius: 10px;
            margin-bottom: 10px;
            padding: 10px;
            color: #003366;
        }
        .stTextInput>div>div>input {
            background-color: #ffffff;
            color: #003366;
            border: 1px solid #cce0ff;
        }
        .stMarkdown h1 {
            color: #003366;
        }
    </style>
""", unsafe_allow_html=True)

# Display logo
st.image(logo_image, width=150)

# Copyright below logo — force one line with no wrapping or clipping
st.markdown(
    """
    <div style='width: 150px; text-align: center; font-size: 0.63em; color: #777777; white-space: nowrap; margin-top: 5px; margin-bottom: 20px;'>
        © 2025 Amrina. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)


# Title and description
st.title("Hello! I'm your AI tutor for Empirical Research Methods (ERM) course.")
st.markdown(
    "Ask me anything about the **Empirical Research Methods (ERM)** course. "
    "I’ll give short, reliable answers based on your course materials — and not just that. "
    "As your AI tutor, I’m also here to guide you, reflect on your questions, and support your learning journey like a real mentor would. 😊"
)

# Chat history setup
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I assist you today?"}
    ]

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Type your question here...")

# On new user input
if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Thinking like a researcher..."):
            try:
                response = bot.chat(user_input)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error("An error occurred while processing your request.")
                st.exception(e)
