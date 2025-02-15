import streamlit as st
import requests
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)

# Fetch available models from Ollama
def get_available_models():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model["name"] for model in models] if models else ["deepseek-r1:1.5b"]
        return ["deepseek-r1:1.5b"]  # Fallback option
    except requests.exceptions.RequestException:
        return ["deepseek-r1:1.5b"]  # Fallback if server is unavailable

# Streamlit UI
st.title("🧠 Nasha Ai Code Companion")
st.caption("🚀 Your AI Pair Programmer with Debugging Superpowers")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    available_models = get_available_models()
    selected_model = st.selectbox("Choose Model", available_models, index=0)
    st.divider()
    st.markdown("### Model Capabilities")
    st.markdown("""
    - 🐍 Python Expert
    - 🐞 Debugging Assistant
    - 📝 Code Documentation
    - 💡 Solution Design
    """)
    st.divider()
with st.sidebar:
    st.markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)")
    
    # Stylish footer below the "Built with Ollama & LangChain"
    st.markdown(
        """
        <style>
            .footer {
                text-align: center;
                font-size: 14px;
                color: gray;
                font-family: 'Arial', sans-serif;
                margin-top: 10px;
                padding-top: 10px;
                border-top: 1px solid rgba(255, 255, 255, 0.2); /* Subtle separator */
            }
            .footer a {
                text-decoration: none;
                font-weight: bold;
                padding: 5px;
                border-radius: 5px;
                transition: all 0.3s ease-in-out;
            }
            .github-link {
                color: #32CD32; /* Green */
            }
            .github-link:hover {
                background: #32CD32;
                color: white;
            }
            .portfolio-link {
                color: #FFD700; /* Gold */
            }
            .portfolio-link:hover {
                background: #FFD700;
                color: black;
            }
        </style>
        <div class="footer">
            Developed by 
            <a href="https://github.com/shafi-1234" target="_blank" class="github-link">
                Mahammad Shafi
            </a> |
            <a href="https://mahammadshafiportfolio.netlify.app/" target="_blank" class="portfolio-link">
                Portfolio
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )


# Initialize the chat engine
llm_engine = ChatOllama(
    model=selected_model,
    base_url="http://localhost:11434",
    temperature=0.3
)

# System prompt configuration
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an expert AI coding assistant. Provide concise, correct solutions "
    "with strategic print statements for debugging. Always respond in English."
)

# Initialize session state
if "message_log" not in st.session_state:
    st.session_state.message_log = [
        {"role": "ai", "content": "Hi! I'm DeepSeek. How can I help you code today? 💻"}
    ]

# Display chat messages
for message in st.session_state.message_log:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_query = st.chat_input("Type your coding question here...")

# Function to generate AI response
def generate_ai_response(prompt_chain):
    processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
    return processing_pipeline.invoke({})  # Ensure correct invocation

# Function to build the chat prompt chain
def build_prompt_chain():
    prompt_sequence = [system_prompt]
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

# Process user input
if user_query:
    # Append user message to chat history
    st.session_state.message_log.append({"role": "user", "content": user_query})

    # Generate AI response
    with st.spinner("🧠 Processing..."):
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)

    # Append AI response to chat history
    st.session_state.message_log.append({"role": "ai", "content": ai_response})

    # Rerun to update UI
    st.rerun()
