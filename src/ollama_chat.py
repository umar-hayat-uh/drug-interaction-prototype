import streamlit as st
import requests
import json
import time

OLLAMA_API = "http://localhost:11434/api/generate"
MODEL = "llama3"

def ask_ollama_stream(prompt):
    headers = {"Content-Type": "application/json"}
    payload = {"model": MODEL, "prompt": prompt, "stream": True}

    with requests.post(OLLAMA_API, headers=headers, json=payload, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            if isinstance(line, bytes):
                line = line.decode("utf-8")
            line = line.strip()
            if line.startswith("data: "):
                line = line[len("data: "):]
            try:
                data = json.loads(line)
                if "response" in data:
                    yield data["response"]  # append full chunk as-is
            except json.JSONDecodeError:
                continue

def ollama_chat_ui():
    st.title("🤖 AI Chatbot (Prototype)")
    st.markdown("Chat with a local LLM about drugs. Prototype only — not medical advice.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "stop_stream" not in st.session_state:
        st.session_state.stop_stream = False

    # Display past chat
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a question about a drug...")
    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Reset stop flag
        st.session_state.stop_stream = False

        # Interrupt button
        if st.button("🛑 Stop Response"):
            st.session_state.stop_stream = True

        # Placeholder for AI message
        bot_msg = st.chat_message("assistant")
        msg_placeholder = bot_msg.empty()
        full_text = ""

        # Stream response without splitting by words
        for chunk in ask_ollama_stream(user_input):
            if st.session_state.stop_stream:
                full_text += "\n\n*Response stopped by user.*"
                msg_placeholder.markdown(full_text)
                st.session_state.chat_history.append({"role": "assistant", "content": full_text})
                break

            full_text += chunk  # append chunk as-is
            msg_placeholder.markdown(full_text)
            time.sleep(0.001)  # smooth typing effect

        # Save full response if not interrupted
        if not st.session_state.stop_stream:
            st.session_state.chat_history.append({"role": "assistant", "content": full_text})
