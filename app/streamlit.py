from langchain_core.messages import HumanMessage
import streamlit as st
import requests
import json
import logging

FASTAPI_URL = "http://0.0.0.0:8000"

st.set_page_config(page_title="Chat with PDFs", layout="wide")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

tab1, tab2 = st.tabs(["📂 Upload PDF", "💬 Query PDF"])

with tab1:
    st.header("Upload PDF Document")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    domain = st.text_input("Enter the domain (e.g., Science, Finance, Legal, Health)")

    if st.button("Upload"):
        if uploaded_file and domain:
            files = {"file": uploaded_file.getvalue()}
            data = {"domain": domain}
            response = requests.post(
                f"{FASTAPI_URL}/upload-pdf", files=files, data=data
            )

            if response.status_code == 200:
                st.success(response.json())
            else:
                st.error(f"Error: {response.json()}")
        else:
            st.warning("Please upload a PDF and provide a domain.")

with tab2:
    st.header("Chat with PDF")
    query = st.text_area("Enter your query:")

    if st.button("Get Response"):
        if query:
            messages_payload = [
                {"type": "human", "content": msg.content}
                for msg in st.session_state.messages
            ]

            messages_payload.append({"type": "human", "content": query})
            request_payload = json.dumps(messages_payload)

            logging.info(f"Sending request: {request_payload}")
            response = requests.post(
                f"{FASTAPI_URL}/chat-with-pdf:latest", data={"query": request_payload}
            )

            if response.status_code == 200:
                chat_response = response.json()
                messages_payload.append(
                    {"type": "ai", "content": chat_response["response"]}
                )
                st.session_state.messages = [
                    HumanMessage(msg["content"])
                    if msg["type"] == "human"
                    else HumanMessage(msg["content"])
                    for msg in messages_payload
                ]

                logging.info(f"Received response: {chat_response}")

                st.success("Response Received!")
                st.markdown(f"**Response:**\n{chat_response['response']}")

                if chat_response.get("citations"):
                    st.markdown("### 📖 Citations:")
                    for doc in chat_response["citations"]:
                        st.write(f"📄 {doc}")

                if chat_response.get("domain"):
                    st.markdown(f"### 🌍 Domain: **{chat_response['domain']}**")
            else:
                error_detail = response.json().get("detail", "Unknown error")
                logging.error(f"Error from API: {error_detail}")
                st.error(f"Error: {error_detail}")
        else:
            st.warning("Please enter a query.")
