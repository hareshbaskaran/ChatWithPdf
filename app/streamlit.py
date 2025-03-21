import streamlit as st
import requests

# Define FastAPI server URL
FASTAPI_URL = "http://localhost:8000"

st.set_page_config(page_title="Chat with PDFs", layout="wide")

# Create Tabs: Upload PDF & Chat with PDF
tab1, tab2 = st.tabs(["📂 Upload PDF", "💬 Query PDF"])

# ---- 📂 UPLOAD PDF TAB ----
with tab1:
    st.header("Upload PDF Document")

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    domain = st.text_input("Enter the domain (e.g., Science, Finance, Legal, Health)")

    if st.button("Upload"):
        if uploaded_file and domain:
            # Send file to FastAPI
            files = {"file": uploaded_file.getvalue()}
            data = {"domain": domain}
            response = requests.post(f"{FASTAPI_URL}/upload-pdf", files=files, data=data)

            if response.status_code == 200:
                st.success(response.json())
            else:
                st.error(f"Error: {response.json()}")
        else:
            st.warning("Please upload a PDF and provide a domain.")

# ---- 💬 QUERY PDF TAB ----
with tab2:
    st.header("Chat with PDF")

    query = st.text_area("Enter your query:")
    if st.button("Get Response"):
        if query:
            response = requests.post(f"{FASTAPI_URL}/chat-with-pdf:latest", data={"query": query})

            if response.status_code == 200:
                chat_response = response.json()
                st.success("Response Received!")

                # Display response
                st.markdown(f"**Response:**\n{chat_response['response']}")

                # Display citations
                if chat_response.get("citations"):
                    st.markdown("### 📖 Citations:")
                    for doc in chat_response["citations"]:
                        st.write(f"📄 {doc}")

                # Display domain if available
                if chat_response.get("domain"):
                    st.markdown(f"### 🌍 Domain: **{chat_response['domain']}**")
            else:
                st.error(f"Error: {response.json()['detail']}")
        else:
            st.warning("Please enter a query.")

