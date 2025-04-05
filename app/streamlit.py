from langchain_core.messages import HumanMessage
import streamlit as st
import requests
import json
import logging
import os
import tempfile
import shutil

FASTAPI_URL = "http://fastapi:8000"

st.set_page_config(page_title="Chat with PDFs", layout="wide")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

tab1, tab2, tab3, tab4 = st.tabs(["📂 Upload PDF", "💬 Query PDF", "📄 Upload with Bib", "📁 Bulk Upload"])

# 📂 Upload Single PDF
with tab1:
    st.header("Upload PDF Document")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    domain = st.text_input("Enter the domain (e.g., Science, Finance, Legal, Health)")

    if st.button("Upload"):
        if uploaded_file and domain:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            data = {"domain": domain}
            response = requests.post(f"{FASTAPI_URL}/upload-pdf", files=files, data=data)

            if response.status_code == 200:
                st.success(response.json())
            else:
                st.error(f"Error: {response.json()}")
        else:
            st.warning("Please upload a PDF and provide a domain.")

# 💬 Query PDF
with tab2:
    st.header("Chat with PDF")
    query = st.text_area("Enter your query:")

    if st.button("Get Response"):
        if query:
            messages_payload = st.session_state.get('messages', [])
            messages_payload.append({"role": "human", "content": query})
            request_payload = json.dumps(messages_payload)

            logging.info(f"Sending request: {request_payload}")
            response = requests.post(f"{FASTAPI_URL}/chat-with-pdf:latest", data={"query": request_payload})

            if response.status_code == 200:
                chat_response = response.json()
                messages_payload.append({"role": "ai", "content": chat_response["response"]})
                st.session_state.messages = messages_payload

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

# 📄 Upload PDF + BibTeX
with tab3:
    st.header("Upload PDF with BibTeX")
    uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"], key="pdf")
    uploaded_bib = st.file_uploader("Upload a BibTeX file", type=["bib"], key="bib")
    domain_bib = st.text_input("Enter the domain (e.g., Science, Finance, Legal, Health)", key="domain_bib")

    if st.button("Upload with Bib"):
        if uploaded_pdf and uploaded_bib and domain_bib:
            files = {
                "file": (uploaded_pdf.name, uploaded_pdf.getvalue(), "application/pdf"),
                "bib_file": (uploaded_bib.name, uploaded_bib.getvalue(), "application/x-bibtex"),
            }
            data = {"domain": domain_bib}
            response = requests.post(f"{FASTAPI_URL}/upload", files=files, data=data)

            if response.status_code == 200:
                st.success(response.json())
            else:
                st.error(f"Error: {response.json()}")
        else:
            st.warning("Please upload a PDF, a BibTeX file, and provide a domain.")

# 📁 Bulk Upload (Drag & Drop Folder)
with tab4:
    st.header("Bulk Upload PDFs & BibTeX")
    root_dir = st.file_uploader("Upload a zip file containing domain folders", type=["zip"])

    if root_dir and st.button("Process Directory"):
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, "uploaded.zip")

            with open(zip_path, "wb") as f:
                f.write(root_dir.getvalue())

            temp_path = os.path.join(temp_dir, "extracted")

            shutil.unpack_archive(zip_path, temp_path)

            st.write(f"Processing directory: `{temp_path}`")

            domain_dirs = [d for d in os.listdir(temp_path) if os.path.isdir(os.path.join(temp_path, d))]
            uploaded_files = []

            progress_bar = st.progress(0)
            total_domains = len(domain_dirs)

            for i, domain in enumerate(domain_dirs):
                domain_path = os.path.join(temp_path, domain)
                pdf_files = [f for f in os.listdir(domain_path) if f.endswith(".pdf")]
                bib_files = {f[:-4]: os.path.join(domain_path, f) for f in os.listdir(domain_path) if
                             f.endswith(".bib")}

                st.write(f"Processing domain: `{domain}` ({len(pdf_files)} PDFs found)")

                for pdf in pdf_files:
                    pdf_path = os.path.join(domain_path, pdf)
                    pdf_basename = pdf[:-4]
                    bib_path = bib_files.get(pdf_basename)

                    if bib_path:
                        with open(pdf_path, "rb") as pdf_file, open(bib_path, "rb") as bib_file:
                            files = {
                                "file": (pdf, pdf_file, "application/pdf"),
                                "bib_file": (os.path.basename(bib_path), bib_file, "application/x-bibtex"),
                            }
                            data = {"domain": domain}

                            response = requests.post(f"{FASTAPI_URL}/upload", files=files, data=data)

                            if response.status_code == 200:
                                uploaded_files.append(f"✅ {domain}/{pdf} uploaded successfully")
                            else:
                                uploaded_files.append(f"❌ {domain}/{pdf} failed to upload: {response.text}")
                    else:
                        uploaded_files.append(f"⚠️ Skipping {domain}/{pdf} (No matching BibTeX found)")

                progress_bar.progress((i + 1) / total_domains)

            st.write("### Upload Summary")

            successful = sum(1 for status in uploaded_files if "✅" in status)
            failed = sum(1 for status in uploaded_files if "❌" in status)
            skipped = sum(1 for status in uploaded_files if "⚠️" in status)

            st.write(f"Total: {len(uploaded_files)} files processed")
            st.write(f"- {successful} files uploaded successfully")
            st.write(f"- {failed} files failed to upload")
            st.write(f"- {skipped} files skipped (missing BibTeX)")

            with st.expander("View detailed upload results"):
                for file_status in uploaded_files:
                    st.write(file_status)
