import os
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from langchain.chains import LLMChain, RetrievalQA
from langchain.indexes import index
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import Document
from langchain_core.output_parsers import PydanticOutputParser
from services.llms import GeminiLLMProvider
from utils.enums import settings
from utils.helpers import ChatService, convert_docs_to_text, parse_to_pydantic
from utils.prompts import response_prompt

from app.utils.models import ChatResponse, PDFUploadResponse
from app.utils.loggers import logger

app = FastAPI()


@app.get("/")
async def root():
    """API health check endpoint."""
    logger.info("Root endpoint accessed.")
    return {"status": "API Running"}


@app.post("/upload-pdf", response_model=PDFUploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF document."""
    logger.info("Upload PDF endpoint accessed.")
    temp_file_path = await chat.handle_temp_dir(file)

    try:
        docs = chat.process_pdfs(doc_path=temp_file_path)
        vector_store = chat.get_vector_store()

        idx = index(
            docs_source=docs,
            record_manager=chat.instantiate_record_manager(),
            vector_store=vector_store.get_vdb(),
            cleanup=None,
            source_id_key="source",
        )

        is_duplicate, response = chat.process_duplicate_doc(idx, docs)
        if not is_duplicate:
            vector_store.add_docs_to_vector_db(docs=docs)

        logger.info("PDF processed successfully.")
        return PDFUploadResponse(message=response)

    except Exception as e:
        logger.error(f"Error in upload_pdf: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info("Temporary file cleaned up.")


@app.post("/chat-with-pdf")
async def chat_with_pdf(query: str = Form(...)):
    """
    EndPoint to Interact with PDF
    :param query:
    :return: LLM Response
    """
    logger.info("Chat with PDF endpoint accessed.")
    try:
        llm = GeminiLLMProvider.get_llm()
        vector_store = chat.get_vector_store()

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vector_store.get_vdb_as_retriever(),
            return_source_documents=True,
        )

        result = qa_chain({"query": query})
        logger.info("Query processed successfully.")
        return parse_to_pydantic(result)

    except Exception as e:
        logger.error(f"Error in chat_with_pdf: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/chat-with-pdf:latest")
async def chat_with_pdf_latest(query: str = Form(...)):
    """ Updated Endpoint to Retrieve Relevant Documents and Process User Query
    version_fix -> Retrieve Unique Document Citation from LLM
    :param query:
    :return: LLM Response
    """
    logger.info("Chat with PDF latest endpoint accessed.")
    try:
        llm = GeminiLLMProvider.get_llm()
        vector_store = chat.get_vector_store()

        retrieved_docs: List[Document] = MultiQueryRetriever.from_llm(
            llm=llm,
            retriever=vector_store.get_vdb_as_retriever(),
        ).get_relevant_documents(query)

        chain = LLMChain(
            llm=llm,
            prompt=response_prompt,
            output_parser=PydanticOutputParser(pydantic_object=ChatResponse),
        )

        response = chain.run(
            {"query": query, "documents": convert_docs_to_text(retrieved_docs)}
        )
        logger.info("Query processed successfully with advanced retriever.")
        return response

    except Exception as e:
        logger.error(f"Error in chat_with_pdf_latest: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    logger.info("Initializing ChatService.")
    chat = ChatService(
        llm=settings.get("LLM"),
        embeddings=settings.get("EMBEDDINGS"),
        vectorstore=settings.get("VECTOR_STORE"),
    )

    logger.info("Starting FastAPI server.")
    uvicorn.run(app, host="127.0.0.1", port=8000)
