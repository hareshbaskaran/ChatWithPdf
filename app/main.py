import os
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from langchain.chains import LLMChain, RetrievalQA
from langchain.indexes import index
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import Document
from langchain_core.output_parsers import PydanticOutputParser
from services.llms import GeminiLLMProvider
from utils.helpers import PDFIngest, parse_to_pydantic, convert_docs_to_text
from utils.prompts import response_prompt

from app.utils.models import ChatResponse, PDFUploadResponse

# Initialize FastAPI application / PDFIngest Service
app = FastAPI()
ingest = PDFIngest()


@app.get("/")
async def root():
    return {"status": "API Running"}


@app.post("/upload-pdf", response_model=PDFUploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Endpoint to upload a PDF document and process it.

    :param file: Uploaded PDF file.
    :return: Response indicating whether the upload and processing were successful.
    """
    # Create a virtual temporary file path
    temp_file_path = await ingest.handle_temp_dir(file)

    try:
        # Step 1: Process the PDF file
        docs = ingest.process_doc(doc_path=temp_file_path)

        # Step 2: Retrieve the vector database
        vector_store = ingest.get_vector_store()

        # Step 3: Add processed documents to the index
        idx = index(
            docs_source=docs,
            record_manager=ingest.instantiate_record_manager(),
            vector_store=vector_store.get_vdb(),
            cleanup=None,
            source_id_key="source",
        )

        # Step 4: Handle duplicates in a modularized way
        is_duplicate, response = ingest.process_duplicate_doc(idx, docs)

        # Step 5: If no duplicates are found, add documents to the vector store
        if not is_duplicate:
            vector_store.add_docs_to_vector_db(docs=docs)

        # Return the appropriate response
        return PDFUploadResponse(message=response)

    except Exception as e:
        # Handle exceptions and return error details
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

    finally:
        # Clean up temporary files
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.post("/chat-with-pdf")
async def chat_with_pdf(query: str = Form(...)):
    """
    Endpoint to interact with the vector store using a query and retrieve relevant documents.

    :param query: User query string.
    :return: Parsed response containing LLM results and citations.
    """
    # Step 1: Initialize LLM and vector store
    try:
        llm = GeminiLLMProvider.get_llm()
        vector_store = ingest.get_vector_store()

        # Step 2: Construct a QA Retrieval Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vector_store.get_vdb_as_retriever(),
            return_source_documents=True,
        )

        # Step 3: Invoke results and retrieve source details
        result = qa_chain({"query": query})

        # Step 4: Parse results into a Pydantic JSON object
        return parse_to_pydantic(result)

    except Exception as e:
        # Handle exceptions and return error details
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")



@app.post("/chat-with-pdf:latest")
async def chat_with_pdf_latest(query: str = Form(...)):
    """
    Latest endpoint to interact with the vector store and retrieve results.
    ###todo : generate specific templates for DB retriever Query from LLM-> question

    :param query: User query string.
    :return: Parsed response containing LLM results and citations.
    """
    try:
        # Step 1: Initialize LLM and vector store
        llm = GeminiLLMProvider.get_llm()
        vector_store = ingest.get_vector_store()

        # Step 2: Use MultiQueryRetriever to fetch relevant documents
        retrieved_docs: List[Document] = MultiQueryRetriever.from_llm(
            llm=llm,
            retriever=vector_store.get_vdb_as_retriever(),
        ).get_relevant_documents(query)

        # Step 3: Use LLM chain with the refined prompt and return the response
        chain = LLMChain(
            llm=llm,
            prompt=response_prompt,
            output_parser= PydanticOutputParser(pydantic_object=ChatResponse))

        return chain.run({"query": query, "documents":convert_docs_to_text(retrieved_docs)})

    except Exception as e:
        # Handle exceptions and return error details
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")



if __name__ == "__main__":
    import uvicorn

    # Run the FastAPI application
    uvicorn.run(app, host="127.0.0.1", port=8000)
