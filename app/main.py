import os
import pprint
from typing import List, Optional, Type
import json
import bibtexparser
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from langchain.chains import LLMChain, RetrievalQA
from langchain.indexes import index
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import Document
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate


from utils.prompts import response_prompt_template
from services.llms import GeminiLLMProvider
from settings import settings
from utils.helpers import ChatService, convert_docs_to_text, parse_to_pydantic

from utils.loggers import logger
from utils.models import ChatResponse, PDFUploadResponse, PDFBibUploadResponse

chat = ChatService(
    llm=settings.get("LLM"),
    embeddings=settings.get("EMBEDDINGS"),
    vectorstore=settings.get("VECTOR_STORE"),
)

route = FastAPI()


@route.get("/")
async def root():
    """API health check endpoint."""
    logger.info("API is running")
    return {"status": "API Running"}



@route.post("/upload", response_model=PDFBibUploadResponse)
async def upload(
        file: UploadFile = File(...),
        bib_file: UploadFile = File(...),
        domain: Optional[str] = Form(...)):
    """Upload and process a PDF document with an additional 'domain' field."""
    logger.info(f"Upload PDF Endpoint is starting for domain: {domain}")
    # Handle PDF file
    temp_file_path = await chat.handle_temp_dir(file)

    try:
        # Read and parse the BibTeX file
        bib: str = (await bib_file.read()).decode("utf-8")  # Convert bytes to string
        bib_data = bibtexparser.loads(bib)

        # Extract BibTeX metadata
        bib_entries = bib_data.entries if hasattr(bib_data, "entries") else []
        bib_metadata = {entry.get("ID", f"entry_{i}"): entry for i, entry in enumerate(bib_entries)}

        # Process the PDF
        docs = chat.process_pdfs(doc_path=temp_file_path)

        # Annotate documents with additional metadata (domain & BibTeX data)
        for doc in docs:
            doc.metadata["domain"] = domain

            # ✅ Add each BibTeX field as an individual metadata key-value pair
            for entry_id, entry_data in bib_metadata.items():
                for key, value in entry_data.items():
                    if key.lower() != "id":  # Skip ID as it's redundant
                        doc.metadata[f"bib_{key.lower()}"] = value  # Prefix `bib_` for clarity

        # Process document indexing
        vector_store = chat.get_vector_store()
        idx = index(
            docs_source=docs,
            record_manager=chat.instantiate_record_manager(),
            vector_store=vector_store.get_vdb(),
            cleanup=None,
            source_id_key="source",
        )

        # Handle duplicate check & add to vector store
        is_duplicate, response = chat.process_duplicate_doc(idx, docs)
        if not is_duplicate:
            vector_store.add_docs_to_vector_db(docs=docs)

        logger.info(f"Paper Uploaded and Processed successfully for domain: {domain}")

        return PDFBibUploadResponse(
            message=response,
            bib_metadata=bib_metadata
        )

    except Exception as e:
        logger.error(f"Error in upload_pdf: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info("Cleaned up temporary files")

            
@route.post("/upload-pdf", response_model=PDFUploadResponse)
async def upload_pdf(
        file: UploadFile = File(...),
        domain: Optional[str] = Form(...)):
    """Upload and process a PDF document with an additional 'domain' field."""
    logger.info(f"Upload PDF Endpoint is starting for domain: {domain}")
    temp_file_path = await chat.handle_temp_dir(file)

    try:
        docs = chat.process_pdfs(doc_path=temp_file_path)

        # Annotate documents with additional metadata -> domain , etc..
        for doc in docs:
            doc.metadata["domain"] = domain  # Store domain in metadata

        print(docs[-1])

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

        logger.info(f"Paper Uploaded and Processed successfully for domain: {domain}")
        return PDFUploadResponse(message=response)

    except Exception as e:
        logger.error(f"Error in upload_pdf: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info("Cleaned up temporary files")


@route.post("/chat-with-pdf:latest")
async def chat_with_pdf_latest(query: str = Form(...)):
    """Updated Endpoint to Retrieve Relevant Documents and Process User Query
    version_fix -> Retrieve Unique Document Citation from LLM
    :param query:
    :return: LLM Response
    """
    logger.info("Chat with PDF latest endpoint starting")
    try:
        llm = GeminiLLMProvider.get_llm()
        vector_store = chat.get_vector_store()

        retrieved_docs = MultiQueryRetriever.from_llm(
            llm=llm, retriever=vector_store.get_vdb_as_retriever()
        ).get_relevant_documents(query)

        parsed_docs = {
            i+1 : {"content": doc.page_content, "metadata": doc.metadata}
            for i,doc in enumerate(retrieved_docs)
        }
        logger.debug("**** PARSED DOCUMENTS ********* \n\n")
        logger.debug(parsed_docs)

        input_docs = {
            doc_id: data["content"]
            for doc_id, data in parsed_docs.items()
        }
        logger.debug("**** INPUT DOCUMENTS ********* \n\n")
        logger.debug(input_docs)


        prompt = PromptTemplate(
            template=response_prompt_template,
            input_variables=["query", "documents", "parser_information"],
        )
        chain = LLMChain(
            llm=llm,
            prompt=prompt,
            output_parser=PydanticOutputParser(pydantic_object=ChatResponse),
        )

        response = chain.run(
            {
                "query": query,
                "documents": str(input_docs),
                "parser_information": PydanticOutputParser(
            pydantic_object=ChatResponse
        ).get_format_instructions(),
            }
        )
        chat_response = ChatResponse.model_validate(response)
        logger.info("**** CHAT RESPONSE ********* \n\n")
        logger.info(chat_response.model_dump_json())

        citations = {}

        for doc_id in chat_response.doc_ids:
            if doc_id not in parsed_docs:
                raise ValueError("Invalid Response of Unique ID")

            parsed_doc = parsed_docs[doc_id]
            source = parsed_doc["metadata"]["source"]
            domain = parsed_doc["metadata"]["domain"]

            if source not in citations:
                citations[source] = domain

        parsed_citation = [
            {"source": citation, "domain": citations[citation]}
            for citation in citations
        ]
        return {
            "response": chat_response.response,
            "citations": parsed_citation
        }

    except Exception as e:
        logger.error(f"Error in chat_with_pdf_latest: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Unicorn server")
    uvicorn.run(route, host="0.0.0.0", port=8000)
    logger.info("FastApi Endpoint is Running Successfully")


#faiss-cpu
#torch
#lark==1.1.5
#onnx==1.16.1
#langchain-experimental


"""
transformers
sentence-transformers
opencv-python

docker compose build --no-cache
docker compose up -d
"""