from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from app.utils.variables import SQL_MANAGER_NAMESPACE, SQLITE_DB_URL
from services.llms import GeminiLLMProvider
from utils.helpers import (
    handle_temp_dir,
    instantiate_record_manager,
    process_doc,
    get_vector_store,
    is_duplicate,
)
from langchain.chains import RetrievalQA
from app.utils.models import PDFUploadResponse, QAResponse
from langchain.indexes import index
import os

app = FastAPI()

@app.post("/")
def health_check():
    return {
        "status" : "OK"
    }

@app.post("/upload-pdf", response_model=PDFUploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    ## Instantiate file_path / record_manager
    temp_file_path = await handle_temp_dir(file)

    record_manager = instantiate_record_manager(
        db_url=SQLITE_DB_URL, namespace=SQL_MANAGER_NAMESPACE
    )

    try:
        # Process the PDF
        docs = process_doc(temp_file_path)

        # get vector database
        vector_store = get_vector_store()

        ### add to index

        idx = index(
            docs,
            record_manager,
            vector_store=vector_store.get_vdb(),
            cleanup=None,
            source_id_key="source"
        )

        # Determine if the document is a duplicate
        is_dup = is_duplicate(idx, docs)

        ## if no dupliactes present add docs to vector_db
        if not is_dup:
            vector_store.add_docs_to_vector_db()

        # Return the response
        return PDFUploadResponse(
            message="Document already uploaded"
            if is_duplicate
            else "PDF uploaded and processed successfully."
        )

        ##todo: add display upload methods

    ## add extensions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

    finally:
        # clean up temp files
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.post("/")
async def chat_with_pdf_lc(
        query: str = Form(...)
):
    """

    :param query:
    :return:
    """

    llm_provider = GeminiLLMProvider()
    llm = llm_provider.get_llm()
@app.post("/chat-with-pdf-qaretrival")
async def chat_with_pdf(query: str = Form(...)):
    """
    Chat with Vector DB
    :param query:
    :return:
    """
    # instantiate LLM
    llm_provider = GeminiLLMProvider()
    llm = llm_provider.get_llm()
    # Load the vector store and retrieve the retriever
    vector_store = get_vector_store()
    retriever = vector_store.get_vdb_as_retriever()


    ### retriver analysis

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    result = qa_chain(
        {"query":query}
    )

    sources = [doc.metadata.get("source") for doc in result["source_documents"]]

    parsed_result = {
        "response" : result["result"],
        "citations" : list(set(sources))
    }
    # Print the parsed result
    print(parsed_result)
    return parsed_result




if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
