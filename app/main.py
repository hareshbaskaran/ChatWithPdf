from fastapi import FastAPI, File, UploadFile, HTTPException
from services.llms import GeminiLLMProvider
from utils.helpers import PDFIngest, parse_to_pydantic
from langchain.chains import RetrievalQA
from app.utils.models import PDFUploadResponse, ChatResponse
from langchain.indexes import index
from utils.prompts import qa_prompt_template, response_prompt_template
from langchain.chains import LLMChain
from langchain_core.output_parsers import PydanticOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from fastapi import Form
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from typing import List
import os

app = FastAPI()
ingest = PDFIngest()


@app.post("/")
def health_check():
    return {"status": "OK"}


@app.post("/upload-pdf", response_model=PDFUploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    # create virtual temp-file path
    temp_file_path = await ingest.handle_temp_dir(file)

    try:
        # Process the PDF
        docs = ingest.process_doc(doc_path=temp_file_path)

        # get vector database
        vector_store = ingest.get_vector_store()

        ### add to index
        idx = index(
            docs_source=docs,
            record_manager=ingest.instantiate_record_manager(),
            vector_store=vector_store.get_vdb(),
            cleanup=None,
            source_id_key="source",
        )

        # Determine if the document is a duplicate
        is_dup = ingest.is_duplicate(idx, docs)

        ## if no dupliactes present add docs to vector_db
        if not is_dup:
            vector_store.add_docs_to_vector_db(docs=docs)

        # Return the response
        return PDFUploadResponse(
            message="Document already uploaded"
            if ingest.is_duplicate
            else "PDF uploaded and processed successfully."
        )

    ##add exceptions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

    finally:
        # clean up temp files
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.post("/chat-with-pdf")
async def chat_with_pdf(query: str = Form(...)):
    """
    Chat With Vector Store retrievers
    Implemented QARetrieval Chain to translate Similarity documents
    :to: Pydantic Output of LLM Response

    :param query:
    :return: {
    response : LLM Response (str),
    citations : Source Documents used for LLM Response(List[str])
    }
    """

    # instantiate LLM / Vector Stores
    llm = GeminiLLMProvider.get_llm()
    vector_store = ingest.get_vector_store()

    # Construct QARetrieval Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.get_vdb_as_retriever(),
        return_source_documents=True,
    )

    # invoke results and source details
    result = qa_chain({"query": query})

    # Parse to Pydantic JSON object
    return parse_to_pydantic(result)


@app.post("/chat-with-pdf:latest")
async def chat_with_pdf(query: str = Form(...)):
    """
    Chat With Vector Store retrievers
    Implemented QA Retrieval Chain to translate Similarity documents
    :to: Pydantic Output of LLM Response

    :param query: User query
    :return: {
        response: LLM Response (str),
        citations: Source Documents used for LLM Response (List[str])
    }
    """

    # Step 1: Instantiate LLM / VectorStore / Retriever
    llm = GeminiLLMProvider.get_llm()
    vector_store = ingest.get_vector_store()
    retriever = vector_store.get_vdb_as_retriever()

    # Step 2 : Invoke Retriever and LLM Prompts
    ## MultiQueryRetriever Prompt -> to return List[Documents]
    prompt = PromptTemplate(
        template=qa_prompt_template, input_variables=["query", "documents"]
    )

    ## LLMChain Prompt -> to return Query response in Pydantic Model
    response_prompt = PromptTemplate(
        template=response_prompt_template, input_variables=["query", "documents"]
    )

    # Step 3: Retrieve documents from Vector Store using MultiQueryRetriever

    question = query ## todo : change the question for relevant document search in future
    retrieved_docs: List[Document] = MultiQueryRetriever.from_llm(
        llm=llm,
        retriever=retriever
        # prompt=prompt
    ).get_relevant_documents(question)


    # Step 4 : Translate Retrieved Documents to Text -> Pass to LLM Prompt Template

    ## GPT Version -> convert to proper text translation
    documents_text = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
        for doc in retrieved_docs
    ) ## todo : change documents to text format -> create updated format helpers


    # Step 4: Call OuptutParser by a Pydantic Object -> ChatResponse
    output_parser = PydanticOutputParser(pydantic_object=ChatResponse)


    # Step 5: Run the LLM chain with the refined prompt
    chain = LLMChain(
        llm=llm,
        prompt=response_prompt,
        output_parser=output_parser

    )
    return chain.run({"query": query, "documents": documents_text})



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
