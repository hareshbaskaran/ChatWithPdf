import json
from typing import List

from langchain.chains import LLMChain
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.indexes import SQLRecordManager, index
from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.vectorstores import FAISS, Chroma
from langchain_core.documents import Document
from pydantic import BaseModel
from services.chunkers import RTChunker
from services.embeddings import HFEmbeddings
from services.llms import GeminiLLMProvider
from services.loaders import PDFLoader
from services.vectordbs import FAISSVectorStore
from utils.variables import VECTOR_DB_PATH


#### indexes in lancghain
########### run pdf - ingestion ###############
def main():
    llm = GeminiLLMProvider.get_llm()
    collection_name = "test_index"
    namespace = f"FAISS/{collection_name}"
    record_manager = SQLRecordManager(
        namespace, db_url="sqlite:///record_manager_cache.sql"
    )
    record_manager.create_schema()

    pdf_loader = PDFLoader.get_docs(doc_path="docs/test_data/test_index2.pdf")
    doc_chunker = RTChunker(docs=pdf_loader)
    docs = doc_chunker.split_docs()

    vectorstore = Chroma.from_documents(
        documents=docs, embedding=HuggingFaceEmbeddings()
    )

    # Step 1: Define Pydantic schema
    class ResponseSchema(BaseModel):
        answer: str
        citations: List[int]

    # output_parser = PydanticOutputParser(pydantic_object=ResponseSchema)

    # Step 2: Define the prompt template
    prompt_template = """
    Answer the following query using only the documents given:

    Query: {query}

    {documents}

    Provide your response in the following format:
    - Answer: <answer>
    - Citations: [document IDs] 
    """

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["query", "documents"]
    )

    # Step 4: Retrieval step
    def retrieve_documents(query, top_k=3):
        retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": top_k}
        )
        results = retriever.get_relevant_documents(query)
        return {f"Doc {i + 1}": doc.page_content for i, doc in enumerate(results)}

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    query_retriever_chain = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(),
        llm=llm,
    )

    # Step 6: Execute query
    """def process_query(query):
        # Retrieve relevant docs
        docs = retrieve_documents(query)
        doc_text = "\n\n".join([f"{key}:\n{value}" for key, value in docs.items()])

        # Generate LLM response
        response = llm_chain.run({
            "query": query,
            "documents": doc_text
        })

        # Map citations to document names
        #doc_mapping = {i + 1: f"doc_{i + 1}.pdf" for i in range(len(docs))}
        #citations_with_names = [doc_mapping[citation] for citation in response.citations]
        return response"""

    """return {
            "answer": response.answer
            #"citations": citations_with_names
        }"""

    # Example usage
    query = "from any 10 different pds provide summary of each document"
    result = query_retriever_chain.invoke(query)
    print(result)

    ### process query ###

    llm_prompt_template = """
    Answer the following query using only the documents given:

    Query: {query}

    {documents}

    Provide your response in the following format:
    - Answer: <answer>
    - Citations: (unique source,page details from metadata of provided documents)
    """

    llm_prompt = PromptTemplate(
        template=llm_prompt_template,
        input_variables=["query", "documents"],
    )

    chain = LLMChain(llm=llm, prompt=llm_prompt)

    documents_json = [
        {
            "page_content": doc.page_content,
            "metadata": {
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page"),
            },
        }
        for doc in docs
    ]

    """# Serialize JSON to text
    documents_json_string = json.dumps(documents_json, indent=4)

    # Print the JSON string
    print("JSON Representation:\n", documents_json_string)"""

    # Convert JSON back to text format
    doc_text = "\n\n".join(
        f"Page Content:\n{doc['page_content']}\nSource: {doc['metadata']['source']}\nPage: {doc['metadata']['page']}"
        for doc in documents_json
    )

    # Print the text representation
    print("\nText Representation:\n", doc_text)

    # doc_text = "\n".join([doc.page_content for doc in docs])

    response = llm_chain.run({"query": query, "documents": doc_text})
    print(response)


if __name__ == "__main__":
    main()

####### working code #########
