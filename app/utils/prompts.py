from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    PromptTemplate,
)

############### LLM Prompt Templates ######################

query_retriever_prompt = """You are an AI language model assistant. Your task is 
    to generate 5 different versions of the given user 
    question to retrieve relevant documents from a vector  database. 
    By generating multiple perspectives on the user question, 
    your goal is to help the user overcome some of the limitations 
    of distance-based similarity search. Provide these alternative 
    questions separated by newlines. Original question: {question}"""

response_prompt_template = """
    You are an AI that answers user questions based on provided documents.

    For the given query:
    {query}

    From the following documents:
    {documents}

    Follow these strict rules:
    1. **Use only the given documents** to generate the response.
    2. **Cite only the sources actually used** in the response.
    3. **Extract `domain` strictly from metadata['domain']** of the cited documents.
    4. **Ensure `domain` is always a valid string (never null).** 
       - If multiple documents are cited, choose the most relevant domain.
       - If no domain is available, use `"Unknown"` instead of `null`.
    5. The output must be **strictly in JSON format**, following this schema:

    {{
        "response": "Your generated response here",
        "citations": List[str],  # Unique document source names strictly from cited documents
        "domain": str  # Extracted strictly from metadata['domain'], must never be null
    }}

    Now, generate the structured JSON response.
"""

response_prompt = PromptTemplate(
    template=response_prompt_template,

    input_variables=["query", "documents"]
)
