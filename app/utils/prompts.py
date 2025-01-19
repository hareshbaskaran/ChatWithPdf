from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate,PromptTemplate

############### LLM Prompt Templates ######################

query_retriever_prompt = """You are an AI language model assistant. Your task is 
    to generate 5 different versions of the given user 
    question to retrieve relevant documents from a vector  database. 
    By generating multiple perspectives on the user question, 
    your goal is to help the user overcome some of the limitations 
    of distance-based similarity search. Provide these alternative 
    questions separated by newlines. Original question: {question}"""

response_prompt_template = """
        For the given query:
        {query}

        From the following documents:
        {documents}

        Provide the necessary response and strictly cite only the used sources and unique document source names. 
        The output must be in the following JSON format:
        {{
            "response": "Your generated response here",
            "citations": Union["Source 1", "Source 2", ...]
        }}
        """

######## LLM Prompts ####################

response_prompt = PromptTemplate(
        template=response_prompt_template, input_variables=["query", "documents"]
    )