############### LLM Prompts ######################

qa_prompt_template = """
    Answer the following query using only the documents given:

    Query: {query}

    {documents}

    Provide your response in the following format:
    - Answer: <answer>
    - Citations: [document IDs] 
    """
## todo: change qa_prompt_template for Multi Query Template

response_prompt_template = """
        For the given query:
        {query}

        From the following documents:
        {documents}

        Provide the necessary response and strictly cite only the used sources. 
        The output must be in the following JSON format:
        {{
            "response": "Your generated response here",
            "citations": ["Source 1", "Source 2", ...]
        }}
        """
