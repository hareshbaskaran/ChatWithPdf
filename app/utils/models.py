from typing import List

from pydantic import BaseModel

############# Response models ##############


class PDFUploadResponse(BaseModel):
    """
    Pydantic Model for Validating PDF Upload Response

    """

    message: str


class LLMResponse(BaseModel):
    """
    Pydantic Model for Validating LLM Response

    """

    response: str


class QAResponse(BaseModel):
    """
    Pydantic Model for Validating QA Response from LLM/Retrieval Chain

    """

    response: str
    citations: List[str]


class ChatResponse(BaseModel):
    """
    Pydantic Model for Validating chat Response linked with Source Citations
    """

    response: str
    citations: list[str]


class ChatResponseTest(BaseModel):
    response: str
    citation: str
