from pydantic import BaseModel
from typing import List


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
    response: str
    citations: list[str]


class ChatResponseTest(BaseModel):
    response: str
    citation: str
