from typing import List, Optional
import uuid
from pydantic import BaseModel, field_validator

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
    Pydantic Model for Validating Chat Response linked with Source Citations.
    """

    response: str
    doc_ids: List[uuid.UUID]
    # citations: list[str]
    # domain : Optional[str] = None
    #
    # @classmethod
    # @field_validator("citations", mode="after")
    # def union_citations(cls, citations):
    #     return List[set(citations)]
