from typing import List, Optional
from pydantic import BaseModel, field_validator, Field

############# Response models ##############


class PDFUploadResponse(BaseModel):
    """
    Pydantic Model for Validating PDF Upload Response

    """

    message: str

class PDFBibUploadResponse(BaseModel):
    message: str
    bib_metadata: Optional[dict] = None


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

    response: str = Field(description="Generated response based on provided documents and user query")
    doc_ids: List[int] = Field(description="List of document IDs used to generate the response")



