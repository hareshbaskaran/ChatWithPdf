from pydantic import BaseModel
from typing import List


############# Response models ##############

class PDFUploadResponse(BaseModel):
    message: str

class LLMResponse(BaseModel):
    response: str

class QAResponse(BaseModel):
    response: str
    citations: List[str]