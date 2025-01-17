from pydantic import BaseModel


class PDFUploadResponse(BaseModel):
    message: str

class LLMResponse(BaseModel):
    response: str
