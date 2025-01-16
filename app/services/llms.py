from langchain_google_genai import ChatGoogleGenerativeAI
from app.utils.env_loader import GOOGLE_API_KEY
from abc import ABC, abstractmethod
from langchain_core.language_models import BaseChatModel


class BaseLLMProvider(ABC):
    provider_name: str
    model_name: str

    @abstractmethod
    def get_llm(self) -> BaseChatModel:
        """
        This method should be implemented by concrete LLM providers to return
        an instance of a specific LLM (Language Model) class.
        """
        pass


class GeminiLLMProvider(BaseLLMProvider):
    provider_name = "Gemini"
    model_name = "gemini-1.5-flash"

    def get_llm(self) -> ChatGoogleGenerativeAI:
        """
        This method returns an instance of the ChatGoogleGenerativeAI model.
        """
        return ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=0.7,
            top_p=0.85,
            google_api_key=GOOGLE_API_KEY,
            tokenize=1028,
        )