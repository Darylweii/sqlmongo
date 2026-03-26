"""
vLLM Client Wrapper for LangChain

This module provides a custom LangChain LLM wrapper that uses requests
directly to communicate with vLLM servers, avoiding compatibility issues
with the OpenAI Python client.
"""

from typing import Any, List, Optional, Dict
import requests
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class VLLMChatModel(BaseChatModel):
    """
    Custom LangChain chat model for vLLM servers.
    
    Uses requests directly to avoid OpenAI client compatibility issues.
    """
    
    base_url: str
    model: str
    api_key: str = "vllm"
    temperature: float = 0.0
    max_tokens: int = 2000
    timeout: int = 30
    
    @property
    def _llm_type(self) -> str:
        return "vllm"
    
    def _convert_messages(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """Convert LangChain messages to OpenAI format."""
        result = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                result.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                result.append({"role": "system", "content": msg.content})
            else:
                # Fallback for other message types
                result.append({"role": "user", "content": str(msg.content)})
        return result
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat completion using vLLM API."""
        
        # Convert messages
        api_messages = self._convert_messages(messages)
        
        # Prepare request payload
        payload = {
            "model": self.model,
            "messages": api_messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        
        if stop:
            payload["stop"] = stop
        
        # Make request
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Extract response
            content = result["choices"][0]["message"]["content"]
            
            # Create ChatGeneration
            generation = ChatGeneration(
                message=AIMessage(content=content),
                generation_info={
                    "finish_reason": result["choices"][0].get("finish_reason"),
                    "model": result.get("model"),
                },
            )
            
            return ChatResult(generations=[generation])
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"vLLM API request failed: {e}")
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Invalid vLLM API response format: {e}")
