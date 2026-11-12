# deepseek_api.py
import requests
import logging
from typing import Optional

class DeepSeekAPI:
    def __init__(self, api_key: str):
        """
        Initialize DeepSeek API wrapper
        
        Args:
            api_key: Your DeepSeek API key
        """
        self.base_url = "https://api.deepseek.com/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.timeout = 10  # seconds

    def generate(self, prompt: str, model: str = "deepseek-chat") -> Optional[str]:
        """
        Generate text completion using DeepSeek API
        
        Args:
            prompt: Input text prompt
            model: Model to use (default: deepseek-chat)
            
        Returns:
            Generated text or None if failed
        """
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 200
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            logging.error(f"DeepSeek API request failed: {str(e)}")
        except (KeyError, ValueError) as e:
            logging.error(f"Failed to parse API response: {str(e)}")
        return None