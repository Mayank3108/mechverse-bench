"""
Unified model wrapper for OpenRouter API.
Supports GPT-5.4, Claude Opus 4.6, and Gemini 3.1 Pro
through a single interface identical to GPT52Model.
"""

import os
import base64
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MODELS = {
    "gpt"    : "openai/gpt-5.4",
    "opus"   : "anthropic/claude-opus-4-6",
    "gemini" : "google/gemini-3.1-pro-preview",
}


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class OpenRouterModel:
    """
    Unified OpenRouter model wrapper for MechVerse assessment.
    """

    def __init__(self, model_key: str):
        if model_key not in MODELS:
            raise ValueError(f"model_key must be one of: {list(MODELS.keys())}")

        self.model_key  = model_key
        self.model_name = MODELS[model_key]

        self.client = OpenAI(
            base_url = "https://openrouter.ai/api/v1",
            api_key  = os.environ.get("OPENROUTER_API_KEY"),
        )

        print(f"  Model key        : {self.model_key}")
        print(f"  Model name       : {self.model_name}")

    def run(self, image_path: str, prompt: str, max_retries: int = 3) -> str:
        """
        Send image + prompt to model via OpenRouter and return raw response text.

        Args:
            image_path  : Full path to the image file
            prompt      : The query prompt to send
            max_retries : Number of retries on API failure

        Returns:
            Raw response string, or "ERROR" after all retries fail
        """
        base64_image = encode_image(image_path)

        params = {
            "model"      : self.model_name,
            "messages"   : [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
            "max_tokens" : 4000,
            "temperature": 0.7,   
        }

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**params)

                content = response.choices[0].message.content
                if content is None or content.strip() == "":
                    return "EMPTY"

                return content.strip()

            except Exception as e:
                print(f"    [Attempt {attempt + 1}/{max_retries}] Error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)   
                else:
                    print(f"    All {max_retries} attempts failed. Skipping.")
                    return "ERROR"
