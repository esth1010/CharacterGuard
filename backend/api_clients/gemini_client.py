"""
Thin wrapper around the Google Gemini API.

Drop-in replacement for AiClient, exposes the same call() interface
so llm_judge.py and question_generator.py don't need to change.

Requires:
    GEMINI_API_KEY in .env file
"""

import json
import os
import time
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()


GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_MODEL    = "gemini-3.1-flash-lite-preview"

DEFAULT_MAX_TOKENS  = 10000
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TIMEOUT     = 30
MAX_RETRIES         = 3
RETRY_DELAY         = 5 # in seconds


# GeminiClient

class GeminiClient:
    """
    Sends a system prompt + user message to Gemini and returns the raw text.

    Usage:
        client = GeminiClient()
        response = client.call(
            system_prompt="You are a strict safety evaluator...",
            user_message="CHARACTER DESCRIPTION: ..."
        )
        print(response)  # raw string (expected to be JSON for llm_judge)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = GEMINI_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.api_key    = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. "
                "Set GEMINI_API_KEY in your .env file or pass it directly."
            )

        self.model       = model
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.timeout     = timeout

        self._url = (
            f"{GEMINI_BASE_URL}/{self.model}:generateContent"
            f"?key={self.api_key}"
        )

    def call(self, system_prompt: str, user_message: str) -> str:
        """
        Send a system prompt + user message to Gemini.

        Args:
            system_prompt : Instructions that control the model's behaviour
            user_message  : The actual content to evaluate

        Returns:
            Raw text string from the model.

        Raises:
            RuntimeError if all retries are exhausted.
        """
        payload = self._build_payload(system_prompt, user_message)
        headers = {"Content-Type": "application/json"}

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = requests.post(
                    url=self._url,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return self._extract_text(response.json())

            except requests.exceptions.Timeout:
                print(f"[Gemini] Timed out (attempt {attempt})")

            except requests.exceptions.HTTPError as e:
                status = e.response.status_code
                if status == 429:
                    if attempt == MAX_RETRIES:
                        raise RuntimeError("Gemini rate limit hit on all retries.")
                    print(f"[Gemini] Rate limited (attempt {attempt}), waiting...")
                    time.sleep(RETRY_DELAY * attempt)
                    continue
                else:
                    print(f"[Gemini] HTTP error {status}: {e.response.text[:200]}")

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

        raise RuntimeError(
            f"Gemini request failed after {MAX_RETRIES} retries. "
            "Check your API key and network connection."
        )

    # Private helpers

    def _build_payload(self, system_prompt: str, user_message: str) -> dict:
        """
        Build the Gemini generateContent request body.

        Gemini separates system instructions from user turns — this maps
        cleanly onto the same system_prompt / user_message pattern used
        by the OpenRouter client.

        safetySettings: all four harm categories are set to BLOCK_NONE so
        Gemini does not refuse to process explicit content that is being
        *evaluated* (not generated). This is the standard approach for
        content-moderation pipelines — we are classifying harmful content,
        not producing it. Without this, Gemini blocks the request whenever
        the character response being judged contains adult/violent content.
        """
        return {
            "system_instruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_message}]
                }
            ],
            "generationConfig": {
                "temperature":     self.temperature,
                "maxOutputTokens": self.max_tokens,
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ],
        }

    def _extract_text(self, response_json: dict) -> str:
        try:
            candidate = response_json["candidates"][0]

            finish_reason = candidate.get("finishReason", "")
            if finish_reason == "PROHIBITED_CONTENT" or "content" not in candidate:
                raise RuntimeError(
                    f"Gemini blocked the response (finishReason: {finish_reason}). "
                    "The prompt may be triggering Gemini's safety filters — "
                    "try reframing the system prompt to sound more like a safety researcher."
                )

            return candidate["content"]["parts"][0]["text"]

        except RuntimeError:
            raise
        except (KeyError, IndexError) as e:
            raise RuntimeError(
                f"Could not extract text from Gemini response: {e}\n"
                f"Response: {json.dumps(response_json)[:300]}"
            )
