import os
import time
from typing import Any

from openai import OpenAI


class Transcriber:
    def __init__(self):
        open_ai_api_key = os.getenv("OPEN_AI_API_KEY")

        if not open_ai_api_key:
            raise ValueError("Transcriber failed to initialize, api key is not set in environment.")

        self.client = OpenAI(api_key=open_ai_api_key)

    def get_audio_transcript(self, path_to_audio: str, prompt: str = None) -> dict[str, Any]:
        """Gets the transcript for an audio file"""

        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                with open(path_to_audio, "rb") as f:
                    resp = self.client.audio.transcriptions.create(
                        model="whisper-1",
                        file=f,
                        response_format="verbose_json",
                        timestamp_granularities=["segment"],
                        language="zh",
                        prompt=prompt
                    )
                return resp.model_dump()
            except Exception as e:
                print(f"OpenAI Whisper attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise e


