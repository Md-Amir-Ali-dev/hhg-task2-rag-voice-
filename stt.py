import os
import asyncio
import time
import requests


class SarvamSTTClient:
    """
    Speech-to-Text client using Sarvam AI's saaras:v3 model.
    Docs: https://docs.sarvam.ai/api-reference-docs/speech-to-text
    Supports: WebM, WAV, MP3, OGG, AAC, MP4, etc.
    """

    def __init__(self, api_key=None, use_mock=False):
        from dotenv import load_dotenv
        load_dotenv(override=True)
        # Check SARVAM_API_KEY first, fallback to ELEVENLABS_API_KEY in case user pasted it there
        self.api_key = api_key or os.getenv("SARVAM_API_KEY") or os.getenv("ELEVENLABS_API_KEY")
        if self.api_key and "your_" in self.api_key:
            self.api_key = None
        self.use_mock = use_mock
        self.api_url = "https://api.sarvam.ai/speech-to-text"

    async def transcribe_stream(self, audio_generator, filename="recording.webm", content_type="audio/webm"):
        """
        Takes an async generator yielding audio bytes.
        Collects all bytes and sends directly to Sarvam STT REST API.
        """
        if self.use_mock or not self.api_key:
            return await self._mock_transcribe(audio_generator)

        try:
            audio_buffer = b""
            async for chunk in audio_generator:
                audio_buffer += chunk

            if not audio_buffer:
                return ""

            loop = asyncio.get_event_loop()
            transcript = await loop.run_in_executor(
                None, self._call_api, audio_buffer, filename, content_type
            )
            return transcript
        except Exception as e:
            print(f"Sarvam STT Error: {e}")
            return f"STT error: {e}"

    def _call_api(self, audio_bytes: bytes, filename: str, content_type: str) -> str:
        """Blocking call to Sarvam AI STT REST API."""
        headers = {
            "api-subscription-key": self.api_key.strip(),
        }
        files = {
            "file": (filename, audio_bytes, content_type),
        }
        data = {
            "model": "saaras:v3",
            "language_code": "unknown",   # Sarvam auto-detects language (en-IN, hi-IN, etc.)
        }
        response = requests.post(self.api_url, headers=headers, files=files, data=data, timeout=30)
        
        if not response.ok:
            print(f"Sarvam API response [{response.status_code}]: {response.text}")
            response.raise_for_status()

        result = response.json()
        return result.get("transcript", "").strip()

    async def _mock_transcribe(self, audio_generator):
        """Fallback mock transcription if no API key is provided."""
        print("Using MOCK STT (No valid SARVAM_API_KEY found in .env)...")
        async for _ in audio_generator:
            await asyncio.sleep(0.01)
        # Randomize mock queries to prevent always returning the same question
        samples = [
            "What is the capital of India?",
            "Where is the Taj Mahal located?",
            "What is machine learning?",
            "Tell me about Indian history and culture"
        ]
        import random
        return random.choice(samples)
