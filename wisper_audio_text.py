import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file with dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Create an OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

audio_file = "audio.mp3"

transcription = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file,
)

print(transcription.text)
