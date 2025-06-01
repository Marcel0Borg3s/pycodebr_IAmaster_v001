import os
from dotenv import load_dotenv
from openai import OpenAI

from mod2_aula004 import client

# Load environment variables from .env file with dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Create an OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

response = client.images.generate(
    model="dall-e-3",
    prompt="A cartoon picture of a children playing with a pet cat",
    size="1024x1024",
    quality="standard"
    n=1 #parametro p gerar somente 1 imagme ou mais
)
image_url = response.data[0].url
print(image_url)


