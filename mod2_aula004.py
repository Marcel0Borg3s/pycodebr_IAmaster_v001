import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file with dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Create an OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Call the OpenAI API
response = client.chat_completion.create(
    model='gpt-3.5-turbo',
    messages=[
        {
            'role': 'user',
            'content': 'Ola, me fale sobre o curso de Python e curva de aprendizado',
        },
    ],
)

# Print the response
print(response.choices[0].message.content)
