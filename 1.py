from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()  # Load variables from .env

api_key = os.getenv("GROQ_API_KEY")

# Configure the Groq client
client = Groq(api_key=api_key)

# Generate a response using Groq
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",  # You can also use "llama-3.1-8b-instant" or "gemma2-9b-it"
    messages=[
        {"role": "user", "content": "hii this is Aji from pwc"}
    ],
    temperature=0.2,
    max_tokens=1000
)

# Print the result
print(response.choices[0].message.content)