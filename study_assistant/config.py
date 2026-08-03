import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")


if LLM_PROVIDER == "groq":
    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )

elif LLM_PROVIDER == "openai":
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )

else:
    raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")