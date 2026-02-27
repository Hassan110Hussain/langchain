from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model="gpt-5-nano")

result = llm.invoke("What is the capital of France?")

print(result)