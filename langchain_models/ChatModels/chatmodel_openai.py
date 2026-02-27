from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-5-nano", temperature=0.5, max_tokens=100)

result = model.invoke("What is the capital of Germany?")

print(result.content)