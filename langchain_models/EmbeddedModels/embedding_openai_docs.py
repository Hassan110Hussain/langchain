from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=32)

documents = ["What is the capital of Pakistan?", "What is the capital of India?"]

result = embeddings.embed_documents(documents)

print(result)
