from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=32)

documents = ["Islamabad is the capital of Pakistan", "Tehran is the capital of Iran", "Virat Kohli the King Kohli"]

query = "tell me about Pakistan"

doc_embed = embeddings.embed_documents(documents)
query_embed = embeddings.embed_query(query)

similarity = cosine_similarity([query_embed], doc_embed)[0]

index, score = sorted(list(enumerate(similarity)),key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print('score is:', score)