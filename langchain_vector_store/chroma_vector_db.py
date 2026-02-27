from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

docs = [
    Document(
        page_content="Virat Kohli scored a match-winning 122* against Pakistan in the Asia Cup.",
        metadata={"source": "ESPNcricinfo", "year": 2023, "type": "news"},
    ),
    Document(
        page_content="Sachin Tendulkar holds the record for 100 international centuries in cricket.",
        metadata={"source": "Wikipedia", "year": 2011, "type": "record"},
    ),
    Document(
        page_content="The ICC Cricket World Cup 2019 final between England and New Zealand ended in a Super Over.",
        metadata={"source": "ICC", "year": 2019, "type": "tournament"},
    ),
    Document(
        page_content="Shane Warne took 708 wickets in Test cricket, making him one of the greatest leg-spinners.",
        metadata={"source": "Cricbuzz", "year": 2007, "type": "career_stats"},
    ),
    Document(
        page_content="In IPL 2020, Mumbai Indians defeated Delhi Capitals to win their fifth title.",
        metadata={"source": "IPL Official", "year": 2020, "type": "tournament_result"},
    ),
]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)

vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="chroma_db",
    collection_name="sample",
)

vector_store.add_documents(docs)
collection = vector_store._collection  
# data = vector_store.get(include=["embeddings", "documents", "metadatas"])
data = collection.get(include=["documents", "metadatas", "embeddings"])
print("IDs:", data["ids"])
print("Documents:", data["documents"])
print("Metadatas:", data["metadatas"])
print("Number of embeddings:", len(data["embeddings"]))

# results = vector_store.similarity_search(query="who among them has taken wickets", k=1)
# print(results)

# results2 = vector_store.similarity_search_with_score(query="who has the 5th title", k=2)
# print(results)
# vector_store.similarity_search_with_score(query="", filter={"team": "RCB player"})

# updated_doc1 = Document(
#     page_content="Virat Kohli scored a match-winning 122* against Pakistan in the Asia Cup.",
#     metadata={"source": "ESPNcricinfo", "year": 2023, "type": "news"},
# )

# vector_store.update_document(document_id="12345", document=updated_doc1)

# vector_store.delete(ids=["12345"])
# vector_store.get(include=["embeddings", "documents", "metadatas"])
