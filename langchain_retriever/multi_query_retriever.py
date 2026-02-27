from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

docs = [
    Document(
        page_content="Daily stretching improves flexibility and reduces muscle stiffness.",
        metadata={"topic": "exercise", "type": "tip", "year": 2023},
    ),
    Document(
        page_content="Consuming leafy greens provides essential vitamins like A, C, and K.",
        metadata={"topic": "nutrition", "type": "fact", "year": 2022},
    ),
    Document(
        page_content="Practicing deep breathing exercises can help manage anxiety and stress.",
        metadata={"topic": "mental_health", "type": "practice", "year": 2021},
    ),
    Document(
        page_content="Adequate sunlight exposure boosts Vitamin D levels, supporting bone health.",
        metadata={"topic": "wellness", "type": "advice", "year": 2020},
    ),
    Document(
        page_content="Drinking herbal teas like chamomile and peppermint aids digestion and relaxation.",
        metadata={"topic": "diet", "type": "recommendation", "year": 2019},
    ),
    Document(
        page_content="The Eiffel Tower was originally built as a temporary structure for the 1889 World Fair.",
        metadata={"topic": "history", "type": "fact", "year": 1889},
    ),
    Document(
        page_content="Mars is home to Olympus Mons, the tallest volcano in the solar system.",
        metadata={"topic": "astronomy", "type": "discovery", "year": 1971},
    ),
    Document(
        page_content="Machine learning algorithms improve by analyzing large datasets to find hidden patterns.",
        metadata={"topic": "technology", "type": "concept", "year": 2023},
    ),
    Document(
        page_content="Shakespeare introduced over 1,700 new words into the English language.",
        metadata={"topic": "literature", "type": "fact", "year": 1600},
    ),
    Document(
        page_content="The Amazon rainforest produces about 20% of the world’s oxygen supply.",
        metadata={"topic": "environment", "type": "research", "year": 2018},
    ),
]

embeddings = OpenAIEmbeddings()

vector_store = FAISS.from_documents(documents=docs, embedding=embeddings)

similarity_retriever = vector_store.as_retriever(
    search_type="similarity", search_kwargs={"k": 4}
)

multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    ),
    llm=ChatOpenAI(model="gpt-5-nano"),
)

query = "What is the capital of Germany?"

similarity_results = similarity_retriever.invoke(query)
multi_query_results = multiquery_retriever.invoke(query)

for i, doc in enumerate(similarity_results):
    print(f"\n--- Result{i+1} ---")
    print(doc.page_content)

print("+" * 150)

for i, doc in enumerate(multi_query_results):
    print(f"\n--- Result{i+1} ---")
    print(doc.page_content)
