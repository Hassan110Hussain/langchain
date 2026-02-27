from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from dotenv import load_dotenv

load_dotenv()

docs = [
    Document(
        page_content=(
            "Regular exercise strengthens the heart and improves blood circulation.\n"
            "Meditation helps calm the mind and reduce stress levels.\n"
            "A balanced diet fuels the body with essential nutrients."
        ),
        metadata={"topic": "health", "year": 2023},
    ),
    Document(
        page_content=(
            "The Great Wall of China was built to protect against invasions.\n"
            "Solar energy is becoming one of the fastest-growing renewable resources.\n"
            "Classical music can enhance concentration and memory retention."
        ),
        metadata={"topic": "mixed_knowledge", "year": 2022},
    ),
    Document(
        page_content=(
            "Drinking water first thing in the morning helps kickstart metabolism.\n"
            "Reading books expands vocabulary and strengthens critical thinking.\n"
            "Traveling exposes you to diverse cultures and new perspectives."
        ),
        metadata={"topic": "lifestyle", "year": 2021},
    ),
    Document(
        page_content=(
            "The moon influences ocean tides through gravitational pull.\n"
            "Regular sleep of 7-8 hours boosts immune system function.\n"
            "Learning a new language improves brain plasticity and memory."
        ),
        metadata={"topic": "science_and_health", "year": 2020},
    ),
    Document(
        page_content=(
            "Volunteering provides a sense of purpose and community connection.\n"
            "Coffee contains antioxidants that may reduce the risk of disease.\n"
            "Mount Everest is the tallest mountain above sea level in the world."
        ),
        metadata={"topic": "general_knowledge", "year": 2019},
    ),
]

embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(docs, embeddings)

base_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(model="gpt-5-nano")
compressor = LLMChainExtractor.from_llm(llm)

compressor_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever, base_compressor=compressor
)

query = "What is the photgraphy?"
results = compressor_retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n--- Result{i+1} ---")
    print(doc.page_content) 
