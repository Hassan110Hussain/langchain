from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

documents = [
    Document(
        page_content="LangChain makes it easier to connect large language models with external tools and data sources."
    ),
    Document(
        page_content="Developers use LangChain to build chatbots, document assistants, and retrieval-augmented generation systems."
    ),
    Document(
        page_content="Chroma is a popular vector database for storing and searching embeddings efficiently."
    ),
    Document(
        page_content="Embeddings turn text into numerical vectors that capture semantic meaning."
    ),
     Document(
        page_content="Maximum Marginal Relevance (MMR) helps balance relevance and diversity in retrieval results."
    ),
     Document(
        page_content="LangChain also provides retrievers that can fetch context directly from vector stores like Chroma."
    ),
]

embedding_model = OpenAIEmbeddings()

vector_store = FAISS.from_documents(
    documents=documents,
    embedding=embedding_model
)

retriever = vector_store.as_retriever(
    search_type='mmr',
    search_kwargs={'k': 3, 'lambda_mult': 0.1}
)

query = 'What is Langchain?'
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n--- Result{i+1} ---")
    print(doc.page_content) 
