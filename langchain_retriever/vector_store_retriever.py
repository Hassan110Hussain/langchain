from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()

documents = [
    Document(
        page_content="The old clock tower chimed loudly as the sun dipped below the horizon."
    ),
    Document(
        page_content="A curious cat leapt onto the windowsill to watch the rain fall."
    ),
    Document(
        page_content="Scientists discovered a new species of butterfly deep in the Amazon rainforest."
    ),
    Document(
        page_content="The bakery’s fresh bread aroma spread across the entire street."
    ),
]

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=32)

vectorstore = Chroma.from_documents(
    documents=documents, embedding=embedding_model, collection_name="my_collection"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

query = "what is Clock tower used for?"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n---Result{i+1}---")
    print(f"Context:\n{doc.page_content}...")
