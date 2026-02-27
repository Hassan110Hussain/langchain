from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# print(dir(YouTubeTranscriptApi))

video_id = "FvZSiDNP6tk"
try:
    yt = YouTubeTranscriptApi()
    transcript_list = yt.fetch(video_id, languages=["en"])

    transcript_text = " ".join([chunk.text for chunk in transcript_list])
    # print(transcript_text)

except TranscriptsDisabled:
    print("Transcripts are disabled for this video")

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = splitter.create_documents([transcript_text])
print(len(chunks))
# print(chunks)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(chunks, embeddings)
# print(vector_store.index_to_docstore_id)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
results = retriever.invoke("What is the main topic of the video?")
# print(results)

llm = ChatOpenAI(model="gpt-5-nano", temperature=0.5)

prompt = PromptTemplate(
    template="""You are a helpful assistant. Answer only from the provided transcript context. If the context is insufficient, say I don't know. 

    {context} 
     Question: {question}
     """,
    input_variables=["question", "context"],
)

question = "who the hell is VIRAT KOHLI?"
retrieved_docs = retriever.invoke(question)
# print(retrieved_docs)

context = "\n\n".join([doc.page_content for doc in retrieved_docs])
final_prompt = prompt.invoke({"question": question, "context": context})
# print(final_prompt)

answer = llm.invoke(final_prompt)
# print(answer.content)


# Much cleaner code starts from here!!
def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text


parallel_chain = RunnableParallel(
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
)

result = parallel_chain.invoke("Who is Virat")
# print(result)

parser = StrOutputParser()
main_chain = parallel_chain | prompt | llm | parser

result = main_chain.invoke("Can u summarize the video")
print(result)
