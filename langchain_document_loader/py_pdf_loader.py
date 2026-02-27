from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-5-nano")

parser = StrOutputParser()

loader = PyPDFLoader("cricket.pdf")

docs = loader.load()

print(docs[0].metadata)
print(docs[0].page_content)
