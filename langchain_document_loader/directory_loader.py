from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# model = ChatOpenAI(model="gpt-5-nano")

parser = StrOutputParser()

loader = DirectoryLoader(path="pdf_files", glob="*.pdf", loader_cls=PyPDFLoader)

# docs = loader.lazy_load() for so many docs
docs = loader.load()

for document in docs:
    print(document.metadata)
