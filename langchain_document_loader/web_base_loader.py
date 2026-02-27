from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-5-nano')

parser = StrOutputParser()

prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question','text']
)

url = 'https://www.flipkart.com/search?q=apple%20macbook&otracker=search&otracker1=search&marketplace=FLIPKART&as-show=on&as=off'
loader = WebBaseLoader(url)

docs = loader.load()

chain = prompt | model | parser
print(chain.invoke({'question':'How much overloaded the site is?', 'text': docs[0].page_content}))
