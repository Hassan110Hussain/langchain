from langchain_core.runnables import Runnable, RunnableParallel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence

load_dotenv()

prompt1 = PromptTemplate(
    template="Generate a Twitter tweet about {topic}", input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate a Linkedin post about {topic}", input_variables=["topic"]
)

model = ChatOpenAI(model="gpt-5-nano")

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        "tweet": RunnableSequence(prompt1 | model | parser),
        "post": RunnableSequence(prompt2 | model | parser),
    }
)

print(parallel_chain.invoke({"topic": "Chris Hemsworth"}))
