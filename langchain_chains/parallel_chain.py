import os
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    huggingfacehub_api_token=hf_token,
)

model1 = ChatHuggingFace(llm=llm)

model2 = ChatOpenAI(model="gpt-5-nano")

prompt1 = PromptTemplate(
    template="Generate short n simple notes from the following text \n {text}",
    input_variables=["text"],
)

prompt2 = PromptTemplate(
    template="Generate 5 short questions from the following text \n {text}",
    input_variables=["text"],
)

prompt3 = PromptTemplate(
    template="merge the provided notes n quiz into the single document \n {notes} n {quiz}",
    input_variables=["notes", "quiz"],
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {"notes": prompt1 | model1 | parser, "quiz": prompt2 | model2 | parser}
)

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """The sun dipped below the horizon, painting the sky in shades of orange and pink.  
A gentle breeze carried the scent of blooming jasmine through the air.  
Children laughed and chased each other in the park, their joy infectious.  
A lone bird sang a melody that echoed across the quiet streets.  
As night fell, the first stars began to twinkle, promising a peaceful evening."""

result = chain.invoke({"text": text})

print(result)
