import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate, prompt
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    huggingfacehub_api_token=hf_token,
)

model = ChatHuggingFace(llm=llm)


class Person(BaseModel):
    name: str = (Field(description="name of a person"),)
    age: int = Field(gt=18, description="age of the person")
    city: str = Field(description="name of the city the person lives")


parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate the name, age n city of a fictional {place} person \n {format_instruction}",
    input_variables=["place"],
    partial_variables={"format_instruction": parser.get_format_instructions()},
)

# prompt = template.invoke({"place": "karachi"})

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)

# print(final_result)

chain = template | model | parser

result = chain.invoke({"place": "karachi"})

print(result)
