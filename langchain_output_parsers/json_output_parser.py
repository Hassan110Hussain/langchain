import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate, prompt
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    huggingfacehub_api_token=hf_token,
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    # template="Give me a name, age n city of a fictional person \n {format_instruction}",
    template="Give me 5 facts about {topic} \n {format_instruction}",  # doesnt enforce schema
    input_variables=["topic"],
    partial_variables={"format_instruction": parser.get_format_instructions()},
)

# prompt = template.format()

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)

# print(final_result)
# print(type(final_result))

chain = template | model | parser

result = chain.invoke({"topic": "black hole"})

print(result)
