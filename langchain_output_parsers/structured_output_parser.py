import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate, prompt
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    huggingfacehub_api_token=hf_token,
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name="fact1", description="Fact 1 about the topic"),
    ResponseSchema(name="fact2", description="Fact 2 about the topic"),
    ResponseSchema(name="fact3", description="Fact 3 about the topic"),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="give 3 facts about {topic} \n {format_instruction}", # doesnt enforce validation
    input_variables=["topic"],
    partial_variables={"format_instruction": parser.get_format_instructions()},
)

# prompt = template.invoke({"topic": "black hole"})

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)

# print(final_result)

chain = template | model | parser

result = chain.invoke({"topic": "sun n moon"})

print(result)
