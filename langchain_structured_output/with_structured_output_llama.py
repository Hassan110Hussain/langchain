from dotenv import load_dotenv
from typing import Optional, Literal
from pydantic import BaseModel, Field
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    # huggingfacehub_api_token=hf_token,
)

model = ChatHuggingFace(llm=llm)


class Review(BaseModel):
    key_themes: list[str] = Field(
        description="Write down all the key themes discused in the review in a list"
    )
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["pos", "neg"] = Field(
        description="return sentiment either positive, negative or neutral"
    )
    pros: Optional[list[str]] = Field(description="wrtie down all the pros")
    cons: Optional[list[str]] = Field(
        description="wrtie down all the cons"
    )  # define in prompt explicitly that dont give cons if not given in the para.
    name: Optional[str] = Field(description="write the name of the reviewer")


structured_model = model.with_structured_output(Review)

result = structured_model.invoke(
    """Modern software ecosystems increasingly depend on finely optimized hardware architectures to achieve exceptional computational performance and responsiveness. Cutting-edge hardware enhancements frequently unlock previously unattainable possibilities for sophisticated software applications, enabling innovations that push the boundaries of efficiency and scalability. In this symbiotic relationship, software functions as the cognitive engine—an intelligent orchestrator of logic and decision-making—while hardware operates as the underlying framework, executing instructions with precision and speed. Achieving seamless compatibility between software and hardware is essential to ensure fluid user experiences, prevent bottlenecks, and maintain system stability under demanding workloads. As emerging technologies continue to evolve, the distinction between software intelligence and hardware capability becomes progressively intertwined, creating a landscape where efficiency, adaptability, and innovation are mutually reinforcing. Pros of Optimized Software-Hardware Integration

High Performance: Optimized hardware enables software to run faster and more efficiently, improving system responsiveness.

Enhanced User Experience: Seamless compatibility ensures smooth operation, fewer crashes, and a more reliable interface."""
)

print(result)
# print(result.summary)
# print(result.sentiment)
