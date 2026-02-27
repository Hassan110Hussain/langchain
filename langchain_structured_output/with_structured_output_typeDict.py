from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal

load_dotenv()

model = ChatOpenAI(model="gpt-5-nano")


class Review(TypedDict):
    key_themes: Annotated[
        list[str], "Write down all the key themes discused in the review in a list"
    ]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[
        Literal["pos", "neg"], "return sentiment either positive, negative or neutral"
    ]
    pros: Annotated[Optional[list[str]], "wrtie down all the pros"]
    cons: Annotated[
        Optional[list[str]], "wrtie down all the cons"
    ]  # define in prompt explicitly that dont give cons if not given in the para.


structured_model = model.with_structured_output(Review)

result = structured_model.invoke(
    """Modern software ecosystems increasingly depend on finely optimized hardware architectures to achieve exceptional computational performance and responsiveness. Cutting-edge hardware enhancements frequently unlock previously unattainable possibilities for sophisticated software applications, enabling innovations that push the boundaries of efficiency and scalability. In this symbiotic relationship, software functions as the cognitive engine—an intelligent orchestrator of logic and decision-making—while hardware operates as the underlying framework, executing instructions with precision and speed. Achieving seamless compatibility between software and hardware is essential to ensure fluid user experiences, prevent bottlenecks, and maintain system stability under demanding workloads. As emerging technologies continue to evolve, the distinction between software intelligence and hardware capability becomes progressively intertwined, creating a landscape where efficiency, adaptability, and innovation are mutually reinforcing. Pros of Optimized Software-Hardware Integration

High Performance: Optimized hardware enables software to run faster and more efficiently, improving system responsiveness.

Enhanced User Experience: Seamless compatibility ensures smooth operation, fewer crashes, and a more reliable interface."""
)

print(result)
# print(result["summary"])
# print(result["sentiment"])
