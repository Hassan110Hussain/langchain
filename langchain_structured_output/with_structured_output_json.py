from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

model = ChatOpenAI(model="gpt-5-nano")


json_schema = {
    "title": "student",
    "type": "object",
    "properties": {
        "key_themes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Write down all the key themes discussed in the review in a list",
        },
        "summary": {"type": "string", "description": "A brief summary of the review"},
        "sentiment": {
            "type": "string",
            "enum": ["pos", "neg", "neutral"],
            "description": "Return sentiment as positive, negative, or neutral",
        },
    },
    "required": ["key_themes", "summary"],
}


structured_model = model.with_structured_output(json_schema)

result = structured_model.invoke(
    """Modern software ecosystems increasingly depend on finely optimized hardware architectures to achieve exceptional computational performance and responsiveness. Cutting-edge hardware enhancements frequently unlock previously unattainable possibilities for sophisticated software applications, enabling innovations that push the boundaries of efficiency and scalability. In this symbiotic relationship, software functions as the cognitive engine—an intelligent orchestrator of logic and decision-making—while hardware operates as the underlying framework, executing instructions with precision and speed. Achieving seamless compatibility between software and hardware is essential to ensure fluid user experiences, prevent bottlenecks, and maintain system stability under demanding workloads. As emerging technologies continue to evolve, the distinction between software intelligence and hardware capability becomes progressively intertwined, creating a landscape where efficiency, adaptability, and innovation are mutually reinforcing. Pros of Optimized Software-Hardware Integration

High Performance: Optimized hardware enables software to run faster and more efficiently, improving system responsiveness.

Enhanced User Experience: Seamless compatibility ensures smooth operation, fewer crashes, and a more reliable interface."""
)

print(result)
# print(result.summary)
# print(result.sentiment)
