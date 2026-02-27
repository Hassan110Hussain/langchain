from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type


class MultiplyInput(BaseModel):
    a: int = Field(required=True, description="The first number to add")
    b: int = Field(required=True, description="The first number to add")


class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "multiply two numbers"

    args_schema: Type[BaseModel] = MultiplyInput

    def _run(self, a: int, b: int) -> int:
        return a * b


multiply = MultiplyTool()

result = multiply.invoke({"a": 4, "b": 4})
print(result)

print(multiply.name)
print(multiply.description)
print(multiply.args)
