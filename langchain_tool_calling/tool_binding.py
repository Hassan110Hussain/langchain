from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# tool create


@tool
def multiply(a: int, b: int) -> int:
    """Given 2 numbers a and b this tool returns their product"""
    return a * b


# print(multiply.invoke({"a": 4, "b": 4}))

# print(multiply.name)
# print(multiply.description)
# print(multiply.args)

# tool binding

llm = ChatOpenAI()

llm_mul = llm.bind_tools([multiply])

result = llm_mul.invoke("How r u?")

# print(result)

query = HumanMessage("can u multiply 32 with 113")

messages = [query]

result1 = llm_mul.invoke(messages)

messages.append(result1)
# print(messages)

# print(result1.tool_calls[0]['args'])
# print(result1.tool_calls[0])

result2 = multiply.invoke(result1.tool_calls[0]["args"])

# print(result2)

result3 = multiply.invoke(
    {
        "name": "multiply",
        "args": {"a": 3, "b": 10},
        "id": "call_zUvy3bMBky7s2BlVxdHWrxTJ",
        "type": "tool_call",
    }
)

# print(result3)

tool_result = multiply.invoke(result1.tool_calls[0])

result4 = messages.append(tool_result)

# print(result4)

result5 = llm_mul.invoke(messages).content

print(result5)
