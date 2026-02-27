from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
import requests
from dotenv import load_dotenv
from langchain_core.tools import InjectedToolArg
from typing import Annotated

load_dotenv()

# tool create


@tool
def get_conversion_tool(base_currency: str, target_currency: str) -> float:
    """This function fetches the currency conversion factor between a given base currency and a target currency"""

    url = f"https://v6.exchangerate-api.com/v6/c734fcf96d9dc4274836f259/pair/{base_currency}/{target_currency}"

    response = requests.get(url)

    return response.json()


@tool
def convert(
    base_currency_value: int, conversion_rate: Annotated[float, InjectedToolArg]
) -> float:
    """given a currency conversion rate this function calculates the target currency value from a given base currency value"""

    return base_currency_value * conversion_rate


result = get_conversion_tool.invoke({"base_currency": "USD", "target_currency": "INR"})

# print(result)

result1 = convert.invoke({"base_currency_value": 10, "conversion_rate": 85.17})

# print(result1)


# tool binding

llm = ChatOpenAI(model="gpt-5-nano")

llm_tools = llm.bind_tools([get_conversion_tool, convert])

messages = [
    HumanMessage(
        "What is the conversion factor between USD and INR, and based on that can u convert 10 usd to inr"
    )
]

# print(messages)

ai_message = llm_tools.invoke(messages)
messages.append(ai_message)

# print(ai_message.tool_calls)

# for tool_call in ai_message.tool_calls:
#     # print(tool_call)
#     # execute the first tool and get the value of conversion rate
#     if tool_call["name"] == "get_conversion_tool":
#         tool_message1 = get_conversion_tool.invoke(tool_call["args"])
#         # print(tool_message1)
#         # fetch this conversion rate
#         conversion_rate = json.loads(tool_message1.content)["conversion_rate"]
#         # append this tool message to message list
#         messages.append(tool_message1)
#     # execute the second tool using the conversion rate from tool 1
#     if tool_call["name"] == "convert":
#         # fetch the current arg
#         tool_call["args"]["conversion_rate"] = conversion_rate
#         tool_message2 = convert.invoke(tool_call)
#         messages.append(tool_message2)

for tool_call in ai_message.tool_calls:
    if tool_call["name"] == "get_conversion_tool":
        tool_message1 = get_conversion_tool.invoke(tool_call["args"])
        # print("Tool1 raw output:", tool_message1)

        # ✅ Extract the float value from API response
        conversion_rate = tool_message1["conversion_rate"]

        messages.append(
            ToolMessage(
                content=str(tool_message1),  # convert dict → string
                tool_call_id=tool_call["id"],  # tie it back to the tool call
            )
        )

    if tool_call["name"] == "convert":
        # ✅ Now pass only the float, not the dict
        tool_call["args"]["conversion_rate"] = conversion_rate

        tool_message2 = convert.invoke(tool_call["args"])

        messages.append(
            ToolMessage(content=str(tool_message2), tool_call_id=tool_call["id"])
        )

# print("Final messages:", messages)

resulty = llm_tools.invoke(messages)

print(resulty.content)
