from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

load_dotenv()

search_tool = DuckDuckGoSearchRun()

a = search_tool.invoke("top news for Pakistan")


@tool
def get_weather_data(city: str) -> str:
    """This function fetches the current weather data for a given city"""

    url = f"http://api.weatherstack.com/current?access_key=''&query={city}"

    response = requests.get(url)

    return response.json()


llm = ChatOpenAI(model="gpt-4o-mini", stop=None)

b = llm.invoke("r u fine?")

prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm=llm, tools=[search_tool, get_weather_data], prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent, tools=[search_tool, get_weather_data], verbose=True
)

response = agent_executor.invoke(
    {"input": "Tell me the current weather conditions of Karachi as a weather forecast"}
)

print(response["output"])
