from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import requests
import getpass

# ---------------- API KEYS ----------------

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

load_dotenv()
api_key = os.getenv("WEATHERSTACK_API_KEY")

if not api_key:
    raise ValueError("WEATHERSTACK_API_KEY not found in .env file")

# ---------------- TOOLS ----------------

search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
    """Fetch current weather data for a given city."""
    url = "http://api.weatherstack.com/current"
    params = {"access_key": api_key, "query": city}

    response = requests.get(url, params=params)
    data = response.json()

    if "error" in data:
        return f"Weather API Error: {data['error']['info']}"

    location = data["location"]["name"]
    country = data["location"]["country"]
    temp = data["current"]["temperature"]
    description = data["current"]["weather_descriptions"][0]

    return (
        f"Weather in {location}, {country}:\n"
        f"Temperature: {temp}°C\n"
        f"Condition: {description}"
    )

tools = [search_tool, get_weather_data]

# ---------------- MODEL ----------------

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=1.0,
)

# ---------------- AGENT ----------------

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt="You are a helpful weather assistant."
)

# ---------------- RUN ----------------

response = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "Tell me the current weather conditions of Karachi as a weather forecast"}
        ]
    }
)

print(response["messages"][-1].content)