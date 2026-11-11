from dotenv import load_dotenv

load_dotenv()

from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langchain.messages import HumanMessage


tavily_search = TavilySearch()

@tool
def search(query: str) -> str:
    """
    Tool that searches over Internet.
    Args:
        query: The query to search for
    Returns:
        The search result
    """
    print(f"Searching for {query}")
    return tavily_search.invoke(query)


llm = ChatOpenAI(model="gpt-5")
tools = [search]
agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages":HumanMessage(content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details.")})
    print(result)

if __name__ == "__main__":
    main()

