import pprint
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import asyncio
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOpenAI(model="gpt-4o")

server_params = StdioServerParameters(
    command="python",
    args=["mcp_server.py"],
)


async def run_agent():
    async with stdio_client(server_params) as (read, write):
        # Load the tools from the MCP server
        async with ClientSession(read, write) as session:

            # Initialize the conection
            await session.initialize()

            # Get Tools
            tools = await load_mcp_tools(session)

            # Create and run agent
            agent = create_react_agent(model=model, tools=tools)
            agent_response = await agent.ainvoke({"messages": "what's (3+5) x 12"})
            return agent_response


if __name__ == "__main__":
    response = asyncio.run(run_agent())
    print(response)
    print(response["messages"][4].content)
