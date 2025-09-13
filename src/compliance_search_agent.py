import os
from langchain import hub
from langchain_mistralai import ChatMistralAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from dotenv import load_dotenv

# 1. Load environment variables
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# 2. Configure the LLM
# This connects to the remote Mistral API using your API key.
llm = ChatMistralAI(model="mistral-medium", api_key=MISTRAL_API_KEY)

# 3. Define a simple tool
def get_current_date(query: str) -> str:
    """Returns the current date. The query is ignored."""
    today = date.today()
    return today.strftime("%B %d, %Y")

tools = [
    Tool(
        name="Get_Current_Date",
        func=get_current_date,
        description="Useful for when you need to know the current date."
    )
]

# 4. Pull the agent prompt from the LangChain Hub
prompt = hub.pull("hwchase17/react")

# 5. Create the Agent
agent = create_react_agent(llm, tools, prompt)

# 6. Create the Agent Executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 7. Run the Agent
if __name__ == "__main__":
    response = agent_executor.invoke(
        {"input": "What is the current date?"}
    )
    print("Response:", response["output"])