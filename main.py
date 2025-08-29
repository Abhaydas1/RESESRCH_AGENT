from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import os
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

load_dotenv()

# openai_api_key = os.getenv("OPENAI_API_KEY")
# anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# llm=ChatOpenAI(model="openai/gpt-4-turbo", api_key=openai_api_key)
# llm2=ChatAnthropic(model="anthropic/claude-sonnet-4", api_key=anthropic_api_key)

class ReasearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

llm = ChatOpenAI(
    model= "meta-llama/llama-4-maverick:free",  # Model name from OpenRouter
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
    max_tokens=10000
)
# response = llm.invoke("what is the meaning of life?")     #for checking the llm is responding or not?
# print(response)

parser=PydanticOutputParser(pydantic_object=ReasearchResponse)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool, save_tool]
agent= create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)


agent_executer=AgentExecutor(agent=agent, tools=tools, verbose=True)
query=input("What can i help you in research? ")
raw_response=agent_executer.invoke({"query": query})


# try:
#     output_str = raw_response.get("output", "{}")   # get raw string
#     structured_response = parser.parse(output_str)  # parse using your Pydantic parser
#     print("Structured Response:", structured_response)
# except Exception as e:
#     print("Error parsing response:", e)
#     print("Raw Response:", raw_response)

try:
    structured_response = parser.parse(raw_response.get("output", "{}"))#'[0]["text"]')
    print(structured_response)
except Exception as e:
    print("Error parsing response", e, "Raw Response - ", raw_response)



# from dotenv import load_dotenv
# from pydantic import BaseModel
# from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
# import os
# import json
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import PydanticOutputParser
# from langchain.agents import create_tool_calling_agent, AgentExecutor
# from tools import search_tool, wiki_tool, save_tool

# # Load environment variables
# load_dotenv()

# # Define Pydantic schema
# class ResearchResponse(BaseModel):
#     topic: str
#     summary: str
#     sources: list[str]
#     tools_used: list[str]

# # Load API key
# openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

# # LLM setup (OpenRouter model)
# llm = ChatOpenAI(
#     model="meta-llama/llama-4-maverick:free",  # OpenRouter model
#     api_key=openrouter_api_key,
#     base_url="https://openrouter.ai/api/v1",
#     max_tokens=100000  # safer than 100000
# )

# # Parser for structured output
# parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# # Prompt template
# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             """
#             You are a research assistant that helps generate a research paper.
#             Answer the user query and use necessary tools. 
#             Wrap the output in this format and provide no other text:
#             {format_instructions}
#             """,
#         ),
#         ("placeholder", "{chat_history}"),
#         ("human", "{query}"),
#         ("placeholder", "{agent_scratchpad}"),
#     ]
# ).partial(format_instructions=parser.get_format_instructions())

# # Tools
# tools = [search_tool, wiki_tool, save_tool]

# # Create tool-using agent
# agent = create_tool_calling_agent(
#     llm=llm,
#     prompt=prompt,
#     tools=tools
# )

# # Agent executor with verbose logging (shows intermediate tool calls)
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# # === MAIN LOOP ===
# query = input("What can I help you research? ")

# raw_response = agent_executor.invoke({"query": query})

# # === Final Parsing into Pydantic ===
# try:
#     output_data = raw_response.get("output", "{}")

#     # Ensure it's a string before parsing
#     if isinstance(output_data, dict):
#         output_data = json.dumps(output_data)

#     structured_response = parser.parse(output_data)

#     print("\n✅ Final Structured Response:\n")
#     print(structured_response)

# except Exception as e:
#     print("❌ Error parsing response:", e)
#     print("Raw Response - ", raw_response)
