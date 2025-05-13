# File: app.py
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from tools import save_tool, search_tool, wiki_tool
import os
import json
import traceback # Import traceback for detailed error info


load_dotenv()
st.title("AI Research Assistant")

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources:list[str]
    tools_used: list[str]

llm = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0
)
parser = PydanticOutputParser(pydantic_object=ResearchResponse)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "\"\"\"\n"
            "You are an AI-powered research assistant. Your SOLE purpose is to conduct research using available tools and provide the final results in a STRICT JSON format.\n"
            "Use available tools ONLY when necessary to gather information for the research.\n"
            "When you have completed the research and are ready to provide the final answer, you MUST present your output in the following JSON format, and NOTHING else:\n"
            "\\n{format_instructions}\n"
            "Provide ONLY the JSON object. Do NOT include any introductory text, explanations, or markdown code block delimiters (like ```json```) before or after the JSON.\n\n"
            "If you use the `save_text_to_file` tool, the `data` argument for this tool MUST be the complete JSON string containing the research results (topic, summary, sources, tools_used) that you generate as your final answer.\n"
            "\"\"\"",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# Asegúrate de incluir search_tool y wiki_tool si son necesarios para el agente
# tools = [save_tool, search_tool, wiki_tool]
tools = [save_tool, wiki_tool] # Usando solo save_tool como en tu código actual
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

# Initialize memory in session state for Streamlit
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create agent executor using session state memory
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=st.session_state.memory, verbose=True)

st.title("AI Research Assistant")
## add mistral API key input
mistral_api_key = st.text_input("Enter your Mistral API key:", type="password")
os.environ["MISTRAL_API_KEY"] = mistral_api_key
query = st.text_input("Enter your research query:")

if st.button("Research"):
    if query:
        with st.spinner("Thinking..."):
            try:
                raw_response = agent_executor.invoke({"query": query})

                # Check if 'output' key exists and is a non-empty string
                if "output" in raw_response and isinstance(raw_response["output"], str) and raw_response["output"].strip():
                    output_string = raw_response["output"].strip()

                    # Clean the string: remove markdown code block delimiters if present
                    if output_string.startswith("```json"):
                        output_string = output_string[len("```json"):].lstrip() # Remove ```json and leading whitespace
                    if output_string.endswith("```"):
                        output_string = output_string[:-len("```")].rstrip() # Remove ``` and trailing whitespace

                    try:
                        # Attempt to parse the cleaned string as JSON
                        parsed = json.loads(output_string)

                        st.subheader("Research Results:")
                        st.write(f"**Topic:** {parsed['topic']}")
                        st.write(f"**Summary:** {parsed['summary']}")
                        st.write(f"**Sources:** {', '.join(parsed['sources'])}")
                        st.write(f"**Tools Used:** {', '.join(parsed['tools_used'])}")
                    except json.JSONDecodeError as e:
                        # If JSON parsing fails, display the raw output as text
                        st.subheader("Agent Output (Not JSON):")
                        st.write("The agent did not return output in the expected JSON format.")
                        st.code(raw_response["output"]) # Display the original raw string output
                        st.error(f"Error decoding JSON: {e}") # Still show the error for debugging

                elif "output" in raw_response and isinstance(raw_response["output"], str) and not raw_response["output"].strip():
                     st.error("Agent produced an empty output string.")
                     st.write("Raw response from agent:")
                     st.json(raw_response) # Display the full raw response
                else:
                    st.error("Agent did not produce a valid output in the expected format.")
                    st.write("Raw response from agent:")
                    st.json(raw_response) # Display the full raw response

            except Exception as e:
                st.error(f"An unexpected error occurred during agent execution: {e}")
                st.write("Traceback:")
                st.code(traceback.format_exc())


    else:
        st.warning("Please enter a query.")

#add a section here to display the chat history from st.session_state.memory
st.subheader("Chat History")
for message in st.session_state.memory.chat_memory.messages:
    st.write(f"**{message.type}:** {message.content}")
    st.write("---")
