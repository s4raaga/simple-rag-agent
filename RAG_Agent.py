import os
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langgraph.graph.message import add_messages # Reducer function that allows us to append eveything to the State.
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()

# --- LOAD LLM --- #

model = ChatOpenAI(model="gpt-4o-mini",
                   temperature=0.0)


# --- EMBEDDING MODEL (must be compatible with llm) --- #

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


# --- IMPORT PDF DOCUMENT --- #

pdf_path = "RAG AGENT/math1072.pdf"

# If the PDF file is not found.
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found at {pdf_path}")

# Load the PDF.
pdf_loader = PyPDFLoader(pdf_path)


# Check if the PDF is there.
try:
    pages = pdf_loader.load()
    print(f"PDF has been loaded and has {len(pages)} pages.")
except Exception as e:
    print(f"Error loading PDF: {e}")
    raise


# --- PDF CHUNKING --- #

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, 
                                               chunk_overlap=200)

# Apply to the loaded pages.
pages_split = text_splitter.split_documents(pages)

persist_directory = r"/Users/saraarsenio/Desktop/Personal_Projects/Langgraph/RAG AGENT"
collection_name = "math1072_pre_readings"

# If our collection doesnt exist in the directory, create one.
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)


# Create a chroma database using our embeddings model.
try:
    vectorstore = Chroma.from_documents(
        documents = pages_split,
        embedding = embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    print(f"Vectorstore created with {len(pages_split)} documents.")

except Exception as e:
    print(f"Error creating vectorstore: {e}")
    raise


# RAG Retieval Function 
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # Returns top 5 similar chunks.
)

@tool
def retriever_tool(query: str) -> str:
    """This tool searches and returns the information from the MATH1072 Pre-readings PDF."""

    # Grab top 5 most similar chunks from the PDF.
    docs = retriever.invoke(query)

    # If no similar chunks are found.
    if not docs:
        return "No relevant information found in the document."
    
    # Format the results.
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")

    return "\n\n".join(results)


tools = [retriever_tool]
llm = model.bind_tools(tools)


# --- STATE --- #

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# --- NODE FUNCTIONS --- #

def should_continue(state: AgentState) -> AgentState:
    """ Check if the last message contains a tool call. """

    result = state["messages"][-1]

    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0


# SYSTEM PROMPT 
system_prompt = """
You are an intelligent AI tutor who answers questions about the MATH1072 Pre-readings PDF.
Use the retriever tool to fetch relevant information from the document.
If you need to look up information before asking a follow-up question, use the retriever tool.
Always cite the source within the document of the information you provide.
"""

# TOOLS DICTIONARY 
tools_dic = {our_tool.name: our_tool for our_tool in tools}

def call_llm(state: AgentState):
    """ Call the LLM with the system prompt and user messages. """

    messages = list(state["messages"])
    messages = [SystemMessage(content=system_prompt)] + messages
    
    reply = llm.invoke(messages)

    return {"messages": [reply]}


def take_action(state: AgentState) -> AgentState:
    """ Execute tool calls from the LLM's response. """

    tool_calls = state["messages"][-1].tool_calls
    results = []

    for t in tool_calls:
        print(f"Calling tool: {t["name"]} with query: {t["args"].get("query", "No query provided")}")
    
        # Check if valid tool is present in the tools dictionary.
        if not t["name"] in tools_dic:
            print(f"Tool {t['name']} not found.")
            result = "Incorrect tool name. Please retry and select tool from list of available tools."

        else:
            result = tools_dic[t["name"]].invoke(t["args"].get("query", ''))
            print(f"Result length: {len(str(result))} characters.")

        
        # Append the tool message.
        results.append(ToolMessage(tool_call_id = t["id"], name = t["name"], content = str(result)))

    print("Tools called successfully. Returning to the Model.")

    return {"messages": results}


# --- BUILD GRAPH --- #

graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever_agent", False: END}
)

graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()



# --- RUNNING THE AGENT --- #

def running_agent():
    """ Function to run the RAG agent. """

    print("\n ---- RAG AGENT ----")
    print("This agent answers questions about the MATH1072 Pre-readings PDF.")
    print("Type 'exit' or 'quit' to end the session.\n")


    while True:
        user_input = input("\nYou: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the RAG Agent. Goodbye!")
            break

        # Add user message to state.
        messages = [HumanMessage(content=user_input)]

        result = rag_agent.invoke({"messages": messages})

        print("\n--- Agent Response ---")
        print(result["messages"][-1].content)

if __name__ == "__main__":
    running_agent()



