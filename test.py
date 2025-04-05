from dotenv import load_dotenv
import pytz
from datetime import datetime
import os
import pickle
import torch
from typing import TypedDict, List, Optional

from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_transformers import LongContextReorder
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

from langchain_core.messages import HumanMessage, AIMessage

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# Load environment variables and basic settings
load_dotenv()
API_KEY = os.environ.get("OPENAI_API_KEY")
timezone = pytz.timezone('Asia/Seoul')
date = datetime.now(timezone).strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs("database/log", exist_ok=True)

reordering = LongContextReorder()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    max_tokens=1024,
    temperature=0.7,
)

# Define RFP state type (includes awaiting_answer, current_field, and asked_fields)
class RFPState(TypedDict):
    messages: List
    filled_fields: dict
    retrieved_docs: Optional[List[str]]
    complete: bool
    awaiting_answer: bool
    current_field: Optional[str]
    asked_fields: List[str]

# Required input fields list
REQUIRED_FIELDS = ["title", "problem_statement", "proposed_solution", "goals", "target_users"]

# CSV Ingestor class: embeds CSV data and builds a vector store
class CSVIngestor:
    def __init__(
            self,
            model_name: str = 'BAAI/bge-m3',
            data_path: str = 'database/RFP',
            text_save_path: str = 'database',
            vector_store_path: str = 'database/faiss.index',
        ):

        self.vector_store_path = vector_store_path
        self.data_path = data_path
        self.text_save_path = text_save_path

        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        if not os.path.isfile(self.text_save_path + '/rfp_data.pkl'):
            self.docs_list = self.get_docs()
            self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                model_name="gpt-4",
                chunk_size=1024,
                chunk_overlap=64
            )
            doc_splits = self.text_splitter.split_documents(self.docs_list)
            with open(f'{self.text_save_path}/rfp_data.pkl', 'wb') as f:
                pickle.dump(doc_splits, f)
        else:
            with open(f'{self.text_save_path}/rfp_data.pkl', 'rb') as f:
                doc_splits = pickle.load(f)

        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
            cache_folder=f'{text_save_path}/model'
        )

        if os.path.exists(self.vector_store_path) and self.vector_store_path is not None:
            self.vector_store = FAISS.load_local(
                self.vector_store_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            self.vector_store = FAISS.from_documents(
                documents=doc_splits,
                embedding=self.embeddings,
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
            )
            self.vector_store.save_local(self.vector_store_path)

    def get_docs(self):
        if os.path.isdir(self.data_path):
            csv_files = [file_name for file_name in os.listdir(self.data_path) if file_name.endswith(".csv")]
            if csv_files:
                loader = CSVLoader(
                    file_path=os.path.join(self.data_path, csv_files[0]),
                    csv_args={'delimiter': ','},
                    encoding='utf-8'
                )
                documents_list = loader.load()
                
                return documents_list
        else:
            raise ValueError("No valid data source found.")
    
    def get_retriever(self, top_k=10):
        return self.vector_store.as_retriever(search_type="mmr", search_kwargs={"k": top_k})

retriever = CSVIngestor().get_retriever()

# ── Dynamic Question Generation and State Node Definitions ──

def generate_next_question(state: RFPState, field: str) -> str:
    """
    Generate a clarifying question to ask the user for additional details for the given field,
    based on the conversation history.
    """
    conversation_history = "\n".join(msg.content for msg in state["messages"])
    
    prompt = ChatPromptTemplate.from_template("""
You are a Hackathon RFP expert.
Based on the conversation history below, please ask a clarifying question to gather more details about the '{field}' field from the user.
Do not provide any recommendations or answer; only ask a question.

Conversation history:
{conversation_history}
"""
)
    
    formatted_prompt = prompt.format_messages(conversation_history=conversation_history, field=field)
    response = llm.invoke(formatted_prompt)

    return response.content.strip()


def fill_fields(state: RFPState) -> RFPState:
    """
    If there are unfilled fields, ask about the ones that haven't been asked yet.
    Once a field has been asked, it won't be asked again.
    If no more questions are pending, assign default values for unfilled fields.
    """
    if state.get("awaiting_answer", False):
        return state

    # List of unfilled fields
    missing_fields = [field for field in REQUIRED_FIELDS if field not in state["filled_fields"]]
    # Unasked fields only
    fields_to_ask = [field for field in missing_fields if field not in state["asked_fields"]]

    if fields_to_ask:
        field_to_ask = fields_to_ask[0]
        question = generate_next_question(state, field_to_ask)
        state["messages"].append(HumanMessage(content=question))
        state["awaiting_answer"] = True
        state["current_field"] = field_to_ask
        state["asked_fields"].append(field_to_ask)
    else:
        # If already asked for all unfilled fields, assign default values
        for field in missing_fields:
            state["filled_fields"][field] = f"(Default: No information provided for {field})"

    return state


def wait_node(state: RFPState) -> RFPState:
    """
    Wait node: simply returns the current state if waiting for user input.
    """
    return state


def retrieve_docs(state: RFPState) -> RFPState:
    """
    Retrieve relevant documents based on the filled fields.
    """
    query = " ".join(state["filled_fields"].values())
    docs = retriever.get_relevant_documents(query)
    state["retrieved_docs"] = [doc.page_content for doc in docs]
    
    return state


def generate_rfp(state: RFPState) -> RFPState:
    """
    Generate the final RFP draft based on the provided (or default) information.
    """
    template = ChatPromptTemplate.from_template("""
You are a Hackathon RFP expert.
Using the following information, please create a clear and persuasive RFP document.

Title: {title}
Problem Statement:
{problem_statement}
Proposed Solution:
{proposed_solution}
Goals:
{goals}
Target Users:
{target_users}
Reference Documents:
{docs}

[Start of Final RFP Draft]
"""
)

    input_data = {}
    for field in REQUIRED_FIELDS:
        if field in state["filled_fields"]:
            input_data[field] = state["filled_fields"][field]
        else:
            input_data[field] = f"(Not provided: {field})"

    input_data["docs"] = "\n\n".join(state["retrieved_docs"] or [])
    response = llm.invoke(template.format_messages(**input_data))

    state["messages"].append(AIMessage(content=response.content))
    state["complete"] = True

    return state

def check_missing_fields(state: RFPState) -> str:
    """
    Check if all required fields have been filled.
    If waiting for user input, return 'wait'; if any field is missing, return 'fill_fields';
    if all fields are filled, proceed to 'retrieve_docs'.
    """
    if state.get("awaiting_answer", False):
        return "wait"
    for field in REQUIRED_FIELDS:
        if field not in state["filled_fields"]:
            return "fill_fields"
        
    return "retrieve_docs"

# ── Build LangGraph StateGraph ──

builder = StateGraph(RFPState)
builder.add_node("fill_fields", fill_fields)
builder.add_node("wait", wait_node)
builder.add_node("retrieve_docs", retrieve_docs)
builder.add_node("generate_rfp", generate_rfp)

builder.set_entry_point("fill_fields")
builder.add_conditional_edges("fill_fields", check_missing_fields)
builder.add_edge("retrieve_docs", "generate_rfp")
builder.add_edge("generate_rfp", END)

graph = builder.compile()

# ── Process User Input Function ──

def process_user_input(state: RFPState, user_input: str) -> RFPState:
    """
    Add the user input to the conversation, store it under the current field,
    and release the awaiting_answer flag.
    """
    state["messages"].append(AIMessage(content=user_input))
    if state.get("current_field"):
        state["filled_fields"][state["current_field"]] = user_input
    state["awaiting_answer"] = False
    state["current_field"] = None
    
    return state

# ── Main Execution Loop ──

def main():
    print("\n🎯 Hackathon RFP Generator")
    user_idea = input("👤 Please enter your idea:\n")
    if user_idea.lower() in ["exit", "quit", "done", "finish"]:
        print("Exiting.")
        return

    state: RFPState = {
        "messages": [HumanMessage(content=user_idea)],
        "filled_fields": {},
        "retrieved_docs": [],
        "complete": False,
        "awaiting_answer": False,
        "current_field": None,
        "asked_fields": []
    }

    while not state.get("complete"):
        state = graph.invoke(state)
        
        if state.get("awaiting_answer", False):
            last_msg = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
            if last_msg:
                print(f"\n🤖 {last_msg.content}")
            else:
                print("\n❌ Failed to generate question")
                break

            user_input = input("👤 Your input: ")
            if user_input.lower() in ["exit", "quit", "done", "finish"]:
                print("\n🚀 Exiting manually. Proceeding to RFP generation...\n")
                break
            state = process_user_input(state, user_input)

    print("\n📄 Final RFP Draft:\n")
    for msg in state["messages"]:
        if isinstance(msg, AIMessage):
            print(msg.content)

if __name__ == "__main__":
    main()
