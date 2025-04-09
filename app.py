import os
import json
import pytz
import torch
import pickle
import re
import ast
from datetime import datetime
from typing import TypedDict, Optional, List
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory

from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage
from langchain.prompts import ChatPromptTemplate

# ==== Environment Setup ====
load_dotenv()
timezone = pytz.timezone('Asia/Seoul')
date = datetime.now(timezone).strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs("database/log", exist_ok=True)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    max_tokens=1024,
    temperature=0.7,
)

memory = ConversationBufferMemory(return_messages=True)

# ==== CSV Ingestor ====
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

        if os.path.exists(self.vector_store_path):
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
                return loader.load()
        raise ValueError("No valid data source found.")

    def get_retriever(self, top_k=10):
        return self.vector_store.as_retriever(search_type="mmr", search_kwargs={"k": top_k})

retriever = CSVIngestor().get_retriever()

# ==== Graph State ====
class GraphState(TypedDict, total=False):
    rfp: dict
    is_complete: Optional[bool]
    retrieved_docs: Optional[List]
    draft: Optional[str]
    approved: Optional[bool]

rfp_template = {
    "rpf_title": None,
    "pain_point": None,
    "solution": None,
    "goals": None,
    "funding_size": None,
    "target_users": None,
    "milestones": None,
    "background": None,
    "expected_impact": None,
    "feature_importances": None,
    "about_team": None,
    "media_resources": None
}
required_fields = ["rpf_title", "pain_point", "solution", "goals", "funding_size"]

# ==== Helpers ====
def extract_dict_from_response(text: str) -> dict:
    try:
        if "```" in text:
            text = re.sub(r"```(?:json|python)?", "", text)
            text = text.replace("```", "").strip()
        parsed = ast.literal_eval(text)
        if not isinstance(parsed, dict):
            raise ValueError("Parsed object is not a dictionary.")
        clean_dict = {
            key: (val if str(val).strip().lower() not in ["", "none", "null"] else None)
            for key, val in parsed.items()
        }
        return clean_dict
    except Exception as e:
        print("⚠️ Failed to parse dict from LLM output:", e)
        print("🔍 Raw output:\n", text)
        return {}

# ==== Node: Collect Input ====
prompt_template = ChatPromptTemplate.from_messages([
    ("system", 
     "You're an assistant helping to write an RFP document.\n"
     "During the conversation, try to naturally collect the following fields:\n"
     "- rpf_title\n- pain_point\n- solution\n- goals\n- funding_size\n\n"
     "Don't ask for everything at once. If some fields are still missing, ask follow-up questions.\n"
     "Respond conversationally.")
])

def collect_input(state: GraphState) -> GraphState:
    print("💬 Let's start a conversation to collect RFP information. (max 5 turns)\n")

    system_messages = prompt_template.format_messages()
    memory.chat_memory.add_message(system_messages[0])

    for i in range(5):
        user_input = input(f"[You >] ").strip()
        if not user_input:
            print("⚠️ Please type something.")
            continue
        memory.chat_memory.add_user_message(user_input)
        response = llm.invoke(memory.buffer_as_messages)
        memory.chat_memory.add_ai_message(response.content)
        print(f"[AI >] {response.content}\n")

    memory.chat_memory.add_user_message(
        f"Based on our conversation so far, return a Python dictionary with the following keys:\n\n"
        f"{json.dumps(rfp_template, indent=2)}\n\n"
        f"Use None for anything that wasn't clearly discussed. Return ONLY the dict."
    )
    response = llm.invoke(memory.buffer_as_messages)
    rfp_data = extract_dict_from_response(response.content)
    return {"rfp": rfp_data}

# ==== Node: Check Completion ====
def check_completion(state: GraphState) -> GraphState:
    rfp = state.get("rfp", {})
    is_complete = all(rfp.get(field) is not None for field in required_fields)
    return {"is_complete": is_complete}

# ==== Node: Retrieve Docs ====
def retrieve_docs(state: GraphState) -> GraphState:
    query = state.get("rfp", {}).get("rpf_title", "")
    docs = retriever.invoke(query)

    return {"retrieved_docs": docs}

# ==== Node: Generate Draft ====
def generate_draft(state: GraphState) -> GraphState:
    docs_text = "\n".join([doc.page_content for doc in state.get("retrieved_docs", [])])
    prompt = f"Using the following RFP info and reference documents, write a complete proposal draft.\n\nRFP Info:\n{json.dumps(state.get('rfp', {}), indent=2)}\n\nReference Documents:\n{docs_text}"
    response = llm.invoke(prompt)
    return {"draft": response.content}

# ==== Node: User Check ====
def user_check(state: GraphState) -> GraphState:
    print("\n===== Draft Generated =====\n")
    print(state.get("draft", ""))
    feedback = input("\nAre you satisfied with this draft? (yes/no): ")
    return {"approved": feedback.strip().lower() == "yes"}

# ==== Node: Refine Draft ====
def refine_draft(state: GraphState) -> GraphState:
    feedback = input("What would you like to change or improve?: ")
    prompt = f"Please revise the following draft based on the feedback:\n\nDraft:\n{state.get('draft', '')}\n\nFeedback:\n{feedback}"
    response = llm.invoke(prompt)
    return {"draft": response.content}

# ==== Build LangGraph ====
builder = StateGraph(GraphState)
builder.add_node("collect_input", RunnableLambda(collect_input))
builder.add_node("check_completion", RunnableLambda(check_completion))
builder.add_node("retrieve_docs", RunnableLambda(retrieve_docs))
builder.add_node("generate_draft", RunnableLambda(generate_draft))
builder.add_node("user_check", RunnableLambda(user_check))
builder.add_node("refine_draft", RunnableLambda(refine_draft))

builder.set_entry_point("collect_input")
builder.add_edge("collect_input", "check_completion")

builder.add_conditional_edges(
    "check_completion",
    lambda state: "yes" if state.get("is_complete") else "no",
    {"yes": "retrieve_docs", "no": "collect_input"}
)

builder.add_edge("retrieve_docs", "generate_draft")
builder.add_edge("generate_draft", "user_check")

builder.add_conditional_edges(
    "user_check",
    lambda state: "yes" if state.get("approved") else "no",
    {"yes": END, "no": "refine_draft"}
)

builder.add_edge("refine_draft", "user_check")
graph = builder.compile()

# ==== Run ====
initial_state: GraphState = {
    "rfp": {},
    "is_complete": False,
    "retrieved_docs": [],
    "draft": "",
    "approved": False
}

if __name__ == "__main__":
    final_state = graph.invoke(initial_state)
    print("\n✅ Final Generated RFP Draft:\n")
    print(final_state.get("draft", "No draft was generated."))
