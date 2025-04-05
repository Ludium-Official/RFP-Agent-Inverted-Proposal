import gradio as gr
from dotenv import load_dotenv
import pytz
from datetime import datetime
import logging
import os
import pickle
import torch

from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_community.document_transformers import LongContextReorder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

from langchain_openai import ChatOpenAI
load_dotenv()
API_KEY = os.environ.get("OPENAI_API_KEY")

timezone = pytz.timezone('Asia/Seoul')
date = datetime.now(timezone).strftime("%Y-%m-%d_%H-%M-%S")

os.makedirs("database/log", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"database/log/debug-{date}.log",  encoding="utf-8")
    ]
)

TRIGGER_SYSTEM = """You are a model that matches values to the keys in a dictionary by summarizing the information from the user's query.  
Exclude information for keys that have already been collected. If information cannot be found, the value should be returned as an empty string ("").  
**Never fabricate information that the user has not provided.**  

If the user's input is in a language other than English, translate the information into English before adding it to the dictionary.  

Additionally, increment the num_fill field based on the number of keys updated through the user's query.  

If no changes are made to the dictionary, return the dictionary as it was received.
"""

SYSTEM_MESSAGE = """You are an assistant designed to help with drafting RFPs (Request for Proposals). Follow the structure of the provided example RFP to create new RFPs, ensuring consistency and clarity. 

### Key Instructions:
1. **RFP Structure Adherence**: 
   - Use the example RFP format as a strict template.
   - Maintain professionalism and logical flow in the document.

2. **Handling Missing Information**:
   - Avoid directly mentioning any missing details.
   - Steer the dialogue naturally to gather required information without explicitly stating gaps. For example:
     - "Could you share more about the project timeline?"
     - "What are the specific deliverables you'd like to highlight?"

3. **Casual Conversation Handling**:
   - If the user shifts to casual conversation, accommodate smoothly. 
   - Maintain readiness to pivot back to the task upon user indication.

4. **Chain of Thought (CoT) Approach**:
   - Break down the task systematically:
     - Analyze the provided example RFP for its structure and key components.
     - Identify the necessary inputs for each section.
     - Draft the new RFP step-by-step, filling in the sections with the provided information.
     - Ensure consistency and professionalism throughout.
   - This approach ensures clarity and minimizes errors.

5. **User Collaboration**:
   - Engage the user to clarify or expand on ambiguous points through natural dialogue.
   - Present options when uncertainties arise, allowing the user to make informed choices.

### Additional Information to Gather:
- **Funding Ask**: Details about the funding request.
- **Milestones**: Key milestones for the project.
- **Who We Are**: Introduction to the team or organization.
- **Our Story**: Background and motivation for the project.
- **Our Solution & Milestone**: Solution details and associated project milestones.
- **Impact**: Expected impact of the project.
- **Member Details**: Information about team members.
- **Media**: Relevant media links or content.
"""

reordering = LongContextReorder()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    max_tokens=4096,
    temperature=1.2,
)

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=512,
    memory_key="history",
    return_messages=True
)


def format_docs(docs):
    if len(docs) >= 10:
        docs = reordering.transform_documents(docs)

    return "\n\n".join(doc.page_content for doc in docs)


def load_memory(text):
    return memory.load_memory_variables({})['history']


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
                chunk_size=2048,
                chunk_overlap=100
            )
            doc_splits = self.text_splitter.split_documents(self.docs_list)

            with open(f'{self.text_save_path}/rfp_data.pkl', 'wb') as f:
                pickle.dump(doc_splits, f)
        else:
            with open(f'{self.text_save_path}/rfp_data.pkl', 'rb') as f:
                doc_splits = pickle.load(f)

        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={
                "device": device,
            },
            encode_kwargs={
                "normalize_embeddings": True,
            },
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
            if any(file_name.endswith(".csv") for file_name in os.listdir(self.data_path)):
                loader = CSVLoader(
                    file_path=os.path.join(self.data_path, [file_name for file_name in os.listdir(self.data_path) if file_name.endswith(".csv")][0]),
                    csv_args={'delimiter': ','},
                    encoding='utf-8'
                )
            
                documents_list = loader.load()
        else:
            raise ValueError("No valid data source found.")
        
        return documents_list
    
    def get_retriever(self, top_k=10):
        return self.vector_store.as_retriever(search_type="mmr", search_kwargs={"k": top_k})
    

class CollectData(BaseModel):
    funding_ask: str = Field(default="", description="Details regarding the funding request")
    milestones: str = Field(default="", description="Key milestones for the project")
    who_we_are: str = Field(default="", description="Introduction to the team or organization")
    our_story: str = Field(default="", description="Background and motivation for the project")
    our_solution_milestone: str = Field(default="", description="Details of the solution and project milestones")
    impact: str = Field(default="", description="Expected impact of the project")
    member_details: str = Field(default="", description="Details about team members")
    media: str = Field(default="", description="Relevant media and links")
    num_fill: int = Field(default=0, description="Number of fields filled in the dictionary")


def trigger_chain(pydantic_object):
    parser = JsonOutputParser(pydantic_object=pydantic_object)
    
    trigger_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", TRIGGER_SYSTEM),
            ("user", "### Format: {format_instruction}\n\n### Current Json: {current_json}\n\n### Question: {query}")
        ]
    )

    updated_prompt = trigger_prompt.partial(format_instruction=parser.get_format_instructions())

    chain = updated_prompt | llm | parser

    return chain


class Chain():
    def __init__(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_MESSAGE),
                MessagesPlaceholder(variable_name="history"),
                ("human", "[Provided Information]\n{information}\n\n[Example RFP]\n{context}\n\n[User Query]\n{query}"),
            ]
        )

        self.chain = (
            {
                'information': RunnablePassthrough(),
                'context': RunnablePassthrough(),
                'query': RunnablePassthrough()
            }
            | RunnablePassthrough.assign(history=load_memory)
            | prompt
            | llm
            | StrOutputParser()
        )

    def run(self, query, information, context):
        result = self.chain.invoke({"query": query, "information": information, "context": context})

        memory.save_context(
            {"input": query},
            {"output": result},
        )
        return result


def main():
    pdf_ingestor = CSVIngestor(data_path='database/RFP')
    retriever = pdf_ingestor.get_retriever(top_k=5)
    chain = Chain()

    trigger = trigger_chain(CollectData)

    documents = "The information is insufficient to perform a search."
    current_json = {
        "funding_ask": "",
        "milestones": "",
        "who_we_are": "",
        "our_story": "",
        "our_solution_milestone": "",
        "impact": "",
        "member_details": "",
        "media": "",
        "num_fill": 0
    }

    def chat(user_input, *args, **kwargs):
        nonlocal current_json, documents

        try:
            if current_json['num_fill'] < 8:
                current_json = trigger.invoke({'current_json': current_json, 'query': user_input})
            else:
                pass
        except:
            pass

        logging.info(f"Current JSON before processing: {current_json}")

        if current_json['num_fill'] >= 5:
            documents = format_docs(retriever.invoke(user_input))

            logging.info(f"Current JSON after processing: {documents}")

        response = chain.run(user_input, current_json, documents)
        
        return response

    gr.ChatInterface(
        fn=chat,
        textbox=gr.Textbox(placeholder="입력", container=False, scale=7),
        title="RFP Assistant",
        description="An assistant for drafting RFPs.",
        theme="soft",
        examples = [["What content must be included in an RFP?"], ["How should I write the project timeline and budget?"], ["Please provide the criteria for supplier evaluation."]],
    ).launch(share=True, server_port=5000)


if __name__ == "__main__":
    main()
