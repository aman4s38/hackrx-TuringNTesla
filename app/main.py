# app/main.py

import os
import requests
import uuid
from typing import List
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from .ingestion import process_and_get_retriever

# --- Configuration & Models ---
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
HACKRX_TOKEN = "ba3c3f3d3aebe299b4d3574581a1337250ff5d7f435c1e7d8c27ea0cbc30e9b7" # The required token
DOWNLOAD_PATH = "./downloaded_files"
DB_BASE_PATH = "./db"

class HackRxRequest(BaseModel):
    documents: str = Field(...)
    questions: List[str] = Field(...)

class HackRxResponse(BaseModel):
    answers: List[str]

# --- Security Dependency ---
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    if not api_key or not api_key.startswith("Bearer "):
        raise HTTPException(status_code=403, detail="Authorization header missing or invalid")
    token = api_key.split("Bearer ")[1]
    if token == HACKRX_TOKEN:
        return token
    else:
        raise HTTPException(status_code=403, detail="Could not validate credentials")

# --- App Lifespan & FastAPI App ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Server starting up...")
    os.makedirs(DOWNLOAD_PATH, exist_ok=True)
    os.makedirs(DB_BASE_PATH, exist_ok=True)
    global llm_pro, llm_flash
    llm_pro = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=API_KEY, temperature=0)
    llm_flash = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=API_KEY, temperature=0)
    print("Models loaded.")
    yield
    print("Server shutting down.")

app = FastAPI(title="HackRx 6.0 Submission API", lifespan=lifespan)

# --- Helper function ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- Hackathon API Endpoint ---
@app.post("/api/v1/hackrx/run", response_model=HackRxResponse, tags=["HackRx Submission"], dependencies=[Security(get_api_key)])
async def run_submission(request: HackRxRequest):
    document_url = request.documents
    questions = request.questions
    
    try:
        response = requests.get(document_url)
        response.raise_for_status()
        local_filename = os.path.join(DOWNLOAD_PATH, str(uuid.uuid4()) + ".pdf")
        with open(local_filename, 'wb') as f: f.write(response.content)
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download document: {e}")

    document_id = os.path.splitext(os.path.basename(local_filename))[0]
    
    retriever, full_text = process_and_get_retriever(local_filename, document_id)
    if not retriever:
        raise HTTPException(status_code=500, detail="Failed to process document.")

    # --- ROUTER CHAIN ---
    router_template = """You are an expert at routing a user's question. Based on the question, determine if it is a 'Specific Fact' question or a 'General Context' question.
    - 'Specific Fact' questions ask for a precise number, date, name, or a waiting period for a named item (e.g., "What is the waiting period for cataracts?", "What is the limit for room rent?").
    - 'General Context' questions are broader and ask for summaries or conditions (e.g., "Does this policy cover maternity?", "Summarize the organ donor rules.").
    Return only the single word 'Specific Fact' or 'General Context'.
    Question: {question}
    Classification:"""
    router_prompt = ChatPromptTemplate.from_template(router_template)
    router_chain = router_prompt | llm_flash | StrOutputParser()

    # --- FINAL PROMPT ---
    final_prompt_template = """You are a highly precise Q&A engine that answers questions based ONLY on the provided CONTEXT.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    **INSTRUCTIONS FOR YOUR ANSWER:**
    1.  **Strictly Adhere to Context:** Your answer MUST be based exclusively on the information within the provided CONTEXT.
    2.  **CRITICAL REASONING RULE:** To answer the question, you may need to synthesize scattered information. For questions about a 'waiting period' for a specific procedure, you MUST find where the procedure is listed and what waiting period category it falls under.
    3.  **Conciseness vs. Completeness Rule:** Your answer MUST be a single, concise paragraph. 
        - For **definitional questions** (e.g., "How is a 'Hospital' defined?"), you MUST be comprehensive and include all specific criteria listed in the context (like bed counts, staff, etc.).
        - For **all other questions**, you MUST be ruthlessly concise and include ONLY the information that directly answers the question. Do not include extra details.
    4.  **Format:** If the question is objective, you MUST begin your answer with "Yes," or "No,".
    5.  **Data Extraction:** You MUST extract precise numerical values and percentages from the context.
    6.  **Missing Information:** If the information is not in the context, state only: "This information is not available in the provided document."

   

    **ANSWER:**"""
    final_prompt = ChatPromptTemplate.from_template(final_prompt_template)
    
    answers = []
    for question in questions:
        print(f"--- Answering question: {question} ---")
        try:
            route = router_chain.invoke({"question": question})
            print(f"Router choice: {route}")
            
            if "Specific Fact" in route:
                print("Using Path A: Full Context Search")
                context = full_text
                chain = final_prompt | llm_pro | StrOutputParser()
                answer = chain.invoke({"context": context, "question": question})
            else:
                print("Using Path B: Retrieval-Based Search")
                chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | final_prompt
                    | llm_pro
                    | StrOutputParser()
                )
                answer = chain.invoke(question)
            
            answers.append(answer)
        except Exception as e:
            answers.append(f"Error processing question: {e}")
            print(f"Error on question '{question}': {e}")
            
    return HackRxResponse(answers=answers)
