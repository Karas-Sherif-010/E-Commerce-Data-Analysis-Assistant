import os, re, time, json
import pandas as pd
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr

from groq import RateLimitError
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------- تحميل البيانات (لازم تترفع مع المشروع، مش upload يدوي) ----------
required = ["business_data.csv", "customer_segments.csv",
            "daily_sales.csv", "sales_anomalies.csv", "kpis.json"]
business_data     = pd.read_csv("agent_data/business_data.csv")
customer_segments = pd.read_csv("agent_data/customer_segments.csv")
daily_sales       = pd.read_csv("agent_data/daily_sales.csv")
sales_anomalies   = pd.read_csv("agent_data/sales_anomalies.csv")
with open("agent_data/kpis.json") as f:
    kpis = json.load(f)

# ---------- RAG ----------
documents = PyPDFLoader("Business_Policies.pdf").load()
chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ---------- Groq key من environment، مش getpass ----------
from dotenv import load_dotenv
load_dotenv()  # يقرأ .env ويحط القيم في os.environ

GROQ_API_KEY = os.environ["GROQ_API_KEY"]

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, max_tokens=1500, reasoning_effort="medium")
llm_fast = ChatGroq(model="openai/gpt-oss-20b", temperature=0, max_tokens=500, reasoning_effort="low")

def safe_invoke(chain, inputs, retries=8, max_wait=600, verbose=False):
    for attempt in range(retries):
        try:
            return chain.invoke(inputs)
        except RateLimitError as e:
            msg = str(e)
            m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", msg)
            if m:
                minutes = float(m.group(1)) if m.group(1) else 0
                seconds = float(m.group(2))
                wait = min(minutes * 60 + seconds + 1, max_wait)
            else:
                wait = min(15 * (attempt + 1), max_wait)
            if verbose or wait > 60:
                print(f"⏳ Rate limit, waiting {wait/60:.1f} min...")
            time.sleep(wait)
    raise RuntimeError("Rate limit: retries exhausted")

# ---------- انسخ هنا: كل الـ @tool، الـ prompts، run_tools، policy_context، run_business_intelligence ----------
# (نفس الكود اللي في الـ notebook بالظبط، من غير أي تغيير)

# ================== FastAPI ==================
app = FastAPI(title="E-Commerce BI Multi-Agent API")

class Query(BaseModel):
    query: str

@app.post("/analyze")
def analyze(q: Query):
    return run_business_intelligence(q.query)

@app.get("/health")
def health():
    return {"status": "ok"}

# ================== Gradio UI ==================
def gradio_handler(query):
    result = run_business_intelligence(query)
    return (
        result.get("decision", ""),
        result.get("insights", ""),
        result.get("recommendations", ""),
    )

demo = gr.Interface(
    fn=gradio_handler,
    inputs=gr.Textbox(label="Your question", lines=2,
                       placeholder="e.g. Analyze our overall sales performance."),
    outputs=[
        gr.Textbox(label="Orchestrator Decision"),
        gr.Textbox(label="Insights", lines=10),
        gr.Textbox(label="Recommendations", lines=10),
    ],
    title="E-Commerce Business Intelligence — Multi-Agent System",
    examples=[
        "Analyze our overall sales performance.",
        "Which customer segments are most valuable?",
        "Can we offer a 25% discount?",
    ],
)

# ركّب الـ Gradio UI فوق الـ FastAPI app في مسار /gradio
app = gr.mount_gradio_app(app, demo, path="/gradio")