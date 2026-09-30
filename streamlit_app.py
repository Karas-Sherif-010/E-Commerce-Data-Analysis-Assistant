import streamlit as st
import os, re, time, json
import pandas as pd
import numpy as np

from groq import RateLimitError
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="E-Commerce BI Multi-Agent", page_icon="📊")


@st.cache_resource
def load_everything():
    business_data = pd.read_csv("agent_data/business_data.csv")
    customer_segments = pd.read_csv("agent_data/customer_segments.csv")
    daily_sales = pd.read_csv("agent_data/daily_sales.csv")
    sales_anomalies = pd.read_csv("agent_data/sales_anomalies.csv")
    with open("agent_data/kpis.json") as f:
        kpis = json.load(f)

    documents = PyPDFLoader("Business_Policies.pdf").load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100
    ).split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    return business_data, customer_segments, daily_sales, sales_anomalies, kpis, vectorstore, retriever


business_data, customer_segments, daily_sales, sales_anomalies, kpis, vectorstore, retriever = load_everything()

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, max_tokens=1500, reasoning_effort="medium")
llm_fast = ChatGroq(model="openai/gpt-oss-20b", temperature=0, max_tokens=500, reasoning_effort="low")


def safe_invoke(chain, inputs, retries=8, max_wait=600):
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
            time.sleep(wait)
    raise RuntimeError("Rate limit: retries exhausted")


# ---------------- Tools ----------------

@tool
def business_policy_search(query: str) -> str:
    """Search the business policy knowledge base for relevant rules and policies."""
    results = retriever.invoke(query)
    if not results:
        return "No relevant business policy was found."
    return "\n\n".join(d.page_content for d in results)


@tool
def sales_analysis() -> str:
    """Analyze overall sales performance and return important business KPIs."""
    cat = (business_data.groupby("product_category_name_english")["price"]
           .sum().sort_values(ascending=False))
    items_total = business_data["price"].sum()
    top5 = cat.head(5)
    repeat_n = int((customer_segments["orders"] > 1).sum())

    return json.dumps({
        "total_revenue": round(kpis["total_revenue"], 2),
        "total_orders": kpis["total_orders"],
        "total_customers": kpis["total_customers"],
        "average_order_value": round(kpis["average_order_value"], 2),
        "repeat_customers": repeat_n,
        "repeat_customers_pct": round(repeat_n / len(customer_segments) * 100, 2),
        "top_categories_item_sales": top5.round(2).to_dict(),
        "top_categories_share_of_item_sales_pct": (top5 / items_total * 100).round(2).to_dict(),
        "top5_combined_item_sales": round(top5.sum(), 2),
        "top5_combined_share_of_item_sales_pct": round(top5.sum() / items_total * 100, 2),
        "note": "Shares are % of product item sales (sum of price, excl. freight). "
                "They are NOT % of total_revenue, which comes from payments and includes freight.",
    })


@tool
def customer_analysis() -> str:
    """Analyze customer segments and spending behavior."""
    s = (customer_segments.groupby("segment")
         .agg(customers=("customer_unique_id", "count"),
              total_spending=("spending", "sum"),
              average_spending=("spending", "mean"),
              average_orders=("orders", "mean"))
         .reset_index())
    s["pct_customers"] = (s["customers"] / s["customers"].sum() * 100).round(1)
    s["pct_revenue"] = (s["total_spending"] / s["total_spending"].sum() * 100).round(1)
    s["value_index"] = (s["pct_revenue"] / s["pct_customers"]).round(2)

    high_value = s[s["value_index"] > 1]

    result = {
        "segments": s.round(2).to_dict(orient="records"),
        "high_value_segments_combined_pct_customers": round(high_value["pct_customers"].sum(), 2),
        "high_value_segments_combined_pct_revenue": round(high_value["pct_revenue"].sum(), 2),
        "note": "high_value_segments are those with value_index > 1; combined figures are precomputed, do not recompute.",
    }
    return json.dumps(result)


@tool
def product_analysis() -> str:
    """Analyze product category performance and unusual sales days."""
    cat = (business_data.groupby("product_category_name_english")["price"]
           .sum().sort_values(ascending=False))
    items_total = business_data["price"].sum()
    top = cat.head(5)

    an = sales_anomalies.copy()
    median_day = daily_sales["payment_value"].median()
    an["direction"] = np.where(an["payment_value"] > median_day, "high", "low")
    an["month"] = pd.to_datetime(an["date"]).dt.strftime("%Y-%m")

    high8 = an[an["direction"] == "high"].sort_values("payment_value", ascending=False).head(8)
    low5 = an[an["direction"] == "low"].sort_values("payment_value").head(5)

    return json.dumps({
        "top_categories_item_sales": top.round(2).to_dict(),
        "top5_combined_share_of_item_sales_pct": round(top.sum() / items_total * 100, 2),
        "data_range": [str(daily_sales["date"].min()), str(daily_sales["date"].max())],
        "median_daily_payment": round(median_day, 2),
        "number_of_sales_anomalies": len(an),
        "high_anomaly_days": int((an["direction"] == "high").sum()),
        "low_anomaly_days": int((an["direction"] == "low").sum()),
        "anomalies_per_month": an["month"].value_counts().sort_index().to_dict(),
        "top_high_anomalous_days": high8[["date", "payment_value"]].to_dict(orient="records"),
        "lowest_anomalous_days": low5[["date", "payment_value"]].to_dict(orient="records"),
        "note": "Anomaly values are TOTAL DAILY payments (positive numbers), not single transactions. "
                "Direction is relative to the median daily payment. "
                "Shares are % of product item sales, not total revenue.",
    }, default=str)


# ---------------- Orchestrator ----------------

orchestrator_prompt = ChatPromptTemplate.from_template("""
You are the Orchestrator Agent of an e-commerce Business Intelligence system.
Decide which tools are required for the user's request.

Tools:
SALES    - revenue, orders, customers, average order value, top categories
CUSTOMER - customer segments, spending behavior
PRODUCT  - product category performance, sales anomalies
POLICY   - business rules: returns, discounts, shipping, escalation

Return ONLY the tool names separated by commas.

Examples:
"How are our sales performing?" -> SALES
"Which customer segments are most valuable?" -> CUSTOMER
"Why should we investigate unusual sales?" -> PRODUCT
"Can we give a 25% discount?" -> POLICY
"Analyze sales and recommend actions according to our policies." -> SALES, PRODUCT, POLICY

User Request:
{query}
""")

orchestrator_chain = orchestrator_prompt | llm_fast | StrOutputParser()


def run_tools(decision, query):
    d = decision.upper()
    results = {}
    if "SALES" in d:    results["sales"] = sales_analysis.invoke({})
    if "CUSTOMER" in d: results["customer"] = customer_analysis.invoke({})
    if "PRODUCT" in d:  results["product"] = product_analysis.invoke({})
    if "POLICY" in d:   results["policy"] = business_policy_search.invoke({"query": query})
    return results


# ---------------- Insight Agent ----------------

insight_prompt = ChatPromptTemplate.from_template("""
You are the Insight Agent in an e-commerce Business Intelligence system.

Rules:
- Identify important business patterns and explain unusual results.
- Do not invent numbers. Use ONLY numbers that appear in the results.
- Do not add, subtract or derive new numbers, percentages, or counts.
- If the results contain high_value_segments_combined_pct_customers / _pct_revenue, use those values directly; do not sum individual segment percentages yourself.
- Do not classify an anomaly as fraud or an error without evidence.
- Clearly separate facts from interpretations.
- Explanations that use outside knowledge (holidays, campaigns) must be labeled "Hypothesis".
- Segments are unnamed numbers; describe them by their numbers only.
- Do not claim causes or links between metrics unless the results show them (AOV is revenue/orders, so it never "confirms" anything).
- Do not mention margins or profitability; there is no margin data.
- Category shares are % of product item sales. Never call them "% of total revenue".
- Anomaly values are total daily payments (positive), not single transactions.
- Low anomalies near the start of data_range may reflect low early volume rather than weak sales; label that as Hypothesis.
- Say "bulk of revenue" only if a share is above 50%.
- value_index = pct_revenue / pct_customers; above 1 means the segment earns more than its size.
- Anomalies can be "high" or "low" days; treat them separately.
- Category totals and anomaly days are independent results; do not link them.
- Quote dates exactly as given, and count only items actually listed.
- Keep the answer under 250 words.

Analytical Results:
{results}

Provide a concise business insight report.
""")

insight_chain = insight_prompt | llm | StrOutputParser()


# ---------------- Recommendation Agent ----------------

recommendation_prompt = ChatPromptTemplate.from_template("""
You are the Recommendation Agent.
Generate practical recommendations based on the results, insights, and policies.

Rules:
- Do not invent information; recommendations must be supported by the data.
- Do not create findings about plans or risks that are not in the data.
- If the results contain high_value_segments_combined_pct_customers / _pct_revenue, use those values directly; do not sum individual segment percentages yourself.
- Use business policies when applicable, and cite the policy number.
- Policies are rules, never describe a policy scenario as something that happened.
- "Required Approval": write any approval, escalation or review that a cited policy requires (with its number). Otherwise write "None".
- Do not treat anomalies as confirmed problems.
- Do not invent thresholds, alert levels, or numbers. Use ONLY numbers that appear in the results; never derive new ones.
- Do not claim margins, pricing consistency, or causes.
- If a finding needs data we don't have (inventory, margins), say "Needs data" instead of assuming.
- Give at most 3 findings, under 250 words total.

Analytical Results:
{results}

Business Insights:
{insights}

Relevant Policies:
{policies}

For each finding write: Key Finding, Recommended Action, Business Reason, Required Approval.
""")

recommendation_chain = recommendation_prompt | llm | StrOutputParser()


# ---------------- Policy Answer chain ----------------

policy_answer_prompt = ChatPromptTemplate.from_template("""
Answer the user's question using ONLY the policy text below.
Cite the policy number (e.g. 3.2). If the policy text does not cover it, say so.
Lead with the direct answer. If approval or review is required, say that first instead of "Yes".
Policies are rules, not observations: do not describe events or problems.
Keep it under 120 words.

Policy text:
{policies}

Question: {query}
""")

policy_answer_chain = policy_answer_prompt | llm | StrOutputParser()


# ---------------- Policy retrieval ----------------

POLICY_TOPICS = [
    "discount limits and manager approval",
    "escalation rules for repeated anomalies",
    "high-value customer priority",
    "inventory replenishment for fast-moving products",
]


def policy_context(query=None, k=2):
    seen, out = set(), []
    for topic in POLICY_TOPICS:
        for d in vectorstore.similarity_search(topic, k=k):
            if d.page_content not in seen:
                seen.add(d.page_content)
                out.append(d.page_content)
    return "\n\n".join(out)


# ---------------- Pipeline ----------------

def run_business_intelligence(query):
    decision = safe_invoke(orchestrator_chain, {"query": query})

    wants_policy = "POLICY" in decision.upper()
    results = run_tools(decision, query)

    policies = results.pop("policy", None) or business_policy_search.invoke({"query": query})

    if not results:
        answer = safe_invoke(policy_answer_chain, {"policies": policies, "query": query})
        return {"decision": decision, "results": {"policy": policies},
                "insights": "", "recommendations": answer}

    results_text = json.dumps(results, default=str)

    insights = safe_invoke(insight_chain, {"results": results_text})

    if wants_policy:
        policies = policy_context(query)
    else:
        extra = business_policy_search.invoke({"query": insights[:400]})
        if extra != policies:
            policies = f"{policies}\n\n{extra}"

    recommendations = safe_invoke(recommendation_chain, {
        "results": results_text, "insights": insights, "policies": policies})

    return {"decision": decision, "results": results,
            "insights": insights, "recommendations": recommendations}


# ================== Streamlit UI (Chat) ==================

st.title("📊 E-Commerce Business Intelligence — Multi-Agent System")
st.caption("Ask questions about sales, customers, products, or business policies.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if user_query := st.chat_input("e.g. Analyze our overall sales performance."):
    # Show user's message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Run the pipeline and show assistant's reply
    with st.chat_message("assistant"):
        with st.spinner("Agents are working... this may take a few minutes"):
            result = run_business_intelligence(user_query)

        reply_parts = [f"**Orchestrator Decision:** {result.get('decision')}"]

        if result.get("insights"):
            reply_parts.append(f"**Insights:**\n\n{result.get('insights')}")

        reply_parts.append(f"**Recommendations:**\n\n{result.get('recommendations')}")

        full_reply = "\n\n---\n\n".join(reply_parts)
        st.markdown(full_reply)

    st.session_state.messages.append({"role": "assistant", "content": full_reply})
