# E-Commerce Business Intelligence — Multi-Agent AI System

A multi-agent AI system built on the **Olist Brazilian E-Commerce Dataset** that combines exploratory data analysis, machine learning, and LLM-powered agents (with RAG over a business policy document) to answer natural-language business questions and generate data-grounded recommendations.

---

## Project Structure

```
project/
├── E_Commerce_Data_Preparation_&_Analysis.ipynb   # EDA, cleaning, ML, exports
├── E_Commerce_Multi-Agent.ipynb   # RAG + multi-agent pipeline
├── streamlit_app.py                      # Deployed chat interface
├── requirements.txt
├── agent_data/
│   ├── business_data.csv
│   ├── customer_segments.csv
│   ├── daily_sales.csv
│   ├── sales_anomalies.csv
│   └── kpis.json
└── Business_Policies.pdf
```

---

## E_Commerce_Data_Preparation_&_Analysis — Data Preprocessing & Machine Learning

Loads the raw Olist dataset (9 CSV files), cleans and merges it, computes business KPIs, and applies machine learning to produce the inputs the agents rely on later.

**Steps:**
1. Download the Olist dataset via `kagglehub`.
2. Load and inspect all 9 raw tables.
3. Clean data: remove duplicates, parse order timestamps.
4. Merge orders, customers, order items, products, sellers, and category translations into a single `business_data` table.
5. Compute core KPIs: total revenue, total orders, total customers, average order value.
6. Analyze category-level sales performance.
7. **Customer segmentation:** K-Means clustering (4 segments) on customer spending and order count.
8. **Anomaly detection:** Isolation Forest on daily revenue to flag unusually high/low sales days.
9. Export all outputs to `agent_data/` (CSV + JSON) for use in Notebook 2.

**Outputs consumed by Notebook 2:**
| File | Contents |
|---|---|
| `business_data.csv` | Merged order/product/customer/seller data |
| `customer_segments.csv` | Per-customer spending, order count, and K-Means segment |
| `daily_sales.csv` | Daily aggregated payment totals |
| `sales_anomalies.csv` | Flagged anomalous sales days |
| `kpis.json` | Precomputed revenue, orders, customers, AOV |

---

## E_Commerce_Multi-Agent — Multi-Agent System (RAG + LLM Agents)

Builds the agent pipeline on top of Notebook 1's outputs and a business policy PDF, using **LangChain** and the **Groq API**.

### Architecture

```
User query
    │
    ▼
Orchestrator Agent  ──► decides which tools to call (SALES / CUSTOMER / PRODUCT / POLICY)
    │
    ▼
Tools (deterministic Python functions, not LLM calls)
 ├─ sales_analysis()      → revenue, orders, AOV, top categories
 ├─ customer_analysis()   → segment breakdown, value index
 ├─ product_analysis()    → category performance, sales anomalies
 └─ business_policy_search() → RAG retrieval over Business_Policies.pdf
    │
    ▼
Insight Agent  ──► analyzes tool results, separates facts from hypotheses
    │
    ▼
Recommendation Agent  ──► generates 1–3 actionable findings, cites policy numbers
    │
    ▼
Final response (decision + insights + recommendations)
```

If the query is policy-only (e.g. *"Can we offer a 25% discount?"*), the pipeline skips straight to a direct **Policy Answer** chain instead of running Insight/Recommendation.

### Key design decisions

- **Two LLMs, different roles:**
  - `llm` (`openai/gpt-oss-120b`) — Insight, Recommendation, and Policy Answer agents. These need deeper reasoning and strict rule-following (no invented numbers, correct policy citations).
  - `llm_fast` (`openai/gpt-oss-20b`, low reasoning) — Orchestrator only, since it just classifies the query into tool names.
- **All arithmetic is precomputed in the tools**, not left to the LLM. This was a deliberate fix after observing the LLM occasionally miscalculating summed percentages (e.g. reporting 45% instead of 40.6% for combined segment revenue) despite explicit prompt rules against it.
- **RAG** over `Business_Policies.pdf` using `sentence-transformers/all-MiniLM-L6-v2` embeddings and a FAISS vector store.
- **Rate-limit handling:** a `safe_invoke()` wrapper retries on Groq `RateLimitError`, parsing both per-minute and per-day limit messages (`Xm Ys` format) to wait the correct amount of time before retrying.

### Testing & Evaluation

The notebook includes:
- 5 core test cases covering single-tool, multi-tool, and policy-only queries.
- 16 additional phrasing variations across all four tool categories.
- Edge cases: empty input, gibberish, off-topic questions, single-word queries, a disallowed request ("delete all customer data" — correctly refused per an AI-governance policy found in the PDF), and Arabic-language queries (handled correctly, including replying in Arabic for policy questions).

---

## Deployment — Streamlit

The multi-agent pipeline from Notebook 2 is wrapped in a chat interface using **Streamlit** and deployed on **Streamlit Community Cloud**.

### Why Streamlit
Alternative platforms (Hugging Face Spaces, Render) were evaluated first:
- Hugging Face Spaces now requires a PRO subscription for any Gradio/Docker Space running on compute (a policy change as of mid-2026); only Static Spaces and ZeroGPU remain free, and ZeroGPU requires GPU-specific code this project doesn't need.
- Render requires card verification even on its free tier.

Streamlit Community Cloud provides a free, persistent, card-free deployment directly from a GitHub repository.

### Running locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Deploying
1. Push the project to a GitHub repository (excluding `.env` — see below).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Create a new app, pointing to `streamlit_app.py`.
4. Under **Advanced settings → Secrets**, add:
   ```
   GROQ_API_KEY = "your_key_here"
   ```
5. Deploy. The app is served at a persistent `[https://your-app-name.streamlit.app](https://ecommercedataanalysisassistant-iee4ivfzjfup25tf5wjqgm.streamlit.app/)` URL.

### Interface
The deployed app presents a chat-style interface (`st.chat_input` / `st.chat_message`) that keeps conversation history for the session and returns the Orchestrator's decision, the Insight Agent's analysis, and the Recommendation Agent's findings for each query.

---

## Environment & Secrets

- API keys are never committed to the repository.
- Locally: stored in a `.env` file (excluded via `.gitignore`).
- In deployment: stored as a Streamlit Cloud secret (`GROQ_API_KEY`).

`.gitignore` should include:
```
.env
__pycache__/
*.pyc
.ipynb_checkpoints/
venv/
```

---

## Known Limitations

- Response time per query is roughly 1–3 minutes due to sequential Groq API calls (Orchestrator → Insight → Recommendation) and Groq's rate limits (tokens per minute / tokens per day on the free tier).
- The LLM occasionally shows minor rounding drift on values it is asked to read verbatim (mitigated by precomputing all combined percentages in the tools rather than relying on the model).
- The anomaly detection in Notebook 1 flags raw daily revenue outliers; since sales volume grew over the dataset's time range, some flagged "low" days early in the range likely reflect the platform's early growth phase rather than genuine weak sales.

---

## Dataset

[Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — 9 relational CSV files covering orders, customers, products, sellers, payments, and reviews from a Brazilian e-commerce marketplace.
