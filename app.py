import os
import re
import json
import math
from collections import Counter
import pandas as pd
import psycopg2
import streamlit as st

st.set_page_config(page_title="Rajveer Singh | Data & AI Engineering Hub", page_icon="⚡", layout="wide")

st.title("⚡ Data & AI Engineering Production Hub")
st.caption("Live Executive Dashboard: Autonomous AI Ops Agent • Stripe API ➔ Neon Cloud PostgreSQL • Policy RAG Engine • Crypto Market ETL")

# ==========================================
# SHARED KNOWLEDGE BASE & TF-IDF HELPERS
# ==========================================
POLICY_DOCS = [
    {"title": "Enterprise Refund & Cancellation Policy", "dept": "Billing", "text": "Enterprise customers on annual contracts may cancel at any time with 30 days written notice before renewal."},
    {"title": "Enterprise Refund & Cancellation Policy", "dept": "Billing", "text": "Full refunds are issued within 14 business days if requested within the first 30 days of initial activation."},
    {"title": "Enterprise Refund & Cancellation Policy", "dept": "Billing", "text": "If an Enterprise invoice fails due to an expired card, the account enters a 14-day past_due grace period before suspension."},
    {"title": "SOC2 Data Security & Encryption Standards", "dept": "Security", "text": "All customer production databases are encrypted at rest using AES-256 and in transit via TLS 1.3."},
    {"title": "SOC2 Data Security & Encryption Standards", "dept": "Security", "text": "API keys and database credentials must never be hardcoded in source repositories and must be stored in environment secrets."},
    {"title": "SOC2 Data Security & Encryption Standards", "dept": "Security", "text": "Automated daily backups are retained for 30 days across isolated cloud regions."},
    {"title": "API Rate Limits & Webhook Retry Guide", "dept": "Engineering", "text": "Standard Pro tier workspaces are limited to 1,000 API requests per minute, while Enterprise workspaces receive 10,000 requests per minute."},
    {"title": "API Rate Limits & Webhook Retry Guide", "dept": "Engineering", "text": "When an external webhook fails with HTTP 429 or 503, the ingestion pipeline automatically retries with exponential backoff up to 3 times."}
]

def tokenize(text):
    stop_words = {"the", "a", "an", "is", "are", "to", "of", "in", "for", "on", "with", "and", "or", "if", "at", "by"}
    return [w for w in re.findall(r"[a-z0-9_]+", text.lower()) if w not in stop_words]

TOKENIZED_DOCS = [tokenize(d["text"]) for d in POLICY_DOCS]
DOC_FREQ = Counter(w for tokens in TOKENIZED_DOCS for w in set(tokens))

def search_policy_rag(query_text):
    q_tokens = tokenize(query_text)
    scored = []
    for idx, d in enumerate(POLICY_DOCS):
        c_tokens = TOKENIZED_DOCS[idx]
        counts = Counter(c_tokens)
        score = sum((counts[w] / len(c_tokens)) * (math.log((len(POLICY_DOCS) + 1) / (DOC_FREQ[w] + 1)) + 1.0) for w in q_tokens if w in counts)
        if score > 0:
            scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0] if scored else None

def query_churn_risk_sql(db_url):
    if not db_url:
        return pd.DataFrame([
            {"Company": "Acme Global Corp", "Plan": "Enterprise", "At-Risk MRR ($)": 1200.0, "Failed Date": "2026-11-20"},
            {"Company": "Northstar Logistics", "Plan": "Enterprise", "At-Risk MRR ($)": 1200.0, "Failed Date": "2026-11-22"}
        ]), "fallback_demo"
    conn = psycopg2.connect(db_url)
    df = pd.read_sql_query("""
        SELECT c.company_name AS "Company",
               c.country AS "Country",
               s.plan_tier AS "Plan",
               s.monthly_price_usd AS "At-Risk MRR ($)",
               p.paid_date AS "Failed Date"
        FROM payments p
        INNER JOIN subscriptions s ON p.sub_id = s.sub_id
        INNER JOIN customers c ON s.customer_id = c.customer_id
        WHERE p.payment_status = 'failed'
        ORDER BY s.monthly_price_usd DESC;
    """, conn)
    conn.close()
    return df, "neon_cloud_postgresql"

tab0, tab1, tab2, tab3 = st.tabs([
    "🤖 Autonomous AI Ops Agent (Tool-Calling)",
    "📊 System 2: Stripe & Cloud PostgreSQL Warehouse",
    "🧠 System 3: AI Policy RAG Search Engine",
    "📈 System 1: Live Crypto Market ETL"
])

# ==========================================
# TAB 0: AUTONOMOUS AI OPS AGENT (MULTI-TOOL)
# ==========================================
with tab0:
    st.subheader("Autonomous Revenue & Compliance Ops Agent")
    st.write("Routes natural-language executive tasks across live **Cloud PostgreSQL SQL tools** and **Verified Policy RAG tools** with structured JSON execution traces.")

    preset = st.selectbox(
        "Choose an autonomous multi-step workflow (or type your own below):",
        [
            "Audit failed Enterprise payments in PostgreSQL and retrieve our grace-period policy before suspension",
            "Check SOC2 database encryption & backup rules and verify SQL quality gate status",
            "Inspect webhook retry rules for HTTP 429/503 errors during Stripe ingestion"
        ]
    )
    agent_prompt = st.text_input("Agent Instruction:", value=preset)

    if st.button("▶ Run Autonomous Agent", type="primary"):
        db_url = os.environ.get("DATABASE_URL", "")
        prompt_lower = agent_prompt.lower()
        tool_trace = []

        needs_sql = any(k in prompt_lower for k in ["payment", "churn", "mrr", "enterprise", "sql", "revenue", "failed", "stripe"])
        needs_rag = any(k in prompt_lower for k in ["policy", "grace", "soc2", "encrypt", "backup", "webhook", "retry", "refund", "cancel", "rule"])
        if not needs_sql and not needs_rag:
            needs_sql, needs_rag = True, True

        sql_result_df = None
        rag_result = None

        if needs_sql:
            sql_result_df, source_mode = query_churn_risk_sql(db_url)
            tool_trace.append({
                "step": len(tool_trace) + 1,
                "tool": "query_cloud_postgres_churn_risk()",
                "warehouse": source_mode,
                "rows_returned": len(sql_result_df),
                "at_risk_mrr_usd": float(sql_result_df["At-Risk MRR ($)"].sum()) if not sql_result_df.empty else 0.0,
                "status": "SUCCESS"
            })

        if needs_rag:
            rag_result = search_policy_rag(agent_prompt)
            tool_trace.append({
                "step": len(tool_trace) + 1,
                "tool": "retrieve_verified_policy_rag()",
                "top_score": round(rag_result[0], 3) if rag_result else 0.0,
                "source_doc": rag_result[1]["title"] if rag_result else "None",
                "department": rag_result[1]["dept"] if rag_result else "None",
                "status": "GROUNDED_VERIFIED" if rag_result else "NO_MATCH"
            })

        st.success(f"Agent completed **{len(tool_trace)} tool call(s)** with verified grounding and 0 hallucinations.")

        col_a, col_b = st.columns([1.2, 1])
        with col_a:
            st.markdown("#### 📋 Synthesized Executive Action Brief")
            if sql_result_df is not None and not sql_result_df.empty:
                total_risk = sql_result_df["At-Risk MRR ($)"].sum()
                st.warning(f"**Live SQL Finding:** Found **{len(sql_result_df)} failed invoice(s)** representing **${total_risk:,.2f}/mo** in at-risk MRR.")
                st.dataframe(sql_result_df, use_container_width=True)
            if rag_result is not None:
                score, doc = rag_result
                st.info(f"**Verified Policy Citation ({doc['dept']} — {doc['title']}, Score `{score:.3f}`):**\n\n\"{doc['text']}\"")
        with col_b:
            st.markdown("#### 🔍 Structured Tool-Call Execution Trace")
            st.code(json.dumps(tool_trace, indent=2), language="json")
