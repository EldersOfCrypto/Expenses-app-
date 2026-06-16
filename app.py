import streamlit as st
import pdfplumber
import pandas as pd
import anthropic
import json
import io
import uuid
import plotly.express as px
from dotenv import load_dotenv
import os
from datetime import date, timedelta
from pathlib import Path

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Business Expenses Tracker",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
TRANSACTIONS_FILE = DATA_DIR / "transactions.csv"
LOCKS_FILE = BASE_DIR / "locked_periods.json"

load_dotenv(BASE_DIR / ".env")

try:
    api_key = st.secrets.get("ANTHROPIC_API_KEY")
except Exception:
    api_key = None
if not api_key:
    api_key = os.getenv("ANTHROPIC_API_KEY")

# ── Categories ────────────────────────────────────────────────────────────────
EXPENSE_CATEGORIES = [
    "Rent", "Utilities", "Payroll", "Food & Beverages", "Supplies",
    "Equipment", "Marketing", "Transportation", "Bank Fees", "Taxes",
    "Insurance", "Repairs & Maintenance", "Software & Subscriptions",
    "Professional Services", "Transfer / Internal", "Other Expense",
]
INCOME_CATEGORIES = [
    "Sales Revenue", "Transfer In", "Loan / Financing",
    "Refund Received", "Investment", "Other Income",
]
TRANSFER_CATEGORIES = {"Transfer / Internal", "Transfer In"}
FIXED_CATEGORIES    = {"Rent", "Insurance", "Software & Subscriptions", "Bank Fees", "Professional Services"}
OTHER_CATEGORIES    = {"Other Expense", "Other Income"}
ALL_CATEGORIES      = EXPENSE_CATEGORIES + INCOME_CATEGORIES
MICRO_FEE_MAX       = 15.0

client = anthropic.Anthropic(api_key=api_key) if api_key else None

# ── Persistence ───────────────────────────────────────────────────────────────
def load_transactions() -> pd.DataFrame:
    if TRANSACTIONS_FILE.exists():
        df = pd.read_csv(TRANSACTIONS_FILE, dtype={"notes": str, "tx_id": str})
        for col, val in [("notes", ""), ("bank", ""), ("source", ""), ("tx_id", "")]:
            if col not in df.columns:
                df[col] = val
        df["notes"] = df["notes"].fillna("")
        df = _ensure_ids(df)
        return df
    return pd.DataFrame()

def save_transactions(df: pd.DataFrame):
    if not df.empty:
        df.to_csv(TRANSACTIONS_FILE, index=False)
    elif TRANSACTIONS_FILE.exists():
        TRANSACTIONS_FILE.unlink()

def _ensure_ids(df: pd.DataFrame) -> pd.DataFrame:
    if "tx_id" not in df.columns:
        df["tx_id"] = [str(uuid.uuid4())[:8] for _ in range(len(df))]
    else:
        mask = df["tx_id"].isna() | (df["tx_id"] == "")
        df.loc[mask, "tx_id"] = [str(uuid.uuid4())[:8] for _ in range(mask.sum())]
    return df

def _apply_edits_back(full_df: pd.DataFrame, edited_df: pd.DataFrame) -> pd.DataFrame:
    """Merge edited rows back into full_df by tx_id (handles date filtering)."""
    if edited_df.empty or "tx_id" not in edited_df.columns:
        return full_df
    for _, row in edited_df.iterrows():
        mask = full_df["tx_id"] == row["tx_id"]
        if mask.any():
            for col in ["category", "notes", "type", "description", "amount"]:
                if col in row.index:
                    full_df.loc[mask, col] = row[col]
    return full_df

# ── PDF / AI ──────────────────────────────────────────────────────────────────
def extract_text_from_pdf(pdf_file) -> str:
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
            for table in page.extract_tables():
                for row in table:
                    if row:
                        text += " | ".join(str(c) for c in row if c) + "\n"
    return text

def parse_transactions_with_ai(text: str):
    prompt = f"""You are analyzing a financial statement from Mexico.

Identify:
- "bank_name": bank or card issuer (BBVA, Bajio, Inbursa, AMEX, Santander, Banamex, etc.)
- "statement_type": "Credit Card" or "Bank Statement"

Extract ALL transactions with these categorization rules:
- TRASPASO CUENTAS PROPIAS, TRASPASO ENTRE CUENTAS, ENVIO DINERO → "Transfer / Internal", expense
- Payments to credit cards (PAGO TARJETA, PAGO TC, PAGO VISA, PAGO MASTERCARD, PAGO AMEX, LIQUIDACION TARJETA) → "Transfer / Internal", expense
- SPEI/TRASPASO from own entities (ELDURADU SERVICIOS INTEGRALES) → "Transfer In", income
- Money received INTO BBVA from Bajio/Inbursa/SPEI → "Transfer In", income
- Money SENT TO BBVA from another account → "Transfer / Internal", expense
- Other incoming SPEI from unknown sources → "Transfer In", income
- Stripe (STRIPE, STRIPE PAYMENTS MEXICO) → "Sales Revenue", income
- POS settlements (LIQUIDACION ADQUIRENTE, DEPOSITO ADQUIRENTE) → "Sales Revenue", income — money processor pays to merchant
- POS fees (TASA DE DESCUENTO, IVA TASA DE DESCUENTO, COMISION ADQUIRENTE, APLI TASA, IVA TASA) → "Bank Fees", expense
- Payroll (NOMINA, PAGO NOMINA, DISPERSION NOMINA, PAGO CUENTA DE TERCERO) → "Payroll", expense
- SPEI to a person's full name → "Payroll", expense
- Government payroll (SUA, IMSS, INFONAVIT, FONACOT) → "Payroll", expense
- Landscaping/garden/plants (JARDINES, VIVERO, PLANTAS, ARALIA) → "Repairs & Maintenance", expense
- Debit card merchant purchases → use real merchant name, pick best expense category
- Do NOT label merchant purchases as "Transfer / Internal"

Return JSON only:
{{
  "bank_name": "...",
  "statement_type": "...",
  "transactions": [
    {{
      "date": "DD/MM/YYYY",
      "description": "merchant or description",
      "amount": <positive=out/expense, negative=in/income>,
      "type": "expense" or "income",
      "category": "<one of the categories>"
    }}
  ]
}}

Categories for expenses: {", ".join(EXPENSE_CATEGORIES)}
Categories for income: {", ".join(INCOME_CATEGORIES)}

Statement:
{text}"""

    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=16384,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = msg.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    result = json.loads(raw)
    return result["transactions"], result["statement_type"], result.get("bank_name", "Unknown")

# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_date_col(series):
    return pd.to_datetime(series, format="%d/%m/%Y", dayfirst=True, errors="coerce")

def export_excel(df: pd.DataFrame) -> bytes:
    df_exp = df[df["type"] == "expense"]
    df_inc = df[df["type"] == "income"]
    real_exp = df_exp[~df_exp["category"].isin(TRANSFER_CATEGORIES)]
    real_inc = df_inc[~df_inc["category"].isin(TRANSFER_CATEGORIES)]

    drop = ["type", "tx_id"]

    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        real_exp.drop(columns=drop, errors="ignore").to_excel(w, sheet_name="Expenses", index=False)
        real_inc.drop(columns=drop, errors="ignore").to_excel(w, sheet_name="Income", index=False)

        cat_sum = (real_exp.groupby("category")["amount"]
                   .agg(Total="sum", Count="count").reset_index()
                   .sort_values("Total", ascending=False))
        cat_sum.to_excel(w, sheet_name="By Category", index=False)

        bank_e = real_exp.groupby("bank")["amount"].sum().rename("Expenses")
        bank_i = real_inc.groupby("bank")["amount"].apply(lambda x: abs(x.sum())).rename("Income")
        pd.concat([bank_e, bank_i], axis=1).fillna(0).to_excel(w, sheet_name="By Bank")

        tmp = df.copy()
        tmp["_date"] = parse_date_col(tmp["date"])
        tmp = tmp.dropna(subset=["_date"])
        tmp["Month"] = tmp["_date"].dt.to_period("M").astype(str)
        real_t = tmp[~tmp["category"].isin(TRANSFER_CATEGORIES)]
        me = real_t[real_t["type"] == "expense"].groupby("Month")["amount"].sum()
        mi = real_t[real_t["type"] == "income"].groupby("Month")["amount"].apply(lambda x: abs(x.sum()))
        monthly = pd.DataFrame({"Expenses": me, "Income": mi}).fillna(0)
        monthly["Net"] = monthly["Income"] - monthly["Expenses"]
        monthly.to_excel(w, sheet_name="By Month")

        re = real_exp["amount"].sum()
        ri = abs(real_inc["amount"].sum())
        n  = ri - re
        fe = real_exp[real_exp["category"].isin(FIXED_CATEGORIES)]["amount"].sum()
        ve = re - fe
        tr = df_exp[df_exp["category"].isin(TRANSFER_CATEGORIES)]["amount"].sum()
        summary = pd.DataFrame({
            "": ["Real Expenses", "Real Income", "Net Balance", "Fixed Costs", "Variable Costs", "Transfers (excluded)"],
            "Amount": [f"${re:,.2f}", f"${ri:,.2f}", f"${n:,.2f}", f"${fe:,.2f}", f"${ve:,.2f}", f"${tr:,.2f}"],
        })
        summary.to_excel(w, sheet_name="Summary", index=False)
    return out.getvalue()

def load_locked_periods():
    if LOCKS_FILE.exists():
        with open(LOCKS_FILE) as f:
            return json.load(f)
    return []

def save_locked_periods(periods):
    with open(LOCKS_FILE, "w") as f:
        json.dump(periods, f, indent=2)

def find_duplicates(df_new, df_existing) -> int:
    if df_existing.empty:
        return 0
    count = 0
    for _, row in df_new.iterrows():
        if not df_existing[
            (df_existing["date"] == row["date"]) &
            (abs(df_existing["amount"] - row["amount"]) < 0.01) &
            (df_existing["description"] == row["description"])
        ].empty:
            count += 1
    return count

# ── Session State ─────────────────────────────────────────────────────────────
_defaults = [
    ("transactions",    load_transactions()),
    ("upload_key",      0),
    ("processed_files", []),
    ("upload_errors",   []),
    ("dup_warnings",    []),
    ("confirm_clear",   False),
]
for k, v in _defaults:
    if k not in st.session_state:
        st.session_state[k] = v

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #f8fafc; }
.header-box {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
    padding: 1.6rem 2rem 1.3rem;
    border-radius: 14px;
    margin-bottom: 1.4rem;
}
.header-box h1 { color: #fff; margin: 0; font-size: 1.8rem; font-weight: 700; }
.header-box p  { color: #a0aec0; margin: 0.25rem 0 0; font-size: 0.9rem; }
.review-pill {
    background: #fed7d7; color: #c53030;
    padding: 2px 10px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 700;
    display: inline-block; margin-left: 8px; vertical-align: middle;
}
.welcome-card {
    background: #fff; border-radius: 14px;
    padding: 1.6rem 1.4rem; text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    height: 100%;
}
.tip-box {
    background: #ebf8ff; border-left: 4px solid #3182ce;
    border-radius: 8px; padding: 0.9rem 1.2rem; margin-top: 1.2rem;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:

    # ── Upload ────────────────────────────────────────────────────────────────
    with st.expander("📤 Upload Statement", expanded=True):
        for err in st.session_state.upload_errors:
            st.error(err)
        for w in st.session_state.dup_warnings:
            st.warning(w)
        st.session_state.upload_errors = []
        st.session_state.dup_warnings  = []

        uploaded_files = st.file_uploader(
            "PDF files", type="pdf", accept_multiple_files=True,
            key=f"uploader_{st.session_state.upload_key}",
            label_visibility="collapsed",
        )
        if uploaded_files and st.button("Process Statement", type="primary", use_container_width=True):
            errors, dup_warns = [], []
            for f in uploaded_files:
                with st.spinner(f"Processing {f.name}…"):
                    try:
                        text = extract_text_from_pdf(f)
                        txns, stype, bank = parse_transactions_with_ai(text)
                        df_new = pd.DataFrame(txns)
                        if "type" not in df_new.columns:
                            df_new["type"] = df_new["amount"].apply(lambda x: "income" if x < 0 else "expense")
                        df_new["source"] = stype
                        df_new["bank"]   = bank
                        if "notes" not in df_new.columns:
                            df_new["notes"] = ""
                        df_new = _ensure_ids(df_new)

                        dupes = find_duplicates(df_new, st.session_state.transactions)
                        if dupes:
                            dup_warns.append(f"⚠️ {f.name}: {dupes} possible duplicate(s) added.")

                        st.session_state.transactions = (
                            df_new if st.session_state.transactions.empty
                            else pd.concat([st.session_state.transactions, df_new], ignore_index=True)
                        )
                        save_transactions(st.session_state.transactions)
                        st.session_state.processed_files.append(
                            {"name": f.name, "type": stype, "bank": bank, "count": len(txns)}
                        )
                    except Exception as e:
                        errors.append(f"{f.name}: {e}")

            st.session_state.upload_errors = errors
            st.session_state.dup_warnings  = dup_warns
            st.session_state.upload_key   += 1
            st.rerun()

    # ── Manual Entry ──────────────────────────────────────────────────────────
    with st.expander("✏️ Add Manual Entry"):
        with st.form("manual_entry", clear_on_submit=True):
            m_date  = st.date_input("Date", value=date.today(), format="DD/MM/YYYY")
            m_desc  = st.text_input("Description", placeholder="e.g. Coca Cola, gas, lunch…")
            m_amt   = st.number_input("Amount (MXN)", min_value=0.0, step=1.0)
            m_type  = st.radio("Type", ["Expense", "Income"], horizontal=True)
            m_cats  = [c for c in (EXPENSE_CATEGORIES if m_type == "Expense" else INCOME_CATEGORIES)
                       if c not in TRANSFER_CATEGORIES]
            m_cat   = st.selectbox("Category", m_cats)
            m_pay   = st.selectbox("Payment method", ["Cash", "Card", "Transfer", "Other"])
            m_notes = st.text_input("Notes (optional)", placeholder="Any extra detail…")

            if st.form_submit_button("Add Entry", type="primary", use_container_width=True):
                if m_desc.strip() and m_amt > 0:
                    row = pd.DataFrame([{
                        "date":        m_date.strftime("%d/%m/%Y"),
                        "description": m_desc.strip(),
                        "amount":      m_amt if m_type == "Expense" else -m_amt,
                        "type":        m_type.lower(),
                        "category":    m_cat,
                        "source":      "Manual",
                        "bank":        m_pay,
                        "notes":       m_notes,
                        "tx_id":       str(uuid.uuid4())[:8],
                    }])
                    st.session_state.transactions = (
                        row if st.session_state.transactions.empty
                        else pd.concat([st.session_state.transactions, row], ignore_index=True)
                    )
                    save_transactions(st.session_state.transactions)
                    st.success(f"Added: {m_desc} — ${m_amt:,.2f}")
                    st.rerun()
                else:
                    st.error("Fill in description and amount.")

    # ── Date Filter ───────────────────────────────────────────────────────────
    with st.expander("📅 Date Filter", expanded=True):
        today           = date.today()
        first_of_month  = today.replace(day=1)
        last_mo_end     = first_of_month - timedelta(days=1)
        last_mo_start   = last_mo_end.replace(day=1)

        p1, p2, p3 = st.columns(3)
        if p1.button("This Month", use_container_width=True):
            st.session_state.df_from = first_of_month
            st.session_state.df_to   = today
            st.rerun()
        if p2.button("Last Month", use_container_width=True):
            st.session_state.df_from = last_mo_start
            st.session_state.df_to   = last_mo_end
            st.rerun()
        if p3.button("3 Months", use_container_width=True):
            st.session_state.df_from = (today - timedelta(days=90)).replace(day=1)
            st.session_state.df_to   = today
            st.rerun()

        c1, c2 = st.columns(2)
        date_from = c1.date_input("From", value=st.session_state.get("df_from"),
                                  key="df_from", label_visibility="collapsed", format="DD/MM/YYYY")
        c1.caption("From")
        date_to   = c2.date_input("To",   value=st.session_state.get("df_to"),
                                  key="df_to",   label_visibility="collapsed", format="DD/MM/YYYY")
        c2.caption("To")

        if (date_from or date_to) and st.button("Clear Filter", use_container_width=True):
            st.session_state.df_from = None
            st.session_state.df_to   = None
            st.rerun()

    # ── Period Lock ───────────────────────────────────────────────────────────
    with st.expander("🔒 Period Lock"):
        locked_periods = load_locked_periods()
        if date_from and date_to:
            lbl = (date_from.strftime("%b %Y") if date_from.month == date_to.month
                   else f"{date_from.strftime('%d/%m/%y')} – {date_to.strftime('%d/%m/%y')}")
            if st.button(f"Lock {lbl}", use_container_width=True):
                locked_periods.append({"from": str(date_from), "to": str(date_to), "label": lbl})
                save_locked_periods(locked_periods)
                st.success(f"Locked: {lbl}")
                st.rerun()
        else:
            st.caption("Set a date range above to lock a period.")
        for i, lp in enumerate(locked_periods):
            ca, cb = st.columns([4, 1])
            ca.caption(f"🔒 {lp['label']}")
            if cb.button("✕", key=f"unlock_{i}"):
                locked_periods.pop(i)
                save_locked_periods(locked_periods)
                st.rerun()

    # ── Loaded Files ──────────────────────────────────────────────────────────
    if st.session_state.processed_files:
        with st.expander("📁 Loaded Files"):
            for pf in st.session_state.processed_files:
                st.caption(f"✓ {pf['name']} ({pf['bank']} · {pf['type']}) — {pf['count']} rows")

    # ── Clear Data ────────────────────────────────────────────────────────────
    if not st.session_state.transactions.empty:
        st.divider()
        if not st.session_state.confirm_clear:
            if st.button("🗑️ Clear All Data", type="secondary", use_container_width=True):
                st.session_state.confirm_clear = True
                st.rerun()
        else:
            st.warning("This will delete all data permanently. Are you sure?")
            ya, na = st.columns(2)
            if ya.button("Yes, delete", type="primary", use_container_width=True):
                st.session_state.transactions    = pd.DataFrame()
                st.session_state.processed_files = []
                st.session_state.confirm_clear   = False
                save_transactions(pd.DataFrame())
                st.rerun()
            if na.button("Cancel", use_container_width=True):
                st.session_state.confirm_clear = False
                st.rerun()

# ── Header ────────────────────────────────────────────────────────────────────
review_count = (
    int(st.session_state.transactions["category"].isin(OTHER_CATEGORIES).sum())
    if not st.session_state.transactions.empty else 0
)
badge = (f'<span class="review-pill">{review_count} need review</span>'
         if review_count > 0 else "")
st.markdown(f"""
<div class="header-box">
  <h1>💰 Business Expenses Tracker {badge}</h1>
  <p>Track, categorize and analyze your business transactions automatically</p>
</div>
""", unsafe_allow_html=True)

# ── Main ──────────────────────────────────────────────────────────────────────
if not st.session_state.transactions.empty:
    df = st.session_state.transactions.copy()
    df = _ensure_ids(df)

    for col, val in [("bank", ""), ("source", ""), ("notes", "")]:
        if col not in df.columns:
            df[col] = val

    # Date filter
    df["_date"] = parse_date_col(df["date"])
    if date_from:
        df = df[df["_date"] >= pd.Timestamp(date_from)]
    if date_to:
        df = df[df["_date"] <= pd.Timestamp(date_to)]

    # Period lock warning
    locked_periods = load_locked_periods()
    if locked_periods and not df.empty:
        dates = df["_date"].dropna()
        if any(pd.Timestamp(lp["from"]) <= d <= pd.Timestamp(lp["to"])
               for d in dates for lp in locked_periods):
            st.warning("🔒 You are viewing a locked period. Edit with caution.")

    df = df.drop(columns=["_date"], errors="ignore")

    # Metrics
    expenses_df  = df[df["type"] == "expense"]
    income_df    = df[df["type"] == "income"]
    real_exp_df  = expenses_df[~expenses_df["category"].isin(TRANSFER_CATEGORIES)]
    real_inc_df  = income_df[~income_df["category"].isin(TRANSFER_CATEGORIES)]
    real_expenses   = real_exp_df["amount"].sum()
    real_income     = abs(real_inc_df["amount"].sum())
    transfers_total = expenses_df[expenses_df["category"].isin(TRANSFER_CATEGORIES)]["amount"].sum()
    net             = real_income - real_expenses
    fixed_exp       = real_exp_df[real_exp_df["category"].isin(FIXED_CATEGORIES)]["amount"].sum()
    variable_exp    = real_expenses - fixed_exp

    # Top row: download + metrics
    hdr, dl_btn = st.columns([6, 1])
    with dl_btn:
        st.download_button(
            "📥 Export Excel",
            data=export_excel(df),
            file_name="expenses_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("💸 Expenses",    f"${real_expenses:,.0f}", help="Real expenses — transfers excluded")
    m2.metric("💰 Income",      f"${real_income:,.0f}",   help="Real income — transfers excluded")
    m3.metric("📊 Net Balance", f"${net:,.0f}",
              delta=f"{'▲' if net >= 0 else '▼'} ${abs(net):,.0f}",
              delta_color="normal" if net >= 0 else "inverse")
    m4.metric("🏗️ Fixed",       f"${fixed_exp:,.0f}",    help="Rent, Insurance, Bank Fees, Software, Professional Services")
    m5.metric("📦 Variable",    f"${variable_exp:,.0f}")
    m6.metric("🔄 Transfers",   f"${transfers_total:,.0f}", help="Inter-account moves excluded from totals")

    if real_expenses > 0:
        cap1, cap2, _ = st.columns([1, 1, 4])
        cap1.caption(f"Fixed: {fixed_exp / real_expenses * 100:.1f}%")
        cap2.caption(f"Variable: {variable_exp / real_expenses * 100:.1f}%")

    st.divider()

    # ── Search / Filter ───────────────────────────────────────────────────────
    sc1, sc2 = st.columns([2, 2])
    search_q     = sc1.text_input("🔍 Search", placeholder="Search description…", label_visibility="collapsed")
    sc1.caption("Search by description")
    all_cats     = sorted(df["category"].dropna().unique())
    filter_cats  = sc2.multiselect("Filter category", options=all_cats, placeholder="All categories")

    def apply_filters(d: pd.DataFrame) -> pd.DataFrame:
        if search_q:
            d = d[d["description"].str.contains(search_q, case=False, na=False)]
        if filter_cats:
            d = d[d["category"].isin(filter_cats)]
        return d

    df_view = apply_filters(df)

    # ── Tab data ──────────────────────────────────────────────────────────────
    bank_df   = df_view[df_view["source"] == "Bank Statement"].copy()
    cc_df     = df_view[df_view["source"] == "Credit Card"].copy()
    review_df = df_view[df_view["category"].isin(OTHER_CATEGORIES)].copy()

    tab_labels = [
        f"🏦 Bank ({len(bank_df)})",
        f"💳 Cards ({len(cc_df)})",
        f"📋 All ({len(df_view)})",
        f"⚠️ Needs Review ({len(review_df)})" if len(review_df) > 0 else "✅ All Reviewed",
        "📈 Trends",
    ]
    tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_labels)

    # ── Column config ─────────────────────────────────────────────────────────
    def _col_cfg(expense_first: bool = True):
        cats = ALL_CATEGORIES if expense_first else INCOME_CATEGORIES + EXPENSE_CATEGORIES
        return {
            "tx_id":       None,
            "type":        None,
            "date":        st.column_config.TextColumn("Date", width="small"),
            "description": st.column_config.TextColumn("Description"),
            "amount":      st.column_config.NumberColumn("Amount (MXN)", format="$%,.2f"),
            "category":    st.column_config.SelectboxColumn("Category", options=cats, required=True),
            "bank":        st.column_config.TextColumn("Bank / Payment", width="small"),
            "source":      st.column_config.TextColumn("Source", width="small"),
            "notes":       st.column_config.TextColumn("Notes ✏️", help="Add a memo for this transaction"),
        }

    COL_ORDER = ["date", "description", "amount", "category", "bank", "source", "notes"]

    # ── Section renderer ──────────────────────────────────────────────────────
    def render_section(sec_df: pd.DataFrame, key: str):
        if sec_df.empty:
            st.info("No transactions found.")
            return sec_df

        all_exp = sec_df[sec_df["type"] == "expense"].copy().reset_index(drop=True)
        all_inc = sec_df[sec_df["type"] == "income"].copy().reset_index(drop=True)
        t_out   = all_exp[all_exp["category"].isin(TRANSFER_CATEGORIES)]
        t_in    = all_inc[all_inc["category"].isin(TRANSFER_CATEGORIES)]
        exp_df  = all_exp[~all_exp["category"].isin(TRANSFER_CATEGORIES)].reset_index(drop=True)
        inc_df  = all_inc[~all_inc["category"].isin(TRANSFER_CATEGORIES)].reset_index(drop=True)

        micro    = exp_df[(exp_df["category"] == "Bank Fees") & (exp_df["amount"] < MICRO_FEE_MAX)].copy()
        exp_main = exp_df[~((exp_df["category"] == "Bank Fees") & (exp_df["amount"] < MICRO_FEE_MAX))].copy()

        t_exp = exp_df["amount"].sum()
        t_inc = abs(inc_df["amount"].sum())
        c1, c2 = st.columns(2)
        c1.metric("Expenses", f"${t_exp:,.2f}")
        c2.metric("Income",   f"${t_inc:,.2f}")

        results = []

        # Expenses
        if not exp_main.empty:
            st.subheader(f"💸 Expenses — ${t_exp:,.2f}")
            cat_bar = exp_df.groupby("category")["amount"].sum().sort_values(ascending=False).reset_index()
            fig = px.bar(cat_bar, x="category", y="amount", text_auto="$.3s",
                         color_discrete_sequence=["#e53e3e"])
            fig.update_layout(xaxis_title="", yaxis_title="MXN", height=240,
                              margin=dict(t=10, b=0), showlegend=False)
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

            edited_exp = st.data_editor(
                exp_main, column_config=_col_cfg(True),
                column_order=COL_ORDER,
                use_container_width=True, num_rows="dynamic", hide_index=True,
                key=f"exp_{key}",
            )
            edited_exp["type"] = edited_exp["category"].apply(
                lambda c: "income" if c in INCOME_CATEGORIES else "expense"
            )
            results.append(edited_exp)

        # Micro fees
        if not micro.empty:
            micro_total = micro["amount"].sum()
            with st.expander(f"🔸 Small Bank Fees ({len(micro)} entries · ${micro_total:,.2f}) — POS & processing fees under ${MICRO_FEE_MAX:.0f}"):
                st.dataframe(
                    micro[["date", "description", "amount", "bank"]],
                    use_container_width=True, hide_index=True,
                    column_config={"amount": st.column_config.NumberColumn("Amount", format="$%.2f")},
                )
            results.append(micro)

        # Income
        if not inc_df.empty:
            st.subheader(f"💰 Income — ${t_inc:,.2f}")
            inc_disp = inc_df.copy()
            inc_disp["amount"] = inc_disp["amount"].abs()
            cat_inc = inc_disp.groupby("category")["amount"].sum().sort_values(ascending=False).reset_index()
            fig2 = px.bar(cat_inc, x="category", y="amount", text_auto="$.3s",
                          color_discrete_sequence=["#38a169"])
            fig2.update_layout(xaxis_title="", yaxis_title="MXN", height=200,
                               margin=dict(t=10, b=0), showlegend=False)
            fig2.update_traces(textposition="outside")
            st.plotly_chart(fig2, use_container_width=True)

            edited_inc = st.data_editor(
                inc_disp, column_config=_col_cfg(False),
                column_order=COL_ORDER,
                use_container_width=True, num_rows="dynamic", hide_index=True,
                key=f"inc_{key}",
            )
            edited_inc["type"]   = edited_inc["category"].apply(
                lambda c: "income" if c in INCOME_CATEGORIES else "expense"
            )
            edited_inc["amount"] = edited_inc["amount"].abs() * -1
            results.append(edited_inc)

        # Transfers (collapsed, editable)
        tr_combined = pd.concat([t_out, t_in], ignore_index=True)
        if not tr_combined.empty:
            tr_out_tot = t_out["amount"].abs().sum()
            tr_in_tot  = t_in["amount"].abs().sum()
            with st.expander(f"🔄 Transfers — Out: ${tr_out_tot:,.2f} · In: ${tr_in_tot:,.2f} · excluded from totals"):
                st.caption("These are excluded from totals. If a transfer is actually real income or an expense, change its category below and it will move to the right section.")
                tr_disp = tr_combined.copy()
                tr_disp["amount"] = tr_disp["amount"].abs()
                tr_disp["direction"] = tr_combined["type"].map(
                    {"expense": "→ Out", "income": "← In"}
                )
                edited_tr = st.data_editor(
                    tr_disp[["tx_id", "direction", "date", "description", "amount", "category", "bank", "notes"]],
                    use_container_width=True, hide_index=True,
                    key=f"tr_{key}",
                    column_config={
                        "tx_id":       None,
                        "direction":   st.column_config.TextColumn("Dir.", width="small"),
                        "date":        st.column_config.TextColumn("Date", width="small"),
                        "description": st.column_config.TextColumn("Description"),
                        "amount":      st.column_config.NumberColumn("Amount", format="$%.2f"),
                        "category":    st.column_config.SelectboxColumn("Category", options=ALL_CATEGORIES, required=True),
                        "bank":        st.column_config.TextColumn("Bank", width="small"),
                        "notes":       st.column_config.TextColumn("Notes"),
                    },
                )
                # Restore signed amounts and update type from new category
                edited_tr["amount"] = edited_tr.apply(
                    lambda r: -abs(r["amount"]) if r["category"] in INCOME_CATEGORIES else abs(r["amount"]),
                    axis=1,
                )
                edited_tr["type"] = edited_tr["category"].apply(
                    lambda c: "income" if c in INCOME_CATEGORIES else "expense"
                )
                edited_tr = edited_tr.drop(columns=["direction"], errors="ignore")
            results.append(edited_tr)

        return pd.concat(results, ignore_index=True) if results else sec_df

    # ── Tabs ──────────────────────────────────────────────────────────────────
    result_bank = bank_df.copy()
    result_cc   = cc_df.copy()

    with tab1:
        if not bank_df.empty:
            result_bank = render_section(bank_df, "bank")
        else:
            st.info("No bank statement transactions found. Upload a bank statement PDF from the sidebar.")

    with tab2:
        if not cc_df.empty:
            result_cc = render_section(cc_df, "cc")
        else:
            st.info("No credit card transactions found. Upload a credit card PDF from the sidebar.")

    with tab3:
        st.caption("Read-only view of all transactions. Edit categories in the Bank or Cards tabs.")
        if not df_view.empty:
            # Show a clean read-only table with color indicator
            view = df_view.copy()
            view["💡"] = view["type"].map({"expense": "🔴", "income": "🟢"})
            st.dataframe(
                view[["💡", "date", "description", "amount", "category", "bank", "source", "notes"]],
                use_container_width=True, hide_index=True,
                column_config={
                    "amount": st.column_config.NumberColumn("Amount (MXN)", format="$%,.2f"),
                    "💡":     st.column_config.TextColumn("", width="small"),
                },
            )
        else:
            st.info("No transactions match your search/filter.")

    with tab4:
        if review_df.empty:
            st.success("✅ All transactions are properly categorized!")
        else:
            st.info(f"{len(review_df)} transactions need a category. Use the dropdowns below and click Save.")
            edited_review = st.data_editor(
                review_df,
                column_config=_col_cfg(True),
                column_order=COL_ORDER,
                use_container_width=True, num_rows="fixed", hide_index=True,
                key="review_editor",
            )
            if st.button("💾 Save Category Changes", type="primary"):
                full = st.session_state.transactions.copy()
                full = _apply_edits_back(full, edited_review)
                st.session_state.transactions = full
                save_transactions(full)
                st.success("Categories saved!")
                st.rerun()

    with tab5:
        st.subheader("Monthly Overview")
        df_t = df.copy()
        df_t["_date"] = parse_date_col(df_t["date"])
        df_t = df_t.dropna(subset=["_date"])
        df_t["Month"] = df_t["_date"].dt.to_period("M").astype(str)
        real_t = df_t[~df_t["category"].isin(TRANSFER_CATEGORIES)]

        me = real_t[real_t["type"] == "expense"].groupby("Month")["amount"].sum()
        mi = real_t[real_t["type"] == "income"].groupby("Month")["amount"].apply(lambda x: abs(x.sum()))
        monthly = pd.DataFrame({"Expenses": me, "Income": mi}).fillna(0)
        monthly["Net"] = monthly["Income"] - monthly["Expenses"]

        if not monthly.empty:
            fig_m = px.bar(
                monthly.reset_index(), x="Month", y=["Expenses", "Income"],
                barmode="group", text_auto="$.3s",
                color_discrete_map={"Expenses": "#e53e3e", "Income": "#38a169"},
            )
            fig_m.update_layout(height=320, margin=dict(t=10, b=0),
                                xaxis_title="", yaxis_title="MXN")
            fig_m.update_traces(textposition="outside")
            st.plotly_chart(fig_m, use_container_width=True)

            disp = monthly.copy()
            for c in disp.columns:
                disp[c] = disp[c].apply(lambda x: f"${x:,.2f}")
            st.dataframe(disp, use_container_width=True)

            st.subheader("Expenses by Category")
            cat_m = (real_t[real_t["type"] == "expense"]
                     .groupby(["Month", "category"])["amount"].sum().reset_index())
            if not cat_m.empty:
                fig_cat = px.bar(cat_m, x="Month", y="amount", color="category",
                                 barmode="stack", labels={"amount": "MXN", "Month": ""})
                fig_cat.update_layout(height=360, margin=dict(t=10, b=0))
                st.plotly_chart(fig_cat, use_container_width=True)
        else:
            st.info("Not enough data for monthly trends.")

    # ── Persist edits from bank/cc tabs ───────────────────────────────────────
    full = st.session_state.transactions.copy()
    full = _ensure_ids(full)
    full = _apply_edits_back(full, result_bank)
    full = _apply_edits_back(full, result_cc)
    st.session_state.transactions = full
    save_transactions(full)

# ── Welcome Screen ────────────────────────────────────────────────────────────
else:
    st.markdown("""
    <div style="text-align:center; padding: 2.5rem 1rem 1.5rem;">
        <div style="font-size:3.5rem; margin-bottom:.7rem">📊</div>
        <h2 style="color:#2d3748; margin-bottom:.4rem">Welcome to your Business Expenses Tracker</h2>
        <p style="color:#718096; font-size:1.05rem; max-width:520px; margin:0 auto 2rem;">
            Upload your bank or credit card PDFs to automatically extract, categorize,
            and analyze all your transactions — or add cash expenses manually.
        </p>
    </div>
    """, unsafe_allow_html=True)

    w1, w2, w3 = st.columns(3)
    for col, icon, title, desc in [
        (w1, "📤", "1. Upload PDF",   "Drop your bank or credit card statement PDF in the sidebar"),
        (w2, "🤖", "2. AI Reads It",  "Claude AI extracts and categorizes every transaction automatically"),
        (w3, "📥", "3. Export Excel", "Review, edit categories, and download a full Excel report"),
    ]:
        col.markdown(f"""
        <div class="welcome-card">
            <div style="font-size:2.2rem; margin-bottom:.5rem">{icon}</div>
            <h4 style="margin:.2rem 0">{title}</h4>
            <p style="color:#718096; font-size:.88rem; margin:0">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="tip-box">
        <strong>💡 Tip:</strong> You can also add <strong>cash expenses</strong> manually using
        the <strong>✏️ Add Manual Entry</strong> form in the sidebar — no PDF needed.
    </div>
    """, unsafe_allow_html=True)
