import streamlit as st
import pdfplumber
import pandas as pd
import anthropic
import json
import io
from dotenv import load_dotenv
import os
from datetime import date

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

EXPENSE_CATEGORIES = [
    "Rent", "Utilities", "Payroll", "Food & Beverages", "Supplies",
    "Equipment", "Marketing", "Transportation", "Bank Fees", "Taxes",
    "Insurance", "Repairs & Maintenance", "Software & Subscriptions",
    "Professional Services", "Transfer / Internal", "Other Expense"
]

INCOME_CATEGORIES = [
    "Sales Revenue", "Transfer In", "Loan / Financing",
    "Refund Received", "Investment", "Other Income"
]

TRANSFER_CATEGORIES = {"Transfer / Internal", "Transfer In"}
FIXED_CATEGORIES    = {"Rent", "Insurance", "Software & Subscriptions", "Bank Fees", "Professional Services"}
OTHER_CATEGORIES    = {"Other Expense", "Other Income"}
LOCKS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "locked_periods.json")

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
            for table in page.extract_tables():
                for row in table:
                    if row:
                        text += " | ".join(str(c) for c in row if c) + "\n"
    return text


def parse_transactions_with_ai(text):
    prompt = f"""You are analyzing a financial statement from Mexico.

First, identify:
- "bank_name": the name of the bank or card issuer (e.g. "BBVA", "Bajio", "Inbursa", "AMEX", "Santander", "Banamex", etc.) — read it from the document header or logo text
- "statement_type": either "Credit Card" or "Bank Statement"

Then extract ALL transactions.

IMPORTANT categorization rules:
- Explicit own-account movements (TRASPASO CUENTAS PROPIAS, TRASPASO ENTRE CUENTAS, ENVIO DINERO) → category "Transfer / Internal", type "expense"
- Any outgoing payment to a credit card or financial institution (PAGO TARJETA, PAGO TC, LIQUIDACION TARJETA, PAGO VISA, PAGO MASTERCARD, PAGO AMERICAN EXPRESS, PAGO AMEX, or any transaction whose description starts with PAGO + a bank/card name) → category "Transfer / Internal", type "expense"
- SPEI or transfer received from own business entities (ELDURADU SERVICIOS INTEGRALES) → category "Transfer In", type "income"
- Any money received INTO a BBVA account from another internal account (Bajio, Inbursa, or any SPEI/TRASPASO coming in) → category "Transfer In", type "income" — BBVA is the central payment account, incoming funds are internal movements not real income
- Any money SENT TO a BBVA account from another account (SPEI Enviado BBVA, TRASPASO A BBVA, or any transfer whose destination is BBVA) → category "Transfer / Internal", type "expense" — this is funding the BBVA payment account, not a real expense
- Other incoming transfers and SPEI received from unknown sources → category "Transfer In", type "income"
- Stripe payments (STRIPE, STRIPE PAYMENTS MEXICO, STRIPE*) → category "Sales Revenue", type "income"
- POS terminal settlements / card sales payouts (LIQUIDACION ADQUIRENTE, LIQUIDACIÓN ADQUIRENTE, DEPOSITO ADQUIRENTE) → category "Sales Revenue", type "income" — this is money the payment processor pays out to the merchant, NOT an expense
- Payment processor fees (TASA DE DESCUENTO, IVA TASA DE DESCUENTO, COMISION ADQUIRENTE) → category "Bank Fees", type "expense"
- Salary / payroll payments (NOMINA, PAGO NOMINA, DISPERSION NOMINA, DISPERSIÓN NÓMINA, PAGO CUENTA DE TERCERO) → category "Payroll", type "expense"
- SPEI Enviado to an individual person's full name (e.g. "SPEI Enviado Santander - Alfredo Cibrian Ramos") → category "Payroll", type "expense"
- Government payroll obligations (SUA, IMSS, INFONAVIT, FONACOT) → category "Payroll", type "expense"
- Debit card purchases at merchants → use the real merchant name in description, pick the most fitting expense category
- Do NOT label debit card merchant purchases as "Transfer / Internal" — only label actual money movements between accounts or card payments as such

Return a JSON object with:
- "bank_name": the bank or card issuer name
- "statement_type": either "Credit Card" or "Bank Statement"
- "transactions": array where each item has:
  - "date": transaction date as text (e.g. "15/05/2026")
  - "description": merchant or transaction description (be specific, use the actual name from the statement)
  - "amount": number — POSITIVE for money going OUT (expenses/charges/debits), NEGATIVE for money coming IN (deposits/credits/income)
  - "type": either "expense" or "income"
  - "category": for expenses use one of: {", ".join(EXPENSE_CATEGORIES)}
               for income use one of: {", ".join(INCOME_CATEGORIES)}

Return ONLY a valid JSON object, no other text.

Statement text:
{text}"""

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=16384,
        messages=[{"role": "user", "content": prompt}]
    )
    response_text = message.content[0].text.strip()
    if response_text.startswith("```"):
        response_text = response_text.split("```")[1]
        if response_text.startswith("json"):
            response_text = response_text[4:]
    result = json.loads(response_text)
    return result["transactions"], result["statement_type"], result.get("bank_name", "Unknown")


def parse_date_col(series):
    return pd.to_datetime(series, format="%d/%m/%Y", dayfirst=True, errors="coerce")


def make_selectable(df):
    d = df.copy()
    d.insert(0, "_select", False)
    return d


def export_excel(df):
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        df.to_excel(w, index=False)
    return out.getvalue()


def load_locked_periods():
    if os.path.exists(LOCKS_FILE):
        with open(LOCKS_FILE) as f:
            return json.load(f)
    return []


def save_locked_periods(periods):
    with open(LOCKS_FILE, "w") as f:
        json.dump(periods, f, indent=2)


def find_duplicates(df_new, df_existing):
    if df_existing.empty:
        return 0
    count = 0
    for _, row in df_new.iterrows():
        match = df_existing[
            (df_existing["date"] == row["date"]) &
            (abs(df_existing["amount"] - row["amount"]) < 0.01) &
            (df_existing["description"] == row["description"])
        ]
        if not match.empty:
            count += 1
    return count


# ── App ───────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Business Expenses Tracker", layout="wide")
st.title("Business Expenses Tracker")

for key, default in [
    ("transactions",    pd.DataFrame()),
    ("upload_key",      0),
    ("processed_files", []),
    ("upload_errors",   []),
    ("dup_warnings",    []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Upload Statement")

    if st.session_state.upload_errors:
        for err in st.session_state.upload_errors:
            st.error(err)
        st.session_state.upload_errors = []

    if st.session_state.dup_warnings:
        for w in st.session_state.dup_warnings:
            st.warning(w)
        st.session_state.dup_warnings = []

    uploaded_files = st.file_uploader(
        "Choose PDF files", type="pdf", accept_multiple_files=True,
        key=f"uploader_{st.session_state.upload_key}"
    )

    if uploaded_files and st.button("Process Statement", type="primary"):
        errors, dup_warnings = [], []
        for uploaded_file in uploaded_files:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    text = extract_text_from_pdf(uploaded_file)
                    transactions, statement_type, bank_name = parse_transactions_with_ai(text)
                    df_new = pd.DataFrame(transactions)
                    if "type" not in df_new.columns:
                        df_new["type"] = df_new["amount"].apply(lambda x: "income" if x < 0 else "expense")
                    df_new["source"] = statement_type
                    df_new["bank"]   = bank_name
                    if "notes" not in df_new.columns:
                        df_new["notes"] = ""

                    dupes = find_duplicates(df_new, st.session_state.transactions)
                    if dupes:
                        dup_warnings.append(f"⚠️ {uploaded_file.name}: {dupes} possible duplicate transaction(s) detected — added anyway, review manually.")

                    if st.session_state.transactions.empty:
                        st.session_state.transactions = df_new
                    else:
                        st.session_state.transactions = pd.concat(
                            [st.session_state.transactions, df_new], ignore_index=True
                        )
                    st.session_state.processed_files.append({
                        "name": uploaded_file.name, "type": statement_type,
                        "bank": bank_name, "count": len(transactions),
                    })
                except Exception as e:
                    errors.append(f"{uploaded_file.name}: {str(e)}")

        st.session_state.upload_errors  = errors
        st.session_state.dup_warnings   = dup_warnings
        st.session_state.upload_key    += 1
        st.rerun()

    # ── Date Filter ───────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Date Filter")
    c1, c2 = st.columns(2)
    date_from = c1.date_input("From", value=None, key="df_from", label_visibility="collapsed",
                               format="DD/MM/YYYY")
    c1.caption("From")
    date_to   = c2.date_input("To",   value=None, key="df_to",   label_visibility="collapsed",
                               format="DD/MM/YYYY")
    c2.caption("To")
    if date_from or date_to:
        if st.button("Clear Filter", use_container_width=True):
            st.session_state.df_from = None
            st.session_state.df_to   = None
            st.rerun()

    # ── Period Lock ───────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Period Lock")
    locked_periods = load_locked_periods()

    if date_from and date_to:
        label = f"{date_from.strftime('%b %Y')}" if date_from.month == date_to.month else f"{date_from.strftime('%d/%m/%y')} – {date_to.strftime('%d/%m/%y')}"
        if st.button(f"🔒 Lock  {label}", use_container_width=True):
            locked_periods.append({"from": str(date_from), "to": str(date_to), "label": label})
            save_locked_periods(locked_periods)
            st.success(f"Period locked: {label}")
            st.rerun()
    else:
        st.caption("Set a date range above to lock a period.")

    if locked_periods:
        for i, lp in enumerate(locked_periods):
            ca, cb = st.columns([4, 1])
            ca.caption(f"🔒 {lp['label']}")
            if cb.button("✕", key=f"unlock_{i}"):
                locked_periods.pop(i)
                save_locked_periods(locked_periods)
                st.rerun()

    # ── Loaded Files ──────────────────────────────────────────────────────────
    if st.session_state.processed_files:
        st.divider()
        st.caption("Loaded files:")
        for pf in st.session_state.processed_files:
            st.caption(f"✓ {pf['name']} ({pf['bank']} · {pf['type']}) — {pf['count']} transactions")

    if not st.session_state.transactions.empty:
        st.divider()
        if st.button("Clear All Data", type="secondary"):
            st.session_state.transactions = pd.DataFrame()
            st.session_state.processed_files = []
            st.rerun()

# ── Main ──────────────────────────────────────────────────────────────────────
if not st.session_state.transactions.empty:
    df = st.session_state.transactions.copy()

    if "bank"  not in df.columns: df["bank"]  = "Unknown"
    if "source" not in df.columns: df["source"] = "Bank Statement"
    if "notes" not in df.columns: df["notes"] = ""

    # Apply date filter
    df["_date"] = parse_date_col(df["date"])
    if date_from:
        df = df[df["_date"] >= pd.Timestamp(date_from)]
    if date_to:
        df = df[df["_date"] <= pd.Timestamp(date_to)]

    # Period lock banner
    locked_periods = load_locked_periods()
    if locked_periods and not df.empty:
        filtered_dates = df["_date"].dropna()
        viewing_locked = any(
            pd.Timestamp(lp["from"]) <= d <= pd.Timestamp(lp["to"])
            for d in filtered_dates for lp in locked_periods
        )
        if viewing_locked:
            st.warning("🔒 You are viewing a locked period. This data has been reviewed — edit with caution.")

    df = df.drop(columns=["_date"], errors="ignore")

    expenses_df = df[df["type"] == "expense"].copy()
    income_df   = df[df["type"] == "income"].copy()
    real_exp_df = expenses_df[~expenses_df["category"].isin(TRANSFER_CATEGORIES)]
    real_inc_df = income_df[~income_df["category"].isin(TRANSFER_CATEGORIES)]

    real_expenses   = real_exp_df["amount"].sum()
    real_income     = abs(real_inc_df["amount"].sum())
    transfers_total = expenses_df[expenses_df["category"].isin(TRANSFER_CATEGORIES)]["amount"].sum()
    net             = real_income - real_expenses
    fixed_exp       = real_exp_df[real_exp_df["category"].isin(FIXED_CATEGORIES)]["amount"].sum()
    variable_exp    = real_expenses - fixed_exp

    # Needs review alert
    review_count = len(df[df["category"].isin(OTHER_CATEGORIES)])
    if review_count > 0:
        st.warning(f"⚠️ {review_count} transaction(s) are in 'Other Expense' or 'Other Income' and need proper categorization.")

    # Metrics row 1
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Real Expenses", f"${real_expenses:,.2f}", help="Excludes transfers & card payments")
    col2.metric("Real Income",   f"${real_income:,.2f}",   help="Excludes incoming transfers")
    col3.metric("Net Balance",   f"${net:,.2f}", delta=f"{'+ ' if net >= 0 else ''}{net:,.2f}")
    col4.metric("Transfers (noise)", f"${transfers_total:,.2f}", help="Inter-account transfers and card payments")

    # Metrics row 2 — Fixed vs Variable
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Fixed Costs",    f"${fixed_exp:,.2f}",    help=f"Rent, Insurance, Bank Fees, Software, Professional Services")
    col6.metric("Variable Costs", f"${variable_exp:,.2f}", help="All other real expenses")
    col7.metric("Fixed %",    f"{(fixed_exp/real_expenses*100):.1f}%"    if real_expenses > 0 else "—")
    col8.metric("Variable %", f"{(variable_exp/real_expenses*100):.1f}%" if real_expenses > 0 else "—")

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["Bank Statements", "Credit Cards", "All Transactions", "Monthly Trends"])

    # ── Sort helper ───────────────────────────────────────────────────────────
    def apply_sort(df, sort_state_key):
        if sort_state_key not in st.session_state:
            st.session_state[sort_state_key] = ("original", False)
        cur_col, cur_asc = st.session_state[sort_state_key]
        s1, s2, s3, s4 = st.columns([1.2, 1.2, 1.2, 4])

        if s1.button("Date  ↺", key=f"{sort_state_key}_btn_date", use_container_width=True):
            st.session_state[sort_state_key] = ("original", False)
            cur_col, cur_asc = "original", False

        cat_label = ("Category ▲" if cur_asc else "Category ▼") if cur_col == "category" else "Category"
        if s2.button(cat_label, key=f"{sort_state_key}_btn_cat", use_container_width=True):
            new_asc = (not cur_asc) if cur_col == "category" else True
            st.session_state[sort_state_key] = ("category", new_asc)
            cur_col, cur_asc = "category", new_asc

        amt_label = ("Amount ▲" if cur_asc else "Amount ▼") if cur_col == "amount" else "Amount"
        if s3.button(amt_label, key=f"{sort_state_key}_btn_amt", use_container_width=True):
            new_asc = (not cur_asc) if cur_col == "amount" else False
            st.session_state[sort_state_key] = ("amount", new_asc)
            cur_col, cur_asc = "amount", new_asc

        if cur_col == "category":
            return df.sort_values("category", ascending=cur_asc).reset_index(drop=True)
        if cur_col == "amount":
            return df.sort_values("amount", ascending=cur_asc).reset_index(drop=True)
        parsed = parse_date_col(df["date"])
        return df.assign(_d=parsed).sort_values("_d").drop(columns="_d").reset_index(drop=True)

    # ── Section renderer ──────────────────────────────────────────────────────
    def render_section(filtered_df, key_suffix):
        all_exp = filtered_df[filtered_df["type"] == "expense"].copy().reset_index(drop=True)
        all_inc = filtered_df[filtered_df["type"] == "income"].copy().reset_index(drop=True)

        transfers_out = all_exp[all_exp["category"].isin(TRANSFER_CATEGORIES)].copy()
        transfers_in  = all_inc[all_inc["category"].isin(TRANSFER_CATEGORIES)].copy()
        exp_df = all_exp[~all_exp["category"].isin(TRANSFER_CATEGORIES)].copy().reset_index(drop=True)
        inc_df = all_inc[~all_inc["category"].isin(TRANSFER_CATEGORIES)].copy().reset_index(drop=True)

        t_exp = exp_df["amount"].sum()
        t_inc = abs(inc_df["amount"].sum())

        c1, c2 = st.columns(2)
        c1.metric("Expenses", f"${t_exp:,.2f}")
        c2.metric("Income",   f"${t_inc:,.2f}")

        col_cfg_base = {
            "_select":     st.column_config.CheckboxColumn("✓", default=False),
            "date":        st.column_config.TextColumn("Date"),
            "description": st.column_config.TextColumn("Description"),
            "amount":      st.column_config.NumberColumn("Amount", format="$%.2f"),
            "bank":        st.column_config.TextColumn("Bank"),
            "source":      st.column_config.TextColumn("Type"),
            "notes":       st.column_config.TextColumn("Notes", help="Add a memo for this transaction"),
            "type":        None,
        }

        if not exp_df.empty:
            st.subheader(f"Expenses — ${t_exp:,.2f}")
            st.bar_chart(exp_df.groupby("category")["amount"].sum().sort_values(ascending=False))
            exp_df = apply_sort(exp_df, f"sort_exp_{key_suffix}")
            edited_exp = st.data_editor(
                make_selectable(exp_df),
                column_config={**col_cfg_base,
                    "category": st.column_config.SelectboxColumn("Category", options=EXPENSE_CATEGORIES + INCOME_CATEGORIES, required=True),
                },
                use_container_width=True, num_rows="dynamic", hide_index=True,
                key=f"exp_{key_suffix}"
            )
            edited_exp["type"] = edited_exp["category"].apply(lambda c: "income" if c in INCOME_CATEGORIES else "expense")
            selected_exp = edited_exp[edited_exp["_select"] == True].drop(columns=["_select", "type"])
            if not selected_exp.empty:
                st.download_button(f"Export {len(selected_exp)} selected to Excel",
                    data=export_excel(selected_exp), file_name="selected_expenses.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"dl_exp_{key_suffix}")
            edited_exp = edited_exp.drop(columns=["_select"])
        else:
            st.info("No expenses found.")
            edited_exp = exp_df

        if not inc_df.empty:
            st.subheader(f"Income — ${t_inc:,.2f}")
            inc_display = inc_df.copy()
            inc_display["amount"] = inc_display["amount"].abs()
            st.bar_chart(inc_display.groupby("category")["amount"].sum().sort_values(ascending=False))
            inc_display = apply_sort(inc_display, f"sort_inc_{key_suffix}")
            edited_inc = st.data_editor(
                make_selectable(inc_display),
                column_config={**col_cfg_base,
                    "category": st.column_config.SelectboxColumn("Category", options=INCOME_CATEGORIES + EXPENSE_CATEGORIES, required=True),
                },
                use_container_width=True, num_rows="dynamic", hide_index=True,
                key=f"inc_{key_suffix}"
            )
            edited_inc["type"]   = edited_inc["category"].apply(lambda c: "income" if c in INCOME_CATEGORIES else "expense")
            edited_inc["amount"] = edited_inc["amount"].abs() * -1
            selected_inc = edited_inc[edited_inc["_select"] == True].drop(columns=["_select", "type"])
            selected_inc["amount"] = selected_inc["amount"].abs()
            if not selected_inc.empty:
                st.download_button(f"Export {len(selected_inc)} selected to Excel",
                    data=export_excel(selected_inc), file_name="selected_income.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"dl_inc_{key_suffix}")
            edited_inc = edited_inc.drop(columns=["_select"])
        else:
            st.info("No income found.")
            edited_inc = inc_df

        transfers_combined = pd.concat([transfers_out, transfers_in], ignore_index=True)
        if not transfers_combined.empty:
            transfers_combined["amount"]    = transfers_combined["amount"].abs()
            transfers_combined["direction"] = transfers_combined["type"].map(
                {"expense": "→ Out (you paid / sent)", "income": "← In (you received)"}
            )
            total_out = transfers_out["amount"].abs().sum()
            total_in  = transfers_in["amount"].abs().sum()
            with st.expander(f"Internal Transfers — {len(transfers_combined)} · Out: ${total_out:,.2f} · In: ${total_in:,.2f} · Not counted in totals"):
                st.caption("Movements between your own accounts or credit card bill payments — excluded to avoid double-counting.")
                tr_display = transfers_combined[["direction","date","description","amount","bank","source"]].copy()
                tr_display.insert(0, "_select", False)
                edited_tr = st.data_editor(tr_display,
                    column_config={
                        "_select":     st.column_config.CheckboxColumn("✓", default=False),
                        "direction":   st.column_config.TextColumn("Direction"),
                        "date":        st.column_config.TextColumn("Date"),
                        "description": st.column_config.TextColumn("Description"),
                        "amount":      st.column_config.NumberColumn("Amount", format="$%.2f"),
                        "bank":        st.column_config.TextColumn("Bank"),
                        "source":      st.column_config.TextColumn("Type"),
                    },
                    use_container_width=True, hide_index=True, key=f"tr_{key_suffix}"
                )
                selected_tr = edited_tr[edited_tr["_select"] == True].drop(columns=["_select"])
                if not selected_tr.empty:
                    st.download_button(f"Export {len(selected_tr)} selected to Excel",
                        data=export_excel(selected_tr), file_name="selected_transfers.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"dl_tr_{key_suffix}")

        return pd.concat([edited_exp, edited_inc, transfers_out, transfers_in], ignore_index=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    with tab1:
        bank_df = df[df["source"] == "Bank Statement"].copy()
        result_bank = render_section(bank_df, "bank") if not bank_df.empty else (st.info("No bank statement transactions yet."), bank_df)[1]

    with tab2:
        cc_df = df[df["source"] == "Credit Card"].copy()
        result_cc = render_section(cc_df, "cc") if not cc_df.empty else (st.info("No credit card transactions yet."), cc_df)[1]

    with tab3:
        render_section(df, "all")

    with tab4:
        st.subheader("Monthly Overview")
        df_t = df.copy()
        df_t["_date"] = parse_date_col(df_t["date"])
        df_t = df_t.dropna(subset=["_date"])
        df_t["_month"] = df_t["_date"].dt.to_period("M").astype(str)

        real_t = df_t[~df_t["category"].isin(TRANSFER_CATEGORIES)]
        monthly_exp = real_t[real_t["type"] == "expense"].groupby("_month")["amount"].sum()
        monthly_inc = real_t[real_t["type"] == "income"].groupby("_month")["amount"].apply(lambda x: abs(x.sum()))

        monthly = pd.DataFrame({"Expenses": monthly_exp, "Income": monthly_inc}).fillna(0)
        monthly["Net"] = monthly["Income"] - monthly["Expenses"]
        monthly.index.name = "Month"

        if not monthly.empty:
            st.bar_chart(monthly[["Expenses", "Income"]])

            display = monthly.copy()
            for col in display.columns:
                display[col] = display[col].apply(lambda x: f"${x:,.2f}")
            st.dataframe(display, use_container_width=True)

            st.subheader("Expenses by Category — Monthly")
            cat_monthly = (
                real_t[real_t["type"] == "expense"]
                .groupby(["_month", "category"])["amount"].sum()
                .unstack(fill_value=0)
            )
            if not cat_monthly.empty:
                st.bar_chart(cat_monthly)
        else:
            st.info("Not enough data for monthly trends.")

    st.session_state.transactions = pd.concat([result_bank, result_cc], ignore_index=True)

    # ── Excel Export ──────────────────────────────────────────────────────────
    st.divider()
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        real_exp_df.drop(columns=["type"], errors="ignore").to_excel(writer, sheet_name="Expenses", index=False)
        real_inc_df.drop(columns=["type"], errors="ignore").to_excel(writer, sheet_name="Income",   index=False)
        summary = pd.DataFrame({
            "": ["Real Expenses", "Real Income", "Net Balance", "Fixed Costs", "Variable Costs", "Transfers (excluded)"],
            "Amount": [
                f"${real_expenses:,.2f}", f"${real_income:,.2f}", f"${net:,.2f}",
                f"${fixed_exp:,.2f}", f"${variable_exp:,.2f}", f"${transfers_total:,.2f}"
            ]
        })
        summary.to_excel(writer, sheet_name="Summary", index=False)

    st.download_button(
        label="Download Full Excel Report",
        data=output.getvalue(),
        file_name="expenses_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info("Upload a bank or credit card statement PDF from the sidebar to get started.")
