import streamlit as st
import pandas as pd
import json
import altair as alt

# -----------------------------
# Load transaction data
# -----------------------------
FILE_PATH = "/Users/manoj/coding/x_tmp/transaction_fraud.json"

with open(FILE_PATH, "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Sort transactions: Fraud first, then by amount
df_sorted = df.sort_values(by=["fraud_flag", "amount"], ascending=[False, False]).reset_index(drop=True)

# -----------------------------
# Streamlit UI Setup
# -----------------------------
st.set_page_config(page_title="Transaction Monitoring Dashboard", layout="wide")
st.title("💳 Real-time Transaction Monitoring Dashboard")

# -----------------------------
# KPI Cards
# -----------------------------
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Transactions", len(df))
with col2:
    st.metric("Fraud Transactions", df["fraud_flag"].sum())
with col3:
    st.metric("Fraud %", f"{(df['fraud_flag'].mean() * 100):.1f}%")

# -----------------------------
# Fraud LED Indicator
# -----------------------------
if df["fraud_flag"].sum() > 0:
    st.markdown(
        "<div style='background:red; width:30px; height:30px; border-radius:50%;'></div> 🚨 Fraud detected",
        unsafe_allow_html=True
    )
else:
    st.markdown(
        "<div style='background:green; width:30px; height:30px; border-radius:50%;'></div> ✅ All Clear",
        unsafe_allow_html=True
    )

# -----------------------------
# Transaction Table with Highlighting
# -----------------------------
st.subheader("📑 Transactions (Fraud on Top)")

def highlight_rows(row):
    if row["fraud_flag"] == 1:
        return ["background-color: #ffcccc"] * len(row)  # light red
    else:
        return ["background-color: #ccffcc"] * len(row)  # light green

st.dataframe(df_sorted.style.apply(highlight_rows, axis=1), use_container_width=True)

# -----------------------------
# Transaction Selection + Drilldown
# -----------------------------
st.subheader("🔍 Transaction Details")

txn_ids = df_sorted["txn_id"].tolist()
selected_txn = st.selectbox("Select Transaction ID:", txn_ids)

txn = df_sorted[df_sorted["txn_id"] == selected_txn].iloc[0]

st.markdown(
    f"""
    <div style="padding:15px; border-radius:10px; background-color:{'#ffe6e6' if txn['fraud_flag']==1 else '#e6ffe6'}; border:2px solid {'red' if txn['fraud_flag']==1 else 'green'};">
        <h3>{'🚨 FRAUD TRANSACTION' if txn['fraud_flag']==1 else '✅ Normal Transaction'}</h3>
        <b>Txn ID:</b> {txn['txn_id']} <br>
        <b>Account:</b> {txn['account']} <br>
        <b>Merchant:</b> {txn['merchant']} <br>
        <b>Amount:</b> ${txn['amount']} <br>
        <b>Timestamp:</b> {txn['timestamp']} <br>
        <b>Status:</b> {"FRAUD" if txn['fraud_flag']==1 else "OK"} <br>
        <b>Reason:</b> {txn['reason'] if txn['fraud_flag']==1 else "N/A"} <br>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Charts
# -----------------------------
st.subheader("📊 Analytics")

c1, c2 = st.columns(2)

# Fraud vs Normal Transaction Amounts
fraud_summary = df.groupby("fraud_flag")["amount"].sum().reset_index()
fraud_summary["fraud_flag"] = fraud_summary["fraud_flag"].map({0: "Normal", 1: "Fraud"})

bar_chart = alt.Chart(fraud_summary).mark_bar().encode(
    x=alt.X("fraud_flag:N", title="Transaction Type"),
    y=alt.Y("amount:Q", title="Total Amount"),
    color=alt.Color("fraud_flag:N", scale=alt.Scale(domain=["Normal", "Fraud"], range=["green", "red"]))
).properties(title="Fraud vs Normal Amounts")

with c1:
    st.altair_chart(bar_chart, use_container_width=True)

# Transactions Over Time
txn_trend = df.groupby(df["timestamp"].dt.strftime("%Y-%m-%d %H:%M"))["txn_id"].count().reset_index()
txn_trend.columns = ["timestamp", "txn_count"]

line_chart = alt.Chart(txn_trend).mark_line(point=True).encode(
    x="timestamp:T",
    y="txn_count:Q"
).properties(title="Transactions Over Time")

with c2:
    st.altair_chart(line_chart, use_container_width=True)

# -----------------------------
# Final Summary Table
# -----------------------------
st.subheader("📌 Raw Data (for export/debugging)")
st.dataframe(df, use_container_width=True)
