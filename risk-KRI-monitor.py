# dashboard.py
import streamlit as st
import pandas as pd
import json
import altair as alt
from datetime import datetime

# -------------------- Load Data --------------------
def load_data(file_path):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

data = load_data("/Users/manoj/coding/x_tmp/risk-KRI-monitor.json")

# -------------------- App Config --------------------
st.set_page_config(layout="wide", page_title="Bank Risk Monitoring Dashboard")

# min/max limits (no sliders)
LIMITS = {
    "cumulative_loss": {"min": 0, "max": 20_000_000, "field": "value", "title": "Cumulative Loss (USD)"},
    "high_risk_clients": {"min": 0, "max": 100, "field": "count", "title": "High-Risk Clients"},
    "cdd_overdue": {"min": 0, "max": 100, "field": "count", "title": "CDD/EDD Overdue"},
    "investigation_backlog": {"min": 0, "max": 15, "field": "Count", "title": "Investigation Backlog (>90d)"},
    "pep_hits": {"min": 0, "max": 30, "field": "count", "title": "PEP Hits"},
}

# -------------------- Styles --------------------
st.markdown("""
<style>
.stApp { background:#f7f8fb; font-family: "Segoe UI", system-ui, -apple-system, sans-serif; }
h1,h2,h3 { color:#0f172a; }
.card { background:white; padding:16px; border-radius:12px; box-shadow:0 1px 6px rgba(2,6,23,.06); }
hr { margin: 8px 0 4px; }
.small { font-size:0.85rem; color:#475569; }
.led { height:14px; width:14px; border-radius:50%; display:inline-block; }
.led.green { background:#22c55e; }
.led.red { background:#ef4444; }
.led.amber { background:#f59e0b; }
th, td { font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

# -------------------- Helpers --------------------
def fmt_money(v): return f"${v:,.0f}"
def rag_led(value, lo, hi):
    if value > hi: return '<span class="led red"></span>'
    if value > hi * 0.8: return '<span class="led amber"></span>'
    return '<span class="led green"></span>'

def breach_pct(value, lo, hi):
    rng = max(hi - lo, 1)
    if value < lo:
        return - (lo - value) / rng * 100
    if value > hi:
        return (value - hi) / rng * 100
    return 0.0

def ts_threshold_chart(df, x, y, lo, hi, title, color="#2563eb", area=False):
    """
    Time-series with threshold lines and % breach label placed at the latest point.
    """
    latest_x = df[x].iloc[-1]
    latest_y = df[y].iloc[-1]
    pct = breach_pct(latest_y, lo, hi)
    label = "OK" if pct == 0 else f"{pct:+.1f}%"

    base = alt.Chart(df).encode(
        x=alt.X(x, timeUnit="month", axis=alt.Axis(title="Month", format="%b")),
        y=alt.Y(y, title=title),
        tooltip=[alt.Tooltip(x, title="Date"), alt.Tooltip(y, title=title)]
    )
    series = (base.mark_area(opacity=0.25) if area else base.mark_line(strokeWidth=2)).encode(color=alt.value(color))
    max_rule = alt.Chart(pd.DataFrame({"y": [hi]})).mark_rule(color="red").encode(y="y")
    min_rule = alt.Chart(pd.DataFrame({"y": [lo]})).mark_rule(color="#22c55e", strokeDash=[6,4]).encode(y="y")
    text = alt.Chart(pd.DataFrame({"x": [latest_x], "y": [hi], "label": [label]})).mark_text(
        dx=6, dy=-6, color="red", fontWeight="bold"
    ).encode(x="x", y="y", text="label")

    return series + max_rule + min_rule + text

def ts_threshold_chart(df, x, y, lo, hi, title, color="#2563eb", area=False):
    """
    Time-series with threshold lines, breach label, and red border highlight if breached.
    Aggregates monthly to last value per month.
    """
    # ensure datetime
    df[x] = pd.to_datetime(df[x])

    # aggregate monthly (last value)
    df = df.groupby(df[x].dt.to_period("M")).agg({y: "last"}).reset_index()
    df[x] = df[x].dt.to_timestamp()

    # breach check
    latest_x = df[x].iloc[-1]
    latest_y = df[y].iloc[-1]
    pct = breach_pct(latest_y, lo, hi)
    label = "OK" if pct == 0 else f"{pct:+.1f}% breach"

    # base chart
    base = alt.Chart(df).encode(
        x=alt.X(x, timeUnit="yearmonth", axis=alt.Axis(title="Month", format="%b %Y")),
        y=alt.Y(y, title=title),
        tooltip=[alt.Tooltip(x, title="Date"), alt.Tooltip(y, title=title)]
    )

    series = (base.mark_area(opacity=0.25) if area else base.mark_line(point=True, strokeWidth=2)) \
                .encode(color=alt.value(color))

    # threshold lines
    max_rule = alt.Chart(pd.DataFrame({"y": [hi]})).mark_rule(color="red", strokeDash=[6,4]).encode(y="y")
    min_rule = alt.Chart(pd.DataFrame({"y": [lo]})).mark_rule(color="#22c55e", strokeDash=[6,4]).encode(y="y")

    # breach text at threshold line
    text = alt.Chart(pd.DataFrame({"x": [latest_x], "y": [hi], "label": [label]})).mark_text(
        dx=6, dy=-6, color="red", fontWeight="bold"
    ).encode(x="x", y="y", text="label")

    chart = series + max_rule + min_rule + text

    # highlight red border if breached
    if pct > 0:
        chart = chart.properties(
            width=400, height=300,
            title=alt.TitleParams(title, anchor="start", fontWeight="bold")
        ).configure_view(
            stroke="red", strokeWidth=2
        )
    else:
        chart = chart.properties(
            width=400, height=300,
            title=alt.TitleParams(title, anchor="start", fontWeight="bold")
        ).configure_view(stroke=None)

    return chart



def cat_bar_with_threshold(df, x, y, lo, hi, title):
    """
    Bar chart for categorical X (e.g., backlog by age) with threshold line and breach text.
    Text anchored at the category with max value.
    """
    # Pick a category to anchor the label
    anchor_cat = df.loc[df[y].idxmax(), x]
    # Compute total value (e.g., sum of >90d buckets already passed in)
    total_value = df[y].sum() if x == "_dummy" else df.loc[df[x].isin(df[x]), y].sum()
    pct = breach_pct(total_value, lo, hi)
    label = "OK" if pct == 0 else f"{pct:+.1f}%"

    bars = alt.Chart(df).mark_bar().encode(
        x=alt.X(f"{x}:N", title=""),
        y=alt.Y(y, title=title),
        tooltip=[x, y]
    )
    max_rule = alt.Chart(pd.DataFrame({"y": [hi]})).mark_rule(color="red").encode(y="y")
    min_rule = alt.Chart(pd.DataFrame({"y": [lo]})).mark_rule(color="#22c55e", strokeDash=[6,4]).encode(y="y")
    text = alt.Chart(pd.DataFrame({x: [anchor_cat], "y": [hi], "label": [label]})).mark_text(
        dx=6, dy=-6, color="red", fontWeight="bold"
    ).encode(x=x, y="y", text="label")

    return bars + max_rule + min_rule + text

# -------------------- Sidebar Navigation --------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "KRI Overview", "Threshold Limits"])

# ============================================================
#                          DASHBOARD
# ============================================================
if page == "Dashboard":
    st.title("🏦 Bank-wide Risk Monitoring Dashboard")

    # ------------------ Summary RAG Table ------------------
    st.subheader("🚦 KRI Breach Overview")
    backlog_90p = pd.DataFrame(data["investigation_backlog"].items(), columns=["Age","Count"]) \
                    .query("Age in ['90-180 Days','>180 Days']")["Count"].sum()

    summary = pd.DataFrame({
        "KRI": [
            "Cumulative Loss",
            "High-Risk Clients",
            "CDD/EDD Overdue",
            "Investigation Backlog (>90d)",
            "PEP Hits"
        ],
        "Current": [
            data["cumulative_loss_trend"][-1]["value"],
            data["high_risk_clients_trend"][-1]["count"],
            data["cdd_edd_overdue_trend"][-1]["count"],
            backlog_90p,
            data["pep_hits_monthly_trend"][-1]["count"]
        ],
        "Threshold (Max)": [
            LIMITS["cumulative_loss"]["max"],
            LIMITS["high_risk_clients"]["max"],
            LIMITS["cdd_overdue"]["max"],
            LIMITS["investigation_backlog"]["max"],
            LIMITS["pep_hits"]["max"]
        ]
    })
    summary["Status"] = [
        rag_led(c, 0, t) for c, t in zip(summary["Current"], summary["Threshold (Max)"])
    ]
    st.markdown(summary.to_html(escape=False, index=False), unsafe_allow_html=True)
    st.markdown("")

    # ------------------ Core metrics ------------------
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.metric("Cumulative Loss", fmt_money(data["cumulative_loss_trend"][-1]["value"]))
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.metric("Incidents (30d)", sum(data["incidents_by_type"].values()))
        st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.metric("High-Risk Clients", data["high_risk_clients_trend"][-1]["count"])
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ------------------ Trend Analysis ------------------
    st.subheader("📈 Trend Analysis of Key Risks")

    colA, colB = st.columns(2)
    with colA:
        loss_df = pd.DataFrame(data["cumulative_loss_trend"])
        loss_df["date"] = pd.to_datetime(loss_df["date"])
        chart = ts_threshold_chart(
            loss_df, "date", "value",
            LIMITS["cumulative_loss"]["min"], LIMITS["cumulative_loss"]["max"],
            "Cumulative Loss (USD)", color="#2563eb"
        )
        st.altair_chart(chart, use_container_width=True)

    with colB:
        hrc_df = pd.DataFrame(data["high_risk_clients_trend"])
        hrc_df["date"] = pd.to_datetime(hrc_df["date"])
        chart = ts_threshold_chart(
            hrc_df, "date", "count",
            LIMITS["high_risk_clients"]["min"], LIMITS["high_risk_clients"]["max"],
            "High-Risk Clients", color="#7c3aed"
        )
        st.altair_chart(chart, use_container_width=True)

    st.markdown("---")

    # ------------------ Compliance & Transactions ------------------
    st.subheader("🧭 Compliance & Transaction Status")
    colC, colD, colE = st.columns(3)

    with colC:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.caption("KYC Overdue vs Completed")
        kyc_df = pd.DataFrame({
            "Status": ["Overdue", "Completed"],
            "Count": [data["kyc_overdue"], data["kyc_total"] - data["kyc_overdue"]]
        })
        kyc_chart = alt.Chart(kyc_df).mark_bar().encode(
            x=alt.X("Status:N", title=""),
            y=alt.Y("Count:Q", title="Count"),
            color=alt.Color("Status:N", legend=None)
        )
        st.altair_chart(kyc_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with colD:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.caption("CDD/EDD Overdue Trend")
        cdd_df = pd.DataFrame(data["cdd_edd_overdue_trend"])
        cdd_df["date"] = pd.to_datetime(cdd_df["date"])
        chart = ts_threshold_chart(
            cdd_df, "date", "count",
            LIMITS["cdd_overdue"]["min"], LIMITS["cdd_overdue"]["max"],
            "CDD/EDD Overdue", color="#059669"
        )
        st.altair_chart(chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with colE:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.caption("Fraudulent Transactions (Monthly Sum)")
        fraud_daily = pd.DataFrame(data["fraud_transactions_daily"])
        fraud_daily["date"] = pd.to_datetime(fraud_daily["date"])
        fraud_month = fraud_daily.groupby(pd.Grouper(key="date", freq="MS"))["count"].sum().reset_index()
        fraud_chart = alt.Chart(fraud_month).mark_line().encode(
            x=alt.X("date:T", timeUnit="month", axis=alt.Axis(title="Month", format="%b")),
            y=alt.Y("count:Q", title="Fraud Tx Count"),
            tooltip=["date:T", "count:Q"]
        )
        st.altair_chart(fraud_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ------------------ Incident Breakdown ------------------
    st.subheader("🧨 Incident Breakdown")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    incidents_df = pd.DataFrame(data["incidents_by_type"].items(), columns=["Type","Count"])
    incidents_chart = alt.Chart(incidents_df).mark_bar().encode(
        x=alt.X("Type:N", title=""),
        y=alt.Y("Count:Q", title="Count"),
        color=alt.Color("Type:N", legend=None)
    )
    st.altair_chart(incidents_chart, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ------------------ Wealth Management & PB ------------------
    st.subheader("💼 Wealth Management & Private Banking Risk")
    w1, w2 = st.columns(2)
    with w1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        aum_df = pd.DataFrame(data["aum_trend"])
        aum_df["date"] = pd.to_datetime(aum_df["date"])
        aum_chart = ts_threshold_chart(  # no thresholds? still show as area for variety
            aum_df, "date", "value", 0, aum_df["value"].max()*1.1, "AUM (USD)", color="#0ea5e9"
        )
        st.altair_chart(aum_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with w2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        complaints_df = pd.DataFrame(data["client_complaints_trend"])
        complaints_df["date"] = pd.to_datetime(complaints_df["date"])
        complaints_chart = alt.Chart(complaints_df).mark_area(opacity=0.25).encode(
            x=alt.X("date:T", timeUnit="month", axis=alt.Axis(title="Month", format="%b")),
            y=alt.Y("count:Q", title="Client Complaints"),
            tooltip=["date:T", "count:Q"]
        ).encode(color=alt.value("#f59e0b"))
        st.altair_chart(complaints_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    w3, w4 = st.columns(2)
    with w3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        training_df = pd.DataFrame(data["compliance_training"].items(), columns=["Status","Count"])
        training_chart = alt.Chart(training_df).mark_bar().encode(
            x=alt.X("Status:N", title=""),
            y=alt.Y("Count:Q", title="Employees"),
            color=alt.Color("Status:N", legend=None)
        )
        st.altair_chart(training_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with w4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.metric("Client Churn Rate", f"{data['client_churn_rate']*100:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ------------------ Operational & Investigation Risk ------------------
    st.subheader("🛠️ Operational & Investigation Risk")
    o1, o2 = st.columns(2)
    with o1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        inv_df = pd.DataFrame(data["investigation_backlog"].items(), columns=["Age","Count"])
        backlog_chart = cat_bar_with_threshold(
            inv_df, "Age", "Count",
            LIMITS["investigation_backlog"]["min"], LIMITS["investigation_backlog"]["max"],
            "Investigation Backlog (# cases)"
        )
        st.altair_chart(backlog_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with o2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        res_df = pd.DataFrame(data["complaint_resolution_trend"])
        res_df["date"] = pd.to_datetime(res_df["date"])
        res_month = res_df.groupby(pd.Grouper(key="date", freq="MS"))["avg_days"].mean().reset_index()
        res_chart = alt.Chart(res_month).mark_line().encode(
            x=alt.X("date:T", timeUnit="month", axis=alt.Axis(title="Month", format="%b")),
            y=alt.Y("avg_days:Q", title="Avg. Resolution Days"),
            tooltip=["date:T", "avg_days:Q"]
        ).encode(color=alt.value("#10b981"))
        st.altair_chart(res_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ------------------ PEP Hits ------------------
    st.subheader("🧾 PEP Hit Monitoring")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    pep_df = pd.DataFrame(data["pep_hits_monthly_trend"])
    pep_df["date"] = pd.to_datetime(pep_df["date"])
    pep_chart = ts_threshold_chart(
        pep_df, "date", "count",
        LIMITS["pep_hits"]["min"], LIMITS["pep_hits"]["max"],
        "PEP Hits", color="#db2777"
    )
    st.altair_chart(pep_chart, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
#                        KRI OVERVIEW
# ============================================================
elif page == "KRI Overview":
    st.header("KRI Definitions and Values")

    st.subheader("KRI Definitions")
    kri_definitions = pd.DataFrame({
        "KRI Name": [
            "Cumulative Loss Value","Total Incidents","High-Risk Clients","KYC Overdue vs. Completed",
            "CDD/EDD Overdue","Daily Fraudulent Transactions","AUM","Client Complaints",
            "Compliance Training Completion","Client Churn Rate","Financial Crime Investigation Backlog",
            "Average Complaint Resolution Time","PEP Hit Monitoring"
        ],
        "Definition": [
            "Total losses from fraud and non-compliance over time.",
            "Total count of reported compliance and security incidents.",
            "Number of clients with a high-risk classification.",
            "Past-due vs. completed Know Your Customer reviews.",
            "Number of past-due Customer/Enhanced Due Diligence reviews.",
            "Daily count of transactions flagged as fraudulent.",
            "Total value of assets managed by the wealth division.",
            "Number of client complaints received over time.",
            "Completion count of mandatory compliance training.",
            "Percentage of clients who left the business.",
            "Cases under financial crime investigation by age.",
            "Average time (days) to resolve client complaints.",
            "Monthly count of politically exposed persons flagged."
        ]
    })
    st.table(kri_definitions)

    st.subheader("Current and Previous Month KRI Values")
    fraud_daily = pd.DataFrame(data["fraud_transactions_daily"])
    fraud_daily["date"] = pd.to_datetime(fraud_daily["date"])
    fraud_month = fraud_daily.groupby(pd.Grouper(key="date", freq="MS"))["count"].sum().reset_index()

    current_month_date = fraud_month["date"].iloc[-1]
    previous_month_date = fraud_month["date"].iloc[-2]

    kri_values = {
        "KRI": [
            "Cumulative Loss Value","Total Incidents","High-Risk Clients","KYC Overdue vs. Completed",
            "CDD/EDD Overdue","Fraud Tx (Monthly Sum)","AUM","Client Complaints",
            "Client Churn Rate","PEP Hits"
        ],
        f"Value ({current_month_date.strftime('%B')})": [
            fmt_money(data['cumulative_loss_trend'][-1]['value']),
            sum(data['incidents_by_type'].values()),
            data['high_risk_clients_trend'][-1]['count'],
            f"{data['kyc_overdue']} overdue / {data['kyc_total'] - data['kyc_overdue']} completed",
            data['cdd_edd_overdue_trend'][-1]['count'],
            int(fraud_month['count'].iloc[-1]),
            fmt_money(data['aum_trend'][-1]['value']),
            data['client_complaints_trend'][-1]['count'],
            f"{data['client_churn_rate']*100:.2f}%",
            data['pep_hits_monthly_trend'][-1]['count']
        ],
        f"Value ({previous_month_date.strftime('%B')})": [
            fmt_money(data['cumulative_loss_trend'][-2]['value']),
            "N/A",
            data['high_risk_clients_trend'][-2]['count'],
            "N/A",
            data['cdd_edd_overdue_trend'][-2]['count'],
            int(fraud_month['count'].iloc[-2]),
            fmt_money(data['aum_trend'][-2]['value']),
            data['client_complaints_trend'][-2]['count'],
            "N/A",
            data['pep_hits_monthly_trend'][-2]['count']
        ]
    }
    st.table(pd.DataFrame(kri_values))

# ============================================================
#                      THRESHOLD LIMITS
# ============================================================
elif page == "Threshold Limits":
    st.header("KRI Threshold Limits (Fixed)")

    limits_tbl = pd.DataFrame({
        "KRI": [
            "Cumulative Loss","High-Risk Clients","CDD/EDD Overdue",
            "Investigation Backlog (>90d)","PEP Hits"
        ],
        "Min Limit": [LIMITS["cumulative_loss"]["min"], 0, 0, 0, 0],
        "Max Limit": [LIMITS["cumulative_loss"]["max"], LIMITS["high_risk_clients"]["max"],
                      LIMITS["cdd_overdue"]["max"], LIMITS["investigation_backlog"]["max"],
                      LIMITS["pep_hits"]["max"]],
        "Current": [
            data["cumulative_loss_trend"][-1]["value"],
            data["high_risk_clients_trend"][-1]["count"],
            data["cdd_edd_overdue_trend"][-1]["count"],
            pd.DataFrame(data["investigation_backlog"].items(), columns=["Age","Count"]) \
              .query("Age in ['90-180 Days','>180 Days']")["Count"].sum(),
            data["pep_hits_monthly_trend"][-1]["count"]
        ]
    })
    limits_tbl["Breach %"] = [
        f"{breach_pct(c, lo, hi):+.1f}%" for c, lo, hi in zip(
            limits_tbl["Current"], limits_tbl["Min Limit"], limits_tbl["Max Limit"]
        )
    ]
    limits_tbl["Status"] = [
        rag_led(c, lo, hi) for c, lo, hi in zip(
            limits_tbl["Current"], limits_tbl["Min Limit"], limits_tbl["Max Limit"]
        )
    ]
    st.markdown(limits_tbl.to_html(escape=False, index=False), unsafe_allow_html=True)
