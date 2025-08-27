# dashboard.py

import streamlit as st
import pandas as pd
import json
import altair as alt
from datetime import datetime, timedelta

# Function to load data from the JSON file
def load_data(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please create it with the provided JSON content.")
        st.stop()
    except json.JSONDecodeError:
        st.error(f"Error: The file '{file_path}' is not a valid JSON file.")
        st.stop()

# Load data from the external JSON file.
# Note: The file path has been updated as requested.
data = load_data('/Users/manoj/coding/x_tmp/risk-KRI-monitor.json')

# --- Dashboard Configuration and Title ---
st.set_page_config(layout="wide", page_title="Bank Risk Monitoring Dashboard")

# Custom CSS for sans-serif font, dark black color, and bold chart axes.
st.markdown("""
    <style>
    body {
        font-family: sans-serif;
        color: #1a1a1a;
    }
    .stApp, .stAlert, .stMetric, .stMarkdown, .stSubheader, .stHeader {
        color: #1a1a1a;
    }
    .stChart > div > canvas {
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🏦 Bank-wide Risk Monitoring Dashboard")
st.markdown("### Key Risk Indicators (KRIs) for Compliance and Security")
st.markdown("---")

# --- KRI Metrics Section ---
st.header("1. Core Financial & Client Risk")

col1, col2, col3 = st.columns(3)

with col1:
    current_loss_value = data['cumulative_loss_trend'][-1]['value']
    previous_loss_value = data['cumulative_loss_trend'][-2]['value']
    st.metric(
        label="Cumulative Loss Value (USD)",
        value=f"${current_loss_value:,}",
        delta=current_loss_value - previous_loss_value,
        delta_color="inverse",
        help="Total losses from fraud and non-compliance."
    )

with col2:
    total_incidents = sum(data['incidents_by_type'].values())
    st.metric(
        label="Total Incidents (Last 30 Days)",
        value=total_incidents,
        delta=-5,  # Dummy delta for demonstration
        delta_color="normal",
        help="Count of reported compliance and security incidents."
    )

with col3:
    current_high_risk = data['high_risk_clients_trend'][-1]['count']
    previous_high_risk = data['high_risk_clients_trend'][-2]['count']
    st.metric(
        label="High-Risk Clients",
        value=current_high_risk,
        delta=current_high_risk - previous_high_risk,
        delta_color="inverse",
        help="Number of clients classified as high-risk."
    )

st.markdown("---")

# --- Trend Analysis Charts ---
st.header("2. Trend Analysis of Key Risks")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Cumulative Loss Value Trend")
    loss_df = pd.DataFrame(data['cumulative_loss_trend'])
    loss_df['date'] = pd.to_datetime(loss_df['date'])
    loss_chart = alt.Chart(loss_df).mark_line().encode(
        x=alt.X('date', timeUnit='month', axis=alt.Axis(title='Month', format='%b')),
        y=alt.Y('value', title='Cumulative Loss Value (USD)'),
        tooltip=[alt.Tooltip('date', title='Date'), 'value']
    ).properties(
        title='Cumulative Loss Value Trend'
    )
    st.altair_chart(loss_chart, use_container_width=True)
    st.info("The cumulative value of financial losses over time.")

with chart_col2:
    st.subheader("High-Risk Client Trend")
    high_risk_df = pd.DataFrame(data['high_risk_clients_trend'])
    high_risk_df['date'] = pd.to_datetime(high_risk_df['date'])
    high_risk_chart = alt.Chart(high_risk_df).mark_line().encode(
        x=alt.X('date', timeUnit='month', axis=alt.Axis(title='Month', format='%b')),
        y=alt.Y('count', title='High-Risk Client Count'),
        tooltip=[alt.Tooltip('date', title='Date'), 'count']
    ).properties(
        title='High-Risk Client Trend'
    )
    st.altair_chart(high_risk_chart, use_container_width=True)
    st.info("Tracks the number of high-risk clients to identify trends.")

st.markdown("---")

# --- Compliance & Transaction Charts ---
st.header("3. Compliance & Transaction Status")

chart_col3, chart_col4, chart_col5 = st.columns(3)

with chart_col3:
    st.subheader("KYC Overdue vs. Completed")
    kyc_df = pd.DataFrame({
        'Status': ['Overdue', 'Completed'],
        'Count': [data['kyc_overdue'], data['kyc_total'] - data['kyc_overdue']]
    })
    # Use Altair for narrower bars
    kyc_chart = alt.Chart(kyc_df).mark_bar(size=20).encode(
        x=alt.X('Status', axis=alt.Axis(labelFontWeight='bold')),
        y=alt.Y('Count', axis=alt.Axis(labelFontWeight='bold'))
    )
    st.altair_chart(kyc_chart, use_container_width=True)
    st.info("Tracks the number of past-due due diligence reviews.")
    
with chart_col4:
    st.subheader("CDD/EDD Overdue Trend")
    cdd_df = pd.DataFrame(data['cdd_edd_overdue_trend'])
    cdd_df['date'] = pd.to_datetime(cdd_df['date'])
    cdd_chart = alt.Chart(cdd_df).mark_line().encode(
        x=alt.X('date', timeUnit='month', axis=alt.Axis(title='Month', format='%b')),
        y=alt.Y('count', title='CDD/EDD Overdue Count'),
        tooltip=[alt.Tooltip('date', title='Date'), 'count']
    ).properties(
        title='CDD/EDD Overdue Trend'
    )
    st.altair_chart(cdd_chart, use_container_width=True)
    st.info("Monitors the number of past-due due diligence reviews.")

with chart_col5:
    st.subheader("Daily Fraudulent Transactions")
    # Aggregate daily data to a monthly sum
    fraud_df_daily = pd.DataFrame(data['fraud_transactions_daily'])
    fraud_df_daily['date'] = pd.to_datetime(fraud_df_daily['date'])
    fraud_df_monthly = fraud_df_daily.groupby(pd.Grouper(key='date', freq='MS')).sum().reset_index()
    
    fraud_chart = alt.Chart(fraud_df_monthly).mark_line().encode(
        x=alt.X('date', timeUnit='month', axis=alt.Axis(title='Month', format='%b')),
        y=alt.Y('count', title='Fraudulent Transaction Count'),
        tooltip=[alt.Tooltip('date', title='Date'), 'count']
    ).properties(
        title='Daily Fraudulent Transactions'
    )
    st.altair_chart(fraud_chart, use_container_width=True)
    st.info("Shows the daily count of detected fraudulent transactions.")

st.markdown("---")

# --- Incident Breakdown ---
st.header("4. Incident Breakdown")
st.subheader("Incident Count by Type")
incidents_df = pd.DataFrame(data['incidents_by_type'].items(), columns=['Type', 'Count'])
# Use Altair for narrower bars
incidents_chart = alt.Chart(incidents_df).mark_bar(size=20).encode(
    x=alt.X('Type', axis=alt.Axis(labelFontWeight='bold')),
    y=alt.Y('Count', axis=alt.Axis(labelFontWeight='bold'))
)
st.altair_chart(incidents_chart, use_container_width=True)

st.markdown("---")

# --- Wealth Management & Private Banking ---
st.header("5. Wealth Management & Private Banking Risk")

wm_col1, wm_col2 = st.columns(2)

with wm_col1:
    st.subheader("Assets Under Management (AUM)")
    aum_df = pd.DataFrame(data['aum_trend'])
    aum_df['date'] = pd.to_datetime(aum_df['date'])
    aum_chart = alt.Chart(aum_df).mark_line().encode(
        x=alt.X('date', timeUnit='month', axis=alt.Axis(title='Month', format='%b')),
        y=alt.Y('value', title='Assets (USD)'),
        tooltip=[alt.Tooltip('date', title='Date'), 'value']
    ).properties(
        title='Assets Under Management (AUM)'
    )
    st.altair_chart(aum_chart, use_container_width=True)
    st.info("Tracks the total value of assets managed over time.")

with wm_col2:
    st.subheader("Client Complaints Trend")
    complaints_df = pd.DataFrame(data['client_complaints_trend'])
    complaints_df['date'] = pd.to_datetime(complaints_df['date'])
    complaints_chart = alt.Chart(complaints_df).mark_line().encode(
        x=alt.X('date', timeUnit='month', axis=alt.Axis(title='Month', format='%b')),
        y=alt.Y('count', title='Complaint Count'),
        tooltip=[alt.Tooltip('date', title='Date'), 'count']
    ).properties(
        title='Client Complaints Trend'
    )
    st.altair_chart(complaints_chart, use_container_width=True)
    st.info("Monitors the number of client complaints to identify patterns.")

wm_col3, wm_col4 = st.columns(2)

with wm_col3:
    st.subheader("Compliance Training Completion")
    training_df = pd.DataFrame(data['compliance_training'].items(), columns=['Status', 'Count'])
    # Use Altair for narrower bars
    training_chart = alt.Chart(training_df).mark_bar(size=20).encode(
        x=alt.X('Status', axis=alt.Axis(labelFontWeight='bold')),
        y=alt.Y('Count', axis=alt.Axis(labelFontWeight='bold'))
    )
    st.altair_chart(training_chart, use_container_width=True)
    st.info("Measures the completion rate of mandatory compliance training.")

with wm_col4:
    st.subheader("Client Churn Rate")
    st.metric(
        label="Current Churn Rate",
        value=f"{data['client_churn_rate'] * 100:.2f}%",
        help="Percentage of clients leaving the business in the last 30 days."
    )
    st.info("A high churn rate can signal underlying service or risk issues.")

st.markdown("---")

# --- New Section: Operational & Investigation Risk ---
st.header("6. Operational & Investigation Risk")

op_col1, op_col2 = st.columns(2)

with op_col1:
    st.subheader("Financial Crime Investigation Backlog")
    investigation_df = pd.DataFrame(data['investigation_backlog'].items(), columns=['Age', 'Count'])
    # Use Altair for narrower bars
    investigation_chart = alt.Chart(investigation_df).mark_bar(size=20).encode(
        x=alt.X('Age', sort=None, axis=alt.Axis(labelFontWeight='bold')),
        y=alt.Y('Count', axis=alt.Axis(labelFontWeight='bold'))
    )
    st.altair_chart(investigation_chart, use_container_width=True)
    st.info("Backlog of financial crime cases by age of investigation.")

with op_col2:
    st.subheader("Average Complaint Resolution Time")
    # Aggregate daily data to a monthly average
    resolution_df_daily = pd.DataFrame(data['complaint_resolution_trend'])
    resolution_df_daily['date'] = pd.to_datetime(resolution_df_daily['date'])
    resolution_df_monthly = resolution_df_daily.groupby(pd.Grouper(key='date', freq='MS')).mean().reset_index()

    resolution_chart = alt.Chart(resolution_df_monthly).mark_line().encode(
        x=alt.X('date', timeUnit='month', axis=alt.Axis(title='Month', format='%b')),
        y=alt.Y('avg_days', title='Avg. Resolution Days'),
        tooltip=[alt.Tooltip('date', title='Date'), 'avg_days']
    ).properties(
        title='Average Complaint Resolution Time'
    )
    st.altair_chart(resolution_chart, use_container_width=True)
    st.info("Average number of days to resolve a client complaint.")

st.markdown("---")

# --- New PEP Hits section
st.header("7. PEP Hit Monitoring")
pep_hits_df = pd.DataFrame(data['pep_hits_monthly_trend'])
pep_hits_df['date'] = pd.to_datetime(pep_hits_df['date'])
pep_hits_chart = alt.Chart(pep_hits_df).mark_line().encode(
    x=alt.X('date', timeUnit='month', axis=alt.Axis(title='Month', format='%b')),
    y=alt.Y('count', title='PEP Hit Count'),
    tooltip=[alt.Tooltip('date', title='Date'), 'count']
).properties(
    title='PEP Hit Monitoring'
)
st.altair_chart(pep_hits_chart, use_container_width=True)
st.info("Shows the monthly count of politically exposed persons flagged.")
