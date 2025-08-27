#!/usr/bin/env python3
"""
generate_transaction_json.py

Fetch transactions from DB, compute extensive rule-based risk scoring,
assign fraud_flag and reason, and write JSON output.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

# ---------------------------
# Configuration
# ---------------------------
DB_URI = os.environ.get("TRANSACTION_DB_URI", "postgresql://manoj:password123@localhost:5432/transactions_db")
OUTPUT_JSON = os.environ.get("OUTPUT_JSON", "/Users/manoj/coding/x_tmp/transaction_fraud.json")
# optionally store a checkpoint to process incremental transactions
CHECKPOINT_FILE = os.environ.get("TXN_CHECKPOINT", "/tmp/txn_checkpoint.txt")

# Parameters / thresholds (tune for your shop)
HIST_LOOKBACK_DAYS = 90  # use past 90 days for customer statistics
AMOUNT_ZSCORE_THRESHOLD = 3.0
ABSOLUTE_SUSPICIOUS_AMOUNT = 10000.0  # any txn above this is high risk (bank-specific)
VELOCITY_WINDOW_MINUTES = 60
VELOCITY_COUNT_THRESHOLD = 5
RISK_SCORE_THRESHOLD = 0.6  # normalized 0..1 threshold to set fraud_flag
MIN_HISTORY_TRX = 5  # require at least this many historical txns to compute stable mean/std

# rule weights (sum of positive weights ideally <= 1)
WEIGHTS = {
    "amount_zscore": 0.30,
    "absolute_amount": 0.20,
    "velocity": 0.15,
    "blacklisted_merchant": 0.25,
    "foreign_location": 0.20,
    "high_risk_mcc": 0.18,
    "device_change": 0.12,
    "auth_failure": 0.15,
    "age_of_account": 0.08,
}

# Domain-specific lists (populate from internal sources)
BLACKLISTED_MERCHANTS = {"ShadyMart", "BadMerchantInc"}
HIGH_RISK_MCC = {6051, 5967}  # example MCCs (use actual MCC codes)
HIGH_RISK_COUNTRIES = {"NG", "RU", "KP"}

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------
# Utilities
# ---------------------------
def read_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                ts_str = f.read().strip()
                return datetime.fromisoformat(ts_str) if ts_str else None
        except Exception:
            logging.exception("Failed to read checkpoint")
    return None


def write_checkpoint(ts: datetime):
    try:
        with open(CHECKPOINT_FILE, "w") as f:
            f.write(ts.isoformat())
    except Exception:
        logging.exception("Failed to write checkpoint")


def to_json_serializable(obj):
    """Ensure values are JSON serializable (timestamps, numpy types)."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return obj


# ---------------------------
# DB Access
# ---------------------------
def get_db_engine():
    return create_engine(DB_URI, connect_args={})


def fetch_recent_transactions(engine, since: datetime = None, limit: int = 500) -> pd.DataFrame:
    """
    Pull candidate transactions for scoring.
    If since is provided, fetch txns after that timestamp (incremental).
    Otherwise fetch most recent 'limit' transactions.
    Expected table schema (adjust the query to your schema):
      transactions (
        txn_id VARCHAR PRIMARY KEY,
        account_id VARCHAR,
        txn_ts TIMESTAMP,
        amount NUMERIC,
        currency VARCHAR,
        merchant_name VARCHAR,
        mcc INTEGER,
        country CHAR(2),
        device_id VARCHAR,
        channel VARCHAR,
        auth_result VARCHAR,
        card_present BOOLEAN,
        ip_country CHAR(2)
      )
    """
    with engine.connect() as conn:
        if since:
            q = text(
                "SELECT * FROM transactions WHERE txn_ts > :since ORDER BY txn_ts ASC"
            )
            df = pd.read_sql(q, conn, params={"since": since})
        else:
            q = text(
                "SELECT * FROM transactions ORDER BY txn_ts DESC LIMIT :lim"
            )
            df = pd.read_sql(q, conn, params={"lim": limit})
    if df.empty:
        return df
    # normalize column names
    df = df.rename(columns=lambda c: c.lower())
    # ensure timestamp column name
    if "txn_ts" in df.columns:
        df["timestamp"] = pd.to_datetime(df["txn_ts"])
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        raise ValueError("Transaction table missing timestamp column (txn_ts or timestamp)")
    return df


def fetch_customer_aggregates(engine, account_ids: List[str], lookback_days: int = HIST_LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Fetch historical aggregate statistics per account over lookback.
    returns DataFrame indexed by account_id with columns:
     - txn_count
     - mean_amount
     - std_amount
     - countries (list)
     - devices (list)
    """
    cutoff = datetime.utcnow() - timedelta(days=lookback_days)
    with engine.connect() as conn:
        # aggregated numeric stats
        q = text("""
            SELECT
              account_id,
              COUNT(*) AS txn_count,
              AVG(amount) AS mean_amount,
              STDDEV_SAMP(amount) AS std_amount,
              ARRAY_AGG(DISTINCT country) AS countries,
              ARRAY_AGG(DISTINCT device_id) AS devices
            FROM transactions
            WHERE txn_ts >= :cutoff
              AND account_id = ANY(:accs)
            GROUP BY account_id
        """)
        # SQLAlchemy (Postgres) accepts arrays for params; adjust for other DBs
        df = pd.read_sql(q, conn, params={"cutoff": cutoff, "accs": account_ids})
    if df.empty:
        # return empty frame with expected columns
        return pd.DataFrame(columns=["account_id", "txn_count", "mean_amount", "std_amount", "countries", "devices"])
    df = df.rename(columns={"account_id": "account_id"})
    return df.set_index("account_id")


# ---------------------------
# Scoring and Rule Logic
# ---------------------------
def compute_features_and_score(txns: pd.DataFrame, customer_stats: pd.DataFrame) -> pd.DataFrame:
    """Add features, compute rule-level signals, score, and pick primary reason per txn."""

    # copy to avoid side-effects
    df = txns.copy()
    df["amount"] = df["amount"].astype(float)
    # basic features
    df["hour"] = df["timestamp"].dt.hour
    df["date"] = df["timestamp"].dt.date

    # join with historical customer stats (if missing, NaNs)
    df = df.merge(customer_stats[["txn_count", "mean_amount", "std_amount", "countries", "devices"]],
                  how="left", left_on="account_id", right_index=True)

    # safe fills
    df["txn_count"] = df["txn_count"].fillna(0)
    df["mean_amount"] = df["mean_amount"].fillna(0.0)
    df["std_amount"] = df["std_amount"].fillna(0.0)
    df["countries"] = df["countries"].apply(lambda x: x if isinstance(x, list) else [])
    df["devices"] = df["devices"].apply(lambda x: x if isinstance(x, list) else [])

    # Rule signals
    signals = {}

    # 1) amount z-score relative to customer's history
    def amount_zscore(row):
        if row["txn_count"] < MIN_HISTORY_TRX or row["std_amount"] == 0:
            return 0.0
        return (row["amount"] - row["mean_amount"]) / (row["std_amount"] + 1e-9)

    df["amount_zscore"] = df.apply(amount_zscore, axis=1)
    signals["amount_zscore"] = df["amount_zscore"].abs() >= AMOUNT_ZSCORE_THRESHOLD

    # 2) absolute suspicious amount
    signals["absolute_amount"] = df["amount"] >= ABSOLUTE_SUSPICIOUS_AMOUNT

    # 3) velocity: count of transactions for same account in last WINDOW minutes (computed via DB or in-memory)
    # For simplicity we compute in-memory by looking at all candidate txns and historical recent txns from DB:
    # Build a helper to count transactions within the time window (this may be replaced by an efficient DB query).
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["recent_count_window"] = 0
    for i, row in df.iterrows():
        window_start = row["timestamp"] - timedelta(minutes=VELOCITY_WINDOW_MINUTES)
        cnt = ((df["account_id"] == row["account_id"]) &
               (df["timestamp"] >= window_start) &
               (df["timestamp"] <= row["timestamp"])).sum()
        df.at[i, "recent_count_window"] = int(cnt)
    signals["velocity"] = df["recent_count_window"] >= VELOCITY_COUNT_THRESHOLD

    # 4) blacklisted merchant
    signals["blacklisted_merchant"] = df["merchant_name"].isin(BLACKLISTED_MERCHANTS)

    # 5) foreign/unusual location (compare txn country/ip_country against customer's past countries)
    def unusual_location(row):
        # flag if txn country not in customer's past countries (excl. empty history)
        if not row["countries"]:
            return False
        txn_country = (row.get("country") or row.get("ip_country") or "").upper()
        return txn_country and txn_country not in [c.upper() for c in row["countries"]]

    signals["foreign_location"] = df.apply(unusual_location, axis=1)

    # 6) high-risk MCC / merchant category
    signals["high_risk_mcc"] = df["mcc"].isin(HIGH_RISK_MCC)

    # 7) device change: if new device not seen historically
    def device_change(row):
        dev = row.get("device_id")
        return bool(dev and dev not in row["devices"])

    signals["device_change"] = df.apply(device_change, axis=1)

    # 8) repeated auth failures on this account in short time (requires query). Here we inspect a column 'auth_result'
    signals["auth_failure"] = df["auth_result"].astype(str).str.lower().isin({"fail", "refused", "declined"})

    # 9) age of account (if available) - younger accounts are riskier
    # Expect optional column account_open_date in txns; fallback to NaN
    if "account_open_date" in df.columns:
        df["account_age_days"] = (pd.to_datetime(df["timestamp"]) - pd.to_datetime(df["account_open_date"])).dt.days
    else:
        df["account_age_days"] = np.nan
    signals["age_of_account"] = df["account_age_days"].fillna(3650) < 30  # account less than 30 days considered risk

    # 10) improbable amounts - e.g., whole round numbers repeated or max for user (domain-specific)
    signals["suspicious_round_amount"] = df["amount"] % 100 == 0

    # Assemble per-txn weighted score
    # Normalize amount_zscore to 0..1 scale for contribution: map |z| >= AMOUNT_ZSCORE_THRESHOLD -> 1, lower -> |z|/threshold
    df["amount_zscore_score"] = df["amount_zscore"].abs().clip(0, AMOUNT_ZSCORE_THRESHOLD) / (AMOUNT_ZSCORE_THRESHOLD)
    # Now compute score
    score_components = []
    for rname, weight in WEIGHTS.items():
        if rname == "amount_zscore":
            comp = df["amount_zscore_score"] * weight
        elif rname == "absolute_amount":
            comp = signals["absolute_amount"].astype(float) * weight
        elif rname == "velocity":
            comp = signals["velocity"].astype(float) * weight
        elif rname == "blacklisted_merchant":
            comp = signals["blacklisted_merchant"].astype(float) * weight
        elif rname == "foreign_location":
            comp = signals["foreign_location"].astype(float) * weight
        elif rname == "high_risk_mcc":
            comp = signals["high_risk_mcc"].astype(float) * weight
        elif rname == "device_change":
            comp = signals["device_change"].astype(float) * weight
        elif rname == "auth_failure":
            comp = signals["auth_failure"].astype(float) * weight
        elif rname == "age_of_account":
            comp = signals["age_of_account"].astype(float) * weight
        else:
            # unknown rule: score 0
            comp = 0
        score_components.append(comp)

    # sum components to get raw score (0..sum(weights)); normalize by sum(weights)
    total_weight = sum(WEIGHTS.values())
    df["raw_score"] = sum(score_components)
    df["risk_score"] = df["raw_score"] / max(total_weight, 1e-9)

    # Compose a prioritized reason per txn: pick highest contributing rule or descriptive mapping
    def pick_reason(row):
        reasons = []
        # order by severity / weight
        if row["amount_zscore_score"] > 0 and row["amount_zscore_score"] * WEIGHTS["amount_zscore"] > 0.10:
            reasons.append(("High amount deviation", row["amount_zscore_score"] * WEIGHTS["amount_zscore"]))
        if row["amount"] >= ABSOLUTE_SUSPICIOUS_AMOUNT:
            reasons.append(("Absolute large amount", WEIGHTS["absolute_amount"]))
        if row["recent_count_window"] >= VELOCITY_COUNT_THRESHOLD:
            reasons.append(("Multiple rapid transactions", WEIGHTS["velocity"]))
        if row["merchant_name"] in BLACKLISTED_MERCHANTS:
            reasons.append(("Blacklisted merchant", WEIGHTS["blacklisted_merchant"]))
        if row["mcc"] in HIGH_RISK_MCC:
            reasons.append(("High-risk merchant category", WEIGHTS["high_risk_mcc"]))
        if row["auth_result"] and str(row["auth_result"]).lower() in {"fail", "declined", "refused"}:
            reasons.append(("Auth failures", WEIGHTS["auth_failure"]))
        if row["account_age_days"] < 30:
            reasons.append(("New account", WEIGHTS["age_of_account"]))
        if row["device_id"] and row["device_id"] not in (row.get("devices") or []):
            reasons.append(("New device used", WEIGHTS["device_change"]))
        if row.get("country") and row.get("countries") and row["country"].upper() not in [c.upper() for c in row["countries"]]:
            reasons.append(("Unusual location", WEIGHTS["foreign_location"]))
        if row["suspicious_round_amount"]:
            reasons.append(("Suspicious round amount", 0.02))

        if not reasons:
            return "No strong rule triggered"
        # return reason with highest score
        reasons_sorted = sorted(reasons, key=lambda x: x[1], reverse=True)
        return reasons_sorted[0][0]

    df["reason"] = df.apply(pick_reason, axis=1)

    # Determine fraud flag using risk_score >= threshold OR presence of specific high-confidence rules
    def decide_flag(row):
        # high-confidence rules that immediately mark fraud
        if row["merchant_name"] in BLACKLISTED_MERCHANTS:
            return 1
        if row["amount"] >= ABSOLUTE_SUSPICIOUS_AMOUNT and row["amount_zscore"] > 1.5:
            return 1
        if row["recent_count_window"] >= VELOCITY_COUNT_THRESHOLD and row["amount"] > 100:
            return 1
        # else use risk_score threshold
        return 1 if row["risk_score"] >= RISK_SCORE_THRESHOLD else 0

    df["fraud_flag"] = df.apply(decide_flag, axis=1).astype(int)

    # Keep main output columns and ensure serializable types
    out_cols = [
        "txn_id", "timestamp", "account_id", "merchant_name", "amount",
        "currency", "mcc", "country", "device_id", "channel",
        "auth_result", "fraud_flag", "reason", "risk_score",
        "recent_count_window", "amount_zscore"
    ]
    for c in out_cols:
        if c not in df.columns:
            df[c] = None

    return df[out_cols].copy()


# ---------------------------
# Main flow
# ---------------------------
def main():
    logging.info("Starting transaction JSON generation")

    engine = get_db_engine()

    # Try incremental since last checkpoint if available; otherwise pull recent N txns
    last_checkpoint = read_checkpoint()
    if last_checkpoint:
        logging.info("Using checkpoint; fetching transactions since %s", last_checkpoint.isoformat())
        txns = fetch_recent_transactions(engine, since=last_checkpoint)
    else:
        logging.info("No checkpoint; fetching latest transactions")
        txns = fetch_recent_transactions(engine, since=None, limit=500)

    if txns.empty:
        logging.info("No transactions to process. Exiting.")
        return

    # Collect account ids to fetch historical stats
    account_ids = txns["account_id"].dropna().unique().tolist()
    customer_stats = fetch_customer_aggregates(engine, account_ids)

    # Compute scoring and reasons
    scored = compute_features_and_score(txns, customer_stats)

    # Convert to JSON-serializable dicts
    records = scored.to_dict(orient="records")
    records_serializable = []
    for r in records:
        # cast/serialize types
        r2 = {k: to_json_serializable(v) for k, v in r.items()}
        # add extra optional fields to ease downstream UI
        r2.setdefault("txn_id", r2.get("txn_id") or f"TXN-{int(np.random.randint(100000,999999))}")
        records_serializable.append(r2)

    # Write JSON output (overwrite)
    logging.info("Writing %d transactions to JSON %s", len(records_serializable), OUTPUT_JSON)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(records_serializable, f, indent=2, default=to_json_serializable)

    # update checkpoint to last txn timestamp processed
    last_ts = pd.to_datetime(txns["timestamp"]).max()
    write_checkpoint(last_ts)
    logging.info("Done. checkpoint updated to %s", last_ts.isoformat())


if __name__ == "__main__":
    main()
