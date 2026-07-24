import sqlite3
import pandas as pd
import streamlit as st
import altair as alt
import os

st.set_page_config(page_title="Oriel Router Dashboard", layout="wide")

st.title("Oriel: Evaluation-Driven Model Routing")
st.markdown("Live view of trial evidence, Wilson confidence bounds, and candidate metrics.")

db_path = os.getenv("ORIEL_DB_PATH", "oriel.db") # We will use a local db file for the dashboard

# Seed some dummy data if the db is completely empty just so the dashboard isn't blank for a fresh run
def seed_data(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM trials")
    if cursor.fetchone()[0] == 0:
        st.info("Seeding dummy evidence for demonstration purposes...")
        # (tenant_id, task, case_id, model, prompt_version, input_hash, passed, latency_ms, cost_microusd)
        trials = [
            ("tenant-1", "intent-routing", "c1", "gpt-4o", "v1", "h1", 1, 800, 5000),
            ("tenant-1", "intent-routing", "c2", "gpt-4o", "v1", "h2", 1, 750, 4800),
            ("tenant-1", "intent-routing", "c3", "gpt-4o", "v1", "h3", 1, 820, 5100),
            ("tenant-1", "intent-routing", "c1", "gpt-4o-mini", "v1", "h1", 1, 300, 200),
            ("tenant-1", "intent-routing", "c2", "gpt-4o-mini", "v1", "h2", 0, 290, 200),
            ("tenant-1", "intent-routing", "c3", "gpt-4o-mini", "v1", "h3", 1, 310, 200),
            ("tenant-1", "intent-routing", "c1", "gpt-3.5-turbo", "v1", "h1", 1, 400, 150),
            ("tenant-1", "intent-routing", "c2", "gpt-3.5-turbo", "v1", "h2", 1, 450, 150),
            ("tenant-1", "intent-routing", "c3", "gpt-3.5-turbo", "v1", "h3", 0, 420, 150),
        ]
        cursor.executemany("INSERT INTO trials VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", trials)
        conn.commit()

try:
    conn = sqlite3.connect(db_path)
    # Ensure table exists
    conn.execute(
        """CREATE TABLE IF NOT EXISTS trials (
            tenant_id TEXT NOT NULL, task TEXT NOT NULL, case_id TEXT NOT NULL, model TEXT NOT NULL,
            prompt_version TEXT NOT NULL, input_hash TEXT NOT NULL,
            passed INTEGER NOT NULL, latency_ms REAL NOT NULL, cost_microusd REAL NOT NULL,
            PRIMARY KEY (tenant_id, task, case_id, model, prompt_version, input_hash)
        )"""
    )
    seed_data(conn)
    
    df = pd.read_sql_query("""
        SELECT 
            model,
            SUM(passed) as successes,
            COUNT(*) as total,
            AVG(latency_ms) as mean_latency_ms,
            AVG(cost_microusd) as mean_cost_microusd
        FROM trials
        WHERE tenant_id='tenant-1' AND task='intent-routing' AND prompt_version='v1'
        GROUP BY model
    """, conn)
    
except Exception as e:
    st.error(f"Failed to load database: {e}")
    st.stop()

import math
def wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float:
    if total == 0:
        return 0.0
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = proportion + z * z / (2 * total)
    spread = z * math.sqrt((proportion * (1 - proportion) + z * z / (4 * total)) / total)
    return (centre - spread) / denominator

df["wilson_lower_bound"] = df.apply(lambda row: wilson_lower_bound(row["successes"], row["total"]), axis=1)
df["observed_accuracy"] = df["successes"] / df["total"]

st.subheader("Model Evidence and Reliability")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Quality Bounds (Wilson Interval)")
    # Bar chart for Wilson bounds
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('model:N', sort='-y', title='Model'),
        y=alt.Y('wilson_lower_bound:Q', title='Wilson Lower Bound (95% CI)'),
        color='model:N'
    ).properties(height=350)
    st.altair_chart(chart, use_container_width=True)

with col2:
    st.markdown("### Cost vs Latency Tradeoff")
    # Scatter plot
    scatter = alt.Chart(df).mark_circle(size=200).encode(
        x=alt.X('mean_latency_ms:Q', title='Mean Latency (ms)'),
        y=alt.Y('mean_cost_microusd:Q', title='Mean Cost (microUSD)'),
        color='model:N',
        tooltip=['model', 'mean_latency_ms', 'mean_cost_microusd', 'observed_accuracy', 'wilson_lower_bound']
    ).properties(height=350)
    st.altair_chart(scatter, use_container_width=True)

st.markdown("### Raw Aggregates")
st.dataframe(df.style.format({
    "mean_latency_ms": "{:.1f}",
    "mean_cost_microusd": "{:.1f}",
    "wilson_lower_bound": "{:.3f}",
    "observed_accuracy": "{:.3f}"
}), use_container_width=True)

