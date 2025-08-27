import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# ----------------- Page Config -----------------
st.set_page_config(page_title="Telecom Billing Analyzer", layout="wide")
st.title("📊 Telecom Billing Analyzer")
st.markdown("Detect anomalies in telecom billing data to catch fraud or errors.")

# ----------------- Sidebar -----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📈 Dashboard", "🚨 Anomaly Detection"])

# ----------------- File Upload -----------------
uploaded_file = st.file_uploader("Upload Billing Dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ----------------- PAGE 1: Dashboard -----------------
    if page == "📈 Dashboard":
        st.subheader("📊 Dashboard")

        # Quick Stats
        st.subheader("📌 Quick Stats")
        col1, col2, col3, col4 = st.columns(4)

        total_users = df['user_id'].nunique()
        total_charges = df['charges'].sum()
        avg_usage = df['data_usage'].mean()
        total_plans = df['plan'].nunique()

        col1.metric("👥 Total Users", total_users)
        col2.metric("💰 Total Charges", f"${total_charges:,.2f}")
        col3.metric("📶 Avg Data Usage", f"{avg_usage:.2f} GB")
        col4.metric("📑 Plans Available", total_plans)

        # Dataset Preview
        st.subheader("Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        # Summary Statistics
        st.subheader("Summary Statistics")
        st.write(df.describe())

    # ----------------- PAGE 2: Anomaly Detection -----------------
    elif page == "🚨 Anomaly Detection":
        st.subheader("🚨 Anomaly Detection using Isolation Forest")

        # Features for anomaly detection
        features = ['charges', 'data_usage']
        X = df[features]

        # Train model
        model = IsolationForest(contamination=0.05, random_state=42)
        df['anomaly'] = model.fit_predict(X)
        df['anomaly'] = df['anomaly'].map({1: 'Normal', -1: 'Anomaly'})

        # Reasoning for anomalies
        df['reason'] = "Normal"
        mean_charges, std_charges = df['charges'].mean(), df['charges'].std()
        mean_usage, std_usage = df['data_usage'].mean(), df['data_usage'].std()

        for i in df.index:
            if df.loc[i, 'anomaly'] == "Anomaly":
                reasons = []
                if df.loc[i, 'charges'] > mean_charges + 3*std_charges:
                    reasons.append("Unusually high charges")
                if df.loc[i, 'data_usage'] > mean_usage + 3*std_usage:
                    reasons.append("Excessive data usage")
                if df.loc[i, 'charges'] < mean_charges - 3*std_charges:
                    reasons.append("Abnormally low charges")
                df.loc[i, 'reason'] = ", ".join(reasons) if reasons else "Outlier detected"

        # KPI Summary Row
        st.subheader("📌 Quick Stats")
        col1, col2, col3, col4 = st.columns(4)

        total_users = df['user_id'].nunique()
        total_charges = df['charges'].sum()
        avg_usage = df['data_usage'].mean()
        anomaly_count = (df['anomaly'] == "Anomaly").sum()

        col1.metric("👥 Total Users", total_users)
        col2.metric("💰 Total Charges", f"${total_charges:,.2f}")
        col3.metric("📶 Avg Data Usage", f"{avg_usage:.2f} GB")
        col4.metric("🚨 Anomalies Detected", anomaly_count)

        # Highlight Anomalies Table
        def highlight_anomalies(row):
            if row['anomaly'] == "Anomaly":
                return ['background-color: #ffcccc'] * len(row)  # light red
            else:
                return [''] * len(row)

        styled_df = df[['user_id', 'plan', 'charges', 'data_usage', 'anomaly', 'reason']].style.apply(highlight_anomalies, axis=1)

        st.subheader("📋 All Users with Anomaly Highlight")
        st.dataframe(styled_df, use_container_width=True)

        # Scatter Plot
        st.subheader("Charges vs Usage (with anomalies)")
        fig, ax = plt.subplots()
        normal = df[df['anomaly'] == "Normal"]
        anomaly = df[df['anomaly'] == "Anomaly"]
        ax.scatter(normal['data_usage'], normal['charges'], label="Normal", c="blue")
        ax.scatter(anomaly['data_usage'], anomaly['charges'], label="Anomaly", c="red")
        ax.set_xlabel("Data Usage")
        ax.set_ylabel("Charges")
        ax.legend()
        st.pyplot(fig)

                # Pie Chart for Normal vs Anomalies
        st.subheader("📊 User Distribution: Normal vs Anomalies")
        anomaly_counts = df['anomaly'].value_counts()

        fig2, ax2 = plt.subplots()
        ax2.pie(
            anomaly_counts,
            labels=anomaly_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=["#66b3ff", "#ff6666"]
        )
        ax2.axis('equal')  # Equal aspect ratio ensures pie is a circle.
        st.pyplot(fig2)


        # Download anomaly report
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Anomaly Report",
            csv,
            "anomaly_report.csv",
            "text/csv"
        )
