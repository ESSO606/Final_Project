import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from tensorflow.keras.models import load_model
st.set_page_config(page_title="Customer Analytics Dashboard", layout="wide")
DATA_DIR = Path("C:/Users/Muhammad/Desktop/Route/project")

CUSTOMER_AGG_PATH = DATA_DIR / "customer_agg.parquet"
TRANSACTIONS_PATH = DATA_DIR / "transactions.parquet"

SCALER_PATH     = DATA_DIR / "scaler.pkl"
ENCODER_PATH    = DATA_DIR / "encoder_ae.h5"
DNN_PATH        = DATA_DIR / "dnn_model.h5"
LE_STOCK_PATH   = DATA_DIR / "le_stock.pkl"
KMEANS_PATH     = DATA_DIR / "kmeans.pkl"
LEGACY_MODEL    = DATA_DIR / "ltv_model.joblib"
@st.cache_data
def load_customer_agg():
    if CUSTOMER_AGG_PATH.exists():
        return pd.read_parquet(CUSTOMER_AGG_PATH)
    return pd.DataFrame()

@st.cache_data
def load_transactions():
    if TRANSACTIONS_PATH.exists():
        return pd.read_parquet(TRANSACTIONS_PATH)
    return pd.DataFrame()

@st.cache_resource
def load_artifacts():
    artifacts = {}
    if SCALER_PATH.exists():   artifacts["scaler"]  = joblib.load(SCALER_PATH)
    if LE_STOCK_PATH.exists(): artifacts["le_stock"] = joblib.load(LE_STOCK_PATH)
    if KMEANS_PATH.exists():   artifacts["kmeans"]   = joblib.load(KMEANS_PATH)
    if LEGACY_MODEL.exists():  artifacts["rf_model"] = joblib.load(LEGACY_MODEL)
    if ENCODER_PATH.exists():  artifacts["encoder"]  = load_model(ENCODER_PATH)
    if DNN_PATH.exists():      artifacts["dnn"]      = load_model(DNN_PATH)
    return artifacts

art = load_artifacts()
customer_agg = load_customer_agg()
transactions  = load_transactions()
def prepare_dl_features(row, art):
    numeric = row[["recency_days", "frequency", "monetary_sum"]].values.reshape(1, -1)

    scaler = art.get("scaler")
    if scaler is not None:
        numeric_scaled = scaler.transform(numeric)
    else:
        numeric_scaled = numeric

    encoder = art.get("encoder")
    if encoder is not None:
        encoded = encoder.predict(numeric_scaled)
    else:
        encoded = np.zeros((1, 8))

    kmeans = art.get("kmeans")
    if kmeans is not None:
        cluster = kmeans.predict(numeric_scaled)
        cluster_feat = np.array(cluster).reshape(1, 1)
    else:
        cluster_feat = np.zeros((1,1))

    le = art.get("le_stock")
    if "Top_StockCode" in row.index and le is not None:
        try:
            stock_enc = np.array([[le.transform([row["Top_StockCode"]])[0]]])
        except:
            stock_enc = np.array([[0]])
    else:
        stock_enc = np.array([[0]])

    dense = np.hstack([numeric_scaled, encoded, cluster_feat])

    return dense, stock_enc
if page == "Customer Lookup":
    st.title("Customer Lookup & Real-Time LTV Prediction")

    cust_id = st.text_input("Enter Customer ID or email")
    btn = st.button("Search")

    if btn and cust_id:
        df = customer_agg.copy()
        row = None

        if cust_id.isdigit():
            row = df[df["customer_id"] == int(cust_id)]
        else:
            if "email" in df.columns:
                row = df[df["email"].str.contains(cust_id, case=False, na=False)]
            else:
                row = df[df["customer_id"].astype(str).str.contains(cust_id)]

        if row is None or row.empty:
            st.warning("Customer not found.")
        else:
            st.subheader("Customer Profile")
            st.dataframe(row.T)

            # Traditional model (RF/XGB)
            features = ["recency_days", "frequency", "monetary_sum"]
            X = row[features].fillna(0)

            if "rf_model" in art:
                preds = art["rf_model"].predict(X)
                st.metric("Traditional Model Prediction", f"{preds[0]:,.2f} USD")

            # Deep learning prediction
            if "dnn" in art:
                dense, stock_enc = prepare_dl_features(row.iloc[0], art)
                try:
                    pred_log = art["dnn"].predict([dense, stock_enc]).ravel()[0]
                except:
                    pred_log = art["dnn"].predict(
                        {"dense_input": dense, "stockcode_input": stock_enc}
                    ).ravel()[0]

                pred = np.expm1(pred_log)
                st.metric("Deep Learning LTV Prediction", f"{pred:,.2f} USD")

            # Show customer transactions
            if not transactions.empty:
                st.subheader("Recent Transactions")
                txs = transactions[
                    transactions["customer_id"] == row["customer_id"].iloc[0]
                ].sort_values("transaction_date", ascending=False).head(20)
                st.dataframe(txs)
elif page == "Segmentation Explorer":
    st.title(" Customer Segmentation Explorer")

    if customer_agg.empty:
        st.error("customer_agg missing.")
        st.stop()

    numeric_cols = [
        c for c in customer_agg.columns
        if np.issubdtype(customer_agg[c].dtype, np.number)
    ]

    features = st.multiselect(
        "Clustering Features",
        numeric_cols,
        default=["recency_days", "frequency", "monetary_sum"]
    )

    k = st.slider("KMeans clusters", 2, 12, 4)

    if st.button("Run Clustering"):
        df = customer_agg[features].fillna(0)

        scaler = StandardScaler()
        Xs = scaler.fit_transform(df)

        pca = PCA(n_components=min(3, Xs.shape[1]))
        Xp = pca.fit_transform(Xs)

        km = art.get("kmeans", None)
        if km is None:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=k, random_state=42).fit(Xp)

        customer_agg["cluster"] = km.predict(Xp)

        st.subheader("Cluster Assignments")
        st.dataframe(customer_agg[["customer_id", "cluster"] + features].head(50))

        pca_df = pd.DataFrame(Xp, columns=[f"pc{i+1}" for i in range(Xp.shape[1])])
        pca_df["cluster"] = customer_agg["cluster"].astype(str)

        fig = px.scatter_3d(
            pca_df,
            x="pc1",
            y="pc2" if Xp.shape[1] >= 2 else None,
            z="pc3" if Xp.shape[1] >= 3 else None,
            color="cluster"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Cluster Profiles")
        profiles = customer_agg.groupby("cluster")[features].mean().round(2)
        st.dataframe(profiles)
elif page == "RFM Dashboard":
    st.title(" RFM Analysis Dashboard")

    if transactions.empty:
        st.error("transactions.parquet not found")
        st.stop()

    rfm = compute_rfm(transactions)
    st.dataframe(rfm.head(50))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(px.histogram(rfm, x="recency_days", nbins=50), use_container_width=True)
    with col2:
        st.plotly_chart(px.histogram(rfm, x="frequency", nbins=50), use_container_width=True)
    with col3:
        st.plotly_chart(px.histogram(rfm, x="monetary_sum", nbins=50), use_container_width=True)
elif page == "Campaign Simulator":
    st.title("📈 Campaign Performance Simulator")

    seg_options = (
        ["All"] + list(customer_agg["segment_label"].unique())
        if "segment_label" in customer_agg.columns
        else ["All"]
    )

    segment = st.selectbox("Select Segment", seg_options)
    budget = st.number_input("Campaign Budget (USD)", 1000.0, 1_000_000.0, 10000.0)
    baseline_conv = st.slider("Baseline Conversion (%)", 0.1, 10.0, 1.0)
    lift = st.slider("Expected Lift (%)", 0.0, 200.0, 20.0)

    avg_order_value = float(customer_agg["monetary_sum"].mean())

    if st.button("Simulate"):
        if segment == "All":
            audience = len(customer_agg)
        else:
            audience = len(customer_agg[customer_agg["segment_label"] == segment])

        base_sales = audience * (baseline_conv / 100)
        new_sales = audience * ((baseline_conv * (1 + lift/100)) / 100)

        incr_sales = max(0, new_sales - base_sales)
        incr_revenue = incr_sales * avg_order_value
        roi = (incr_revenue - budget) / budget

        st.metric("Audience Size", audience)
        st.metric("Incremental Revenue", f"{incr_revenue:,.2f} USD")
        st.metric("ROI", f"{roi:.2%}")


elif page == "Export Lists":
    st.title("Export Targeted Customer Lists")

    seg_options = (
        ["All"] + list(customer_agg["segment_label"].unique())
        if "segment_label" in customer_agg.columns
        else ["All"]
    )

    segment = st.selectbox("Select Segment", seg_options)
    min_monetary = st.number_input("Minimum Monetary Value", 0.0)

    df = customer_agg.copy()
    if segment != "All":
        df = df[df["segment_label"] == segment]
    df = df[df["monetary_sum"] >= min_monetary]

    st.dataframe(df.head(100))

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "target_list.csv")



elif page == "Model Monitoring":
    st.title("Model Performance & Monitoring")

    # Traditional Model
    if "rf_model" in art:
        st.subheader("Traditional Model: Prediction Distribution")
        sample = customer_agg.sample(min(1200, len(customer_agg)))
        X = sample[["recency_days", "frequency", "monetary_sum"]].fillna(0)
        sample["pred"] = art["rf_model"].predict(X)
        st.plotly_chart(px.histogram(sample, x="pred"), use_container_width=True)

    # Deep Learning Monitoring
    if "dnn" in art:
        st.subheader("Deep Learning Model Distribution")

        preds = []
        for _, r in customer_agg.sample(min(1000, len(customer_agg))).iterrows():
            d, s = prepare_dl_features(r, art)
            pred_log = art["dnn"].predict([d, s]).ravel()[0]
            preds.append(np.expm1(pred_log))

        st.plotly_chart(px.histogram(preds, nbins=50), use_container_width=True)


st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select View",
    [
        "Customer Lookup",
        "Segmentation Explorer",
        "RFM Dashboard",
        "Campaign Simulator",
        "Export Lists",
        "Model Monitoring"
    ]
)
if page == "Customer Lookup":
    st.title("Customer Lookup & Real-Time LTV Prediction")

    cust_id = st.text_input("Enter Customer ID or email")
    btn = st.button("Search")

    if btn and cust_id:
        df = customer_agg.copy()
        row = None

        if cust_id.isdigit():
            row = df[df["customer_id"] == int(cust_id)]
        else:
            if "email" in df.columns:
                row = df[df["email"].str.contains(cust_id, case=False, na=False)]
            else:
                row = df[df["customer_id"].astype(str).str.contains(cust_id)]

        if row is None or row.empty:
            st.warning("Customer not found.")
        else:
            st.subheader("Customer Profile")
            st.dataframe(row.T)

            # Traditional model (RF/XGB)
            features = ["recency_days", "frequency", "monetary_sum"]
            X = row[features].fillna(0)

            if "rf_model" in art:
                preds = art["rf_model"].predict(X)
                st.metric("Traditional Model Prediction", f"{preds[0]:,.2f} USD")

            # Deep learning prediction
            if "dnn" in art:
                dense, stock_enc = prepare_dl_features(row.iloc[0], art)
                try:
                    pred_log = art["dnn"].predict([dense, stock_enc]).ravel()[0]
                except:
                    pred_log = art["dnn"].predict(
                        {"dense_input": dense, "stockcode_input": stock_enc}
                    ).ravel()[0]

                pred = np.expm1(pred_log)
                st.metric("Deep Learning LTV Prediction", f"{pred:,.2f} USD")

            # Show customer transactions
            if not transactions.empty:
                st.subheader("Recent Transactions")
                txs = transactions[
                    transactions["customer_id"] == row["customer_id"].iloc[0]
                ].sort_values("transaction_date", ascending=False).head(20)
                st.dataframe(txs)
elif page == "Segmentation Explorer":
    st.title(" Customer Segmentation Explorer")

    if customer_agg.empty:
        st.error("customer_agg missing.")
        st.stop()

    numeric_cols = [
        c for c in customer_agg.columns
        if np.issubdtype(customer_agg[c].dtype, np.number)
    ]

    features = st.multiselect(
        "Clustering Features",
        numeric_cols,
        default=["recency_days", "frequency", "monetary_sum"]
    )

    k = st.slider("KMeans clusters", 2, 12, 4)

    if st.button("Run Clustering"):
        df = customer_agg[features].fillna(0)

        scaler = StandardScaler()
        Xs = scaler.fit_transform(df)

        pca = PCA(n_components=min(3, Xs.shape[1]))
        Xp = pca.fit_transform(Xs)

        km = art.get("kmeans", None)
        if km is None:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=k, random_state=42).fit(Xp)

        customer_agg["cluster"] = km.predict(Xp)

        st.subheader("Cluster Assignments")
        st.dataframe(customer_agg[["customer_id", "cluster"] + features].head(50))

        pca_df = pd.DataFrame(Xp, columns=[f"pc{i+1}" for i in range(Xp.shape[1])])
        pca_df["cluster"] = customer_agg["cluster"].astype(str)

        fig = px.scatter_3d(
            pca_df,
            x="pc1",
            y="pc2" if Xp.shape[1] >= 2 else None,
            z="pc3" if Xp.shape[1] >= 3 else None,
            color="cluster"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Cluster Profiles")
        profiles = customer_agg.groupby("cluster")[features].mean().round(2)
        st.dataframe(profiles)
elif page == "RFM Dashboard":
    st.title(" RFM Analysis Dashboard")

    if transactions.empty:
        st.error("transactions.parquet not found")
        st.stop()

    rfm = compute_rfm(transactions)
    st.dataframe(rfm.head(50))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(px.histogram(rfm, x="recency_days", nbins=50), use_container_width=True)
    with col2:
        st.plotly_chart(px.histogram(rfm, x="frequency", nbins=50), use_container_width=True)
    with col3:
        st.plotly_chart(px.histogram(rfm, x="monetary_sum", nbins=50), use_container_width=True)
elif page == "Campaign Simulator":
    st.title("📈 Campaign Performance Simulator")

    seg_options = (
        ["All"] + list(customer_agg["segment_label"].unique())
        if "segment_label" in customer_agg.columns
        else ["All"]
    )

    segment = st.selectbox("Select Segment", seg_options)
    budget = st.number_input("Campaign Budget (USD)", 1000.0, 1_000_000.0, 10000.0)
    baseline_conv = st.slider("Baseline Conversion (%)", 0.1, 10.0, 1.0)
    lift = st.slider("Expected Lift (%)", 0.0, 200.0, 20.0)

    avg_order_value = float(customer_agg["monetary_sum"].mean())

    if st.button("Simulate"):
        if segment == "All":
            audience = len(customer_agg)
        else:
            audience = len(customer_agg[customer_agg["segment_label"] == segment])

        base_sales = audience * (baseline_conv / 100)
        new_sales = audience * ((baseline_conv * (1 + lift/100)) / 100)

        incr_sales = max(0, new_sales - base_sales)
        incr_revenue = incr_sales * avg_order_value
        roi = (incr_revenue - budget) / budget

        st.metric("Audience Size", audience)
        st.metric("Incremental Revenue", f"{incr_revenue:,.2f} USD")
        st.metric("ROI", f"{roi:.2%}")


elif page == "Export Lists":
    st.title("Export Targeted Customer Lists")

    seg_options = (
        ["All"] + list(customer_agg["segment_label"].unique())
        if "segment_label" in customer_agg.columns
        else ["All"]
    )

    segment = st.selectbox("Select Segment", seg_options)
    min_monetary = st.number_input("Minimum Monetary Value", 0.0)

    df = customer_agg.copy()
    if segment != "All":
        df = df[df["segment_label"] == segment]
    df = df[df["monetary_sum"] >= min_monetary]

    st.dataframe(df.head(100))

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "target_list.csv")



elif page == "Model Monitoring":
    st.title("Model Performance & Monitoring")

    # Traditional Model
    if "rf_model" in art:
        st.subheader("Traditional Model: Prediction Distribution")
        sample = customer_agg.sample(min(1200, len(customer_agg)))
        X = sample[["recency_days", "frequency", "monetary_sum"]].fillna(0)
        sample["pred"] = art["rf_model"].predict(X)
        st.plotly_chart(px.histogram(sample, x="pred"), use_container_width=True)

    # Deep Learning Monitoring
    if "dnn" in art:
        st.subheader("Deep Learning Model Distribution")

        preds = []
        for _, r in customer_agg.sample(min(1000, len(customer_agg))).iterrows():
            d, s = prepare_dl_features(r, art)
            pred_log = art["dnn"].predict([d, s]).ravel()[0]
            preds.append(np.expm1(pred_log))

        st.plotly_chart(px.histogram(preds, nbins=50), use_container_width=True)

