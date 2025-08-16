import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Premium Customer Churn Dashboard", layout="wide")
st.title(" Customer Churn Dashboard")

df = pd.read_excel("Telco_customer_churn.xlsx")
df.columns = df.columns.str.strip()

churn_col = None
for col in df.columns:
    unique_vals = df[col].dropna().unique()
    unique_lower = [str(x).strip().lower() for x in unique_vals]
    if set(unique_lower).issubset({"yes", "no"}) or set(unique_lower).issubset({"0", "1"}):
        churn_col = col
        break

if not churn_col:
    st.error("No churn column with Yes/No or 0/1 values found!")
    st.stop()

if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

df = df.dropna()

st.sidebar.header("Filter Customers")
filter_cols = {}
for i, col in enumerate(["gender", "Contract", "PaymentMethod", "SeniorCitizen"]):
    if col in df.columns:
        filter_cols[col] = st.sidebar.multiselect(
            label=col,
            options=df[col].unique(),
            default=df[col].unique(),
            key=f"{col}_{i}"
        )
    else:
        filter_cols[col] = None

if "tenure" in df.columns:
    tenure_filter = st.sidebar.slider(
        "Tenure (Months)", int(df["tenure"].min()), int(df["tenure"].max()),
        (int(df["tenure"].min()), int(df["tenure"].max())), key="tenure_slider"
    )
if "MonthlyCharges" in df.columns:
    monthly_filter = st.sidebar.slider(
        "Monthly Charges ($)", float(df["MonthlyCharges"].min()), float(df["MonthlyCharges"].max()),
        (float(df["MonthlyCharges"].min()), float(df["MonthlyCharges"].max())), key="monthly_slider"
    )

filtered_df = df.copy()
for col, selected in filter_cols.items():
    if col in df.columns and selected is not None:
        filtered_df = filtered_df[filtered_df[col].isin(selected)]

if "tenure" in df.columns:
    filtered_df = filtered_df[(filtered_df["tenure"] >= tenure_filter[0]) & (filtered_df["tenure"] <= tenure_filter[1])]
if "MonthlyCharges" in df.columns:
    filtered_df = filtered_df[(filtered_df["MonthlyCharges"] >= monthly_filter[0]) & (filtered_df["MonthlyCharges"] <= monthly_filter[1])]

cat_cols = filtered_df.select_dtypes(include="object").columns.tolist()
for i, col in enumerate(cat_cols):
    if col != churn_col:
        options = filtered_df[col].unique()
        selected = st.sidebar.multiselect(
            label=col,
            options=options,
            default=options,
            key=f"{col}_cat_{i}"
        )
        filtered_df = filtered_df[filtered_df[col].isin(selected)]

total_customers = len(filtered_df)
churned_customers = filtered_df[churn_col].map(lambda x: 1 if str(x).lower() == "yes" else 0).sum()
retention_rate = (total_customers - churned_customers) / total_customers
avg_monthly = filtered_df["MonthlyCharges"].mean() if "MonthlyCharges" in filtered_df.columns else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", total_customers)
col2.metric("Churned Customers", churned_customers)
col3.metric("Retention Rate", f"{retention_rate:.2%}")
col4.metric("Avg Monthly Charges", f"${avg_monthly:.2f}")

st.subheader("Dataset Preview")
st.dataframe(filtered_df.head())

st.markdown("### 🔎 Churn Overview")
col1, col2 = st.columns(2)
with col1:
    fig = px.histogram(filtered_df, x=churn_col, color=churn_col, title="Churn Distribution")
    st.plotly_chart(fig)
with col2:
    if "gender" in filtered_df.columns:
        fig = px.histogram(filtered_df, x="gender", color=churn_col, barmode="group", title="Churn by Gender")
        st.plotly_chart(fig)

st.markdown("### 👥 Customer Insights")
col3, col4 = st.columns(2)
with col3:
    if "Contract" in filtered_df.columns:
        fig = px.histogram(filtered_df, x="Contract", color=churn_col, barmode="group", title="Churn by Contract Type")
        st.plotly_chart(fig)
with col4:
    if "tenure" in filtered_df.columns:
        fig = px.histogram(filtered_df, x="tenure", color=churn_col, barmode="stack", nbins=30, title="Tenure vs Churn")
        st.plotly_chart(fig)

if "PaymentMethod" in filtered_df.columns:
    st.markdown("### 💳 Churn by Payment Method")
    payment_churn = filtered_df.groupby("PaymentMethod")[churn_col].apply(lambda x: (x.str.lower()=="yes").sum()).reset_index()
    fig = px.pie(payment_churn, names="PaymentMethod", values=churn_col, title="Churn by Payment Method", hole=0.4)
    st.plotly_chart(fig)

if "Contract" in filtered_df.columns and "tenure" in filtered_df.columns:
    st.markdown("### 📊 Contract vs Tenure Heatmap")
    pivot = pd.pivot_table(filtered_df, index="Contract", columns=pd.cut(filtered_df["tenure"], bins=10), 
                           values=churn_col, aggfunc=lambda x: (x.str.lower()=="yes").sum())
    fig, ax = plt.subplots(figsize=(10,5))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", ax=ax)
    st.pyplot(fig)

st.subheader("Correlation Heatmap")
num_cols = filtered_df.select_dtypes(include=["float64","int64"]).columns
if len(num_cols) > 1:
    corr = filtered_df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

st.markdown("### 🤖 Machine Learning Model")
X = filtered_df.drop(churn_col, axis=1)
y = filtered_df[churn_col].map(lambda x: 1 if str(x).strip().lower() == "yes" else 0)
X = pd.get_dummies(X, drop_first=True)

if len(X.columns) > 0:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.metric(label="✅ Model Accuracy", value=f"{acc:.2%}")

    st.subheader("Top 5 Features Affecting Churn")
    importances = pd.Series(model.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(5)
    st.bar_chart(top_features)

st.subheader("Predict Churn for New Customer")
if st.button("Predict Example"):
    example = X_test.iloc[[0]]
    prediction = model.predict(example)[0]
    st.write("Churn Prediction for example customer:", "Yes" if prediction==1 else "No")

csv = filtered_df.to_csv(index=False)
st.download_button(label="Download Filtered Data", data=csv, file_name="filtered_churn.csv", mime="text/csv")
