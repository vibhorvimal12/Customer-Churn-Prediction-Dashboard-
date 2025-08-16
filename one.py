import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
st.title("Customer Churn Prediction Dashboard")

uploaded_file = st.file_uploader("Upload Telco Customer Churn Excel File", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.sidebar.header("Filters")
    filters = {}
    for col in df.columns:
        if df[col].dtype == "object" and df[col].nunique() < 20:
            filters[col] = st.sidebar.multiselect(f"Filter by {col}", options=df[col].unique(), default=df[col].unique())
    filtered_df = df.copy()
    for col, values in filters.items():
        filtered_df = filtered_df[filtered_df[col].isin(values)]

    st.subheader("Filtered Data")
    st.write(f"Rows: {filtered_df.shape[0]}")
    st.dataframe(filtered_df)

    churn_col = None
    for c in df.columns:
        if "churn" in c.lower():
            churn_col = c
            break

    if churn_col:
        churn_counts = filtered_df[churn_col].value_counts()
        fig, ax = plt.subplots()
        sns.countplot(x=churn_col, data=filtered_df, palette="coolwarm", ax=ax)
        ax.set_title("Churn Distribution")
        st.pyplot(fig)

        num_cols = [c for c in ["Monthly Charges", "Tenure Months"] if c in filtered_df.columns]
        if num_cols:
            churn_groups = filtered_df.groupby(churn_col)[num_cols].mean().reset_index()
            st.write("Churn vs Non-Churn Averages")
            st.dataframe(churn_groups)

            churn_groups_melted = churn_groups.melt(id_vars=churn_col, var_name="Metric", value_name="Average")
            fig, ax = plt.subplots(figsize=(7,5))
            sns.barplot(data=churn_groups_melted, x=churn_col, y="Average", hue="Metric", palette="Set2", ax=ax)
            ax.set_title("Average Monthly Charges & Tenure (Churn vs Non-Churn)")
            st.pyplot(fig)
    else:
        st.warning("No churn column found. Please check your dataset.")
else:
    st.info("Please upload the Telco Customer Churn Excel file to continue.")
