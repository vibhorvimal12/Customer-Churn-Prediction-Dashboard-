import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
st.title("📊 Customer Churn Prediction Dashboard")

df = pd.read_excel("Telco_customer_churn.xlsx")
df.columns = df.columns.str.strip()
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.markdown("### 🔎 Churn Overview")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Churn Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="Churn", data=df, ax=ax, palette="Set2")
    ax.set_title("Churn Distribution")
    fig.tight_layout()
    st.pyplot(fig)

with col2:
    st.subheader("Churn by Gender")
    fig, ax = plt.subplots()
    sns.countplot(x="gender", hue="Churn", data=df, ax=ax, palette="coolwarm")
    ax.set_title("Churn by Gender")
    fig.tight_layout()
    st.pyplot(fig)

st.markdown("### 👥 Customer Insights")
col3, col4 = st.columns(2)

with col3:
    st.subheader("Churn by Contract Type")
    fig, ax = plt.subplots()
    sns.countplot(x="Contract", hue="Churn", data=df, ax=ax, palette="viridis")
    plt.xticks(rotation=30)
    fig.tight_layout()
    st.pyplot(fig)

with col4:
    st.subheader("Tenure vs Churn")
    fig, ax = plt.subplots()
    sns.histplot(data=df, x="tenure", hue="Churn", multiple="stack", bins=30, ax=ax, palette="Set1")
    ax.set_title("Tenure vs Churn")
    fig.tight_layout()
    st.pyplot(fig)

st.markdown("### 💰 Financial Insights")
col5, col6 = st.columns(2)

with col5:
    st.subheader("Monthly Charges vs Churn")
    fig, ax = plt.subplots()
    sns.boxplot(x="Churn", y="MonthlyCharges", data=df, ax=ax, palette="Set3")
    ax.set_title("Monthly Charges vs Churn")
    fig.tight_layout()
    st.pyplot(fig)

with col6:
    st.subheader("Total Charges vs Churn")
    fig, ax = plt.subplots()
    sns.boxplot(x="Churn", y="TotalCharges", data=df, ax=ax, palette="pastel")
    ax.set_title("Total Charges vs Churn")
    fig.tight_layout()
    st.pyplot(fig)

st.markdown("### 🤖 Machine Learning Model")
X = df.drop("Churn", axis=1)
y = df["Churn"].map({"Yes": 1, "No": 0})
X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.metric(label="✅ Model Accuracy", value=f"{acc:.2%}")

st.subheader("Feature Importance")
importances = pd.Series(model.feature_importances_, index=X.columns)
fig, ax = plt.subplots(figsize=(10,6))
importances.sort_values().plot(kind='barh', ax=ax, color="teal")
ax.set_title("Feature Importance")
fig.tight_layout()
st.pyplot(fig)
