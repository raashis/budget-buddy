import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# --- Page Config ---
st.set_page_config(page_title="💰 BudgetBuddy", layout="wide")

# --- Helper Functions ---
def load_expenses():
    try:
        return pd.read_csv("expenses.csv", names=["date", "category", "amount", "note"])
    except:
        return pd.DataFrame(columns=["date", "category", "amount", "note"])

def add_expense(category, amount, note):
    df = pd.DataFrame([{
        "date": datetime.now().strftime("%Y-%m-%d"),
        "category": category,
        "amount": amount,
        "note": note
    }])
    df.to_csv("expenses.csv", mode='a', header=False, index=False)

def predict_next_month():
    df = load_expenses()
    if df.empty:
        return 0.0
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
    return df.groupby("month")["amount"].sum().mean()

# --- Sidebar ---
st.sidebar.image("https://img.icons8.com/external-flaticons-lineal-color-flat-icons/64/000000/external-budget-planning-flaticons-lineal-color-flat-icons.png", width=100)
st.sidebar.title("👋 Welcome to BudgetBuddy")
st.sidebar.markdown("Manage your expenses with AI help.")
st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Settings")
currency = st.sidebar.selectbox("Currency", ["USD", "SGD", "EUR", "INR"])
st.sidebar.markdown("---")

# --- Header ---
st.title("💰 BudgetBuddy: Your AI Financial Assistant")
st.markdown("Gain insights, track your spending, and make smart decisions 💡")

# --- Columns ---
col1, col2 = st.columns([2, 1])

# --- Chat Assistant ---
with col1:
    st.subheader("💬 Chat with Your Financial Assistant")
    st.markdown("Ask anything — budget tips, investment advice, or money-saving hacks!")
    user_input = st.text_area("Ask a question", key="chat_input")
    if st.button("Send", key="send_button"):
        st.info("🔧 GPT reply placeholder. Integrate OpenAI here.")
        st.success("💡 Tip: You can meal prep to save more on food.")

# --- Add Expense ---
with col2:
    st.subheader("🧾 Add an Expense")
    with st.form("expense_form"):
        category = st.selectbox("Category", ["Food", "Transport", "Rent", "Utilities", "Entertainment", "Other"], key="category_select")
        amount = st.number_input(f"Amount ({currency})", min_value=0.0, key="amount_input")
        note = st.text_input("Note (optional)", key="note_input")
        submitted = st.form_submit_button("Add Expense")
        if submitted:
            add_expense(category, amount, note)
            st.success("🎉 Expense added!")

# --- Dashboard ---
st.subheader("📊 Expense Dashboard")
df = load_expenses()

if not df.empty:
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M")

    # Key Metrics
    total_spent = df["amount"].sum()
    avg_spent = df.groupby("month")["amount"].sum().mean()
    top_category = df.groupby("category")["amount"].sum().idxmax()

    k1, k2, k3 = st.columns(3)
    k1.metric("💵 Total Spent", f"{currency} {total_spent:.2f}")
    k2.metric("📈 Monthly Avg", f"{currency} {avg_spent:.2f}")
    k3.metric("🏆 Top Category", top_category)

    # Charts
    fig1 = px.pie(df, names='category', values='amount', title='Spending by Category')
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.line(df.groupby("month")["amount"].sum().reset_index(),
                   x="month", y="amount", title="Monthly Spending Trend")
    st.plotly_chart(fig2, use_container_width=True)

    # Table
    with st.expander("📋 View All Expenses"):
        st.dataframe(df.sort_values(by="date", ascending=False))
else:
    st.info("No expenses found. Start by adding one!")

# --- Budget Prediction ---
st.subheader("🔮 Next Month's Budget Prediction")
pred = predict_next_month()
st.success(f"Estimated budget for next month: {currency} {pred:.2f}")
