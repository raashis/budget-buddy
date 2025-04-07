import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import time

# --- Setup ---
st.set_page_config(page_title="💰 BudgetBuddy", layout="wide")

# --- GPT-2 Model (Load Once) ---
@st.cache_resource
def load_gpt2():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    return tokenizer, model

tokenizer, model = load_gpt2()

def get_gpt_response(user_input):
    input_ids = tokenizer.encode(user_input, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- Helper Functions ---
def load_expenses():
    try:
        return pd.read_csv("expenses.csv", names=["date", "category", "amount", "note"])
    except:
        return pd.DataFrame(columns=["date", "category", "amount", "note"])

def add_expense(category, amount, note):
    with st.spinner("Adding expense..."):
        time.sleep(1)
        df = pd.DataFrame([{
            "date": datetime.now().strftime("%Y-%m-%d"),
            "category": category,
            "amount": amount,
            "note": note
        }])
        df.to_csv("expenses.csv", mode='a', header=False, index=False)
        st.success("🎉 Expense added!")

def predict_next_month():
    df = load_expenses()
    if df.empty:
        return 0.0
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
    return df.groupby("month")["amount"].sum().mean()

# --- Fake Auth (Frontend Only) ---
if "is_logged_in" not in st.session_state:
    st.session_state.is_logged_in = False

def login_page():
    st.title("🔐 Login to BudgetBuddy")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if email and password:
            st.session_state.is_logged_in = True
            st.success("Logged in successfully!")
        else:
            st.error("Please enter both email and password.")

    st.info("Don't have an account? Go to 'Register' from the sidebar.")

def register_page():
    st.title("📝 Register for BudgetBuddy")
    username = st.text_input("Username")
    email = st.text_input("Email")
    phone_number = st.text_input("Phone Number")
    profile_pic = st.file_uploader("Upload Profile Picture", type=["jpg", "png", "jpeg"])

    if st.button("Register"):
        if username and email and phone_number:
            st.success("Registration successful! You can now log in.")
            st.balloons()
        else:
            st.error("Please fill in all required fields.")

# --- Main App Pages ---
def chat_page():
    st.subheader("💬 Chat with Your Financial Assistant")
    user_input = st.text_area("Ask anything about money, budgeting, etc.")
    if st.button("Send"):
        response = get_gpt_response(user_input)
        st.write(response)
        st.success("💡 Tip: You can meal prep to save more on food.")

def add_expense_page():
    st.subheader("🧾 Add an Expense")
    with st.form("expense_form"):
        category = st.selectbox("Category", ["Food", "Transport", "Rent", "Utilities", "Entertainment", "Other"])
        amount = st.number_input("Amount", min_value=0.0)
        note = st.text_input("Note (optional)")
        submitted = st.form_submit_button("Add Expense")
        if submitted:
            add_expense(category, amount, note)

def dashboard_page():
    st.subheader("📊 Expense Dashboard")
    df = load_expenses()

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df["month"] = df["date"].dt.to_period("M")

        total_spent = df["amount"].sum()
        avg_spent = df.groupby("month")["amount"].sum().mean()
        top_category = df.groupby("category")["amount"].sum().idxmax()

        k1, k2, k3 = st.columns(3)
        k1.metric("💵 Total Spent", f"${total_spent:.2f}")
        k2.metric("📈 Monthly Avg", f"${avg_spent:.2f}")
        k3.metric("🏆 Top Category", top_category)

        fig1 = px.pie(df, names='category', values='amount', title='Spending by Category')
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.line(df.groupby("month")["amount"].sum().reset_index(),
                       x="month", y="amount", title="Monthly Spending Trend")
        st.plotly_chart(fig2, use_container_width=True)

        with st.expander("📋 View All Expenses"):
            st.dataframe(df.sort_values(by="date", ascending=False))
    else:
        st.info("No expenses found. Start by adding one!")

    st.subheader("🔮 Budget Prediction")
    pred = predict_next_month()
    st.success(f"Estimated next month’s budget: ${pred:.2f}")

# --- Sidebar Navigation ---
st.sidebar.image("https://img.icons8.com/external-flaticons-lineal-color-flat-icons/64/000000/external-budget-planning-flaticons-lineal-color-flat-icons.png", width=100)
st.sidebar.title("👋 BudgetBuddy")

if not st.session_state.is_logged_in:
    auth_choice = st.sidebar.radio("Choose an option", ["Login", "Register"])
    if auth_choice == "Login":
        login_page()
    else:
        register_page()
else:
    st.sidebar.success("✅ Logged in")
    page = st.sidebar.radio("Navigate", ["Dashboard", "Add Expense", "Chat Assistant"])
    if page == "Dashboard":
        dashboard_page()
    elif page == "Add Expense":
        add_expense_page()
    elif page == "Chat Assistant":
        chat_page()
    if st.sidebar.button("Logout"):
        st.session_state.is_logged_in = False
        st.experimental_rerun()
