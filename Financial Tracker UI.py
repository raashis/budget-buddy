# 💖 Pink-Themed BudgetBuddy with Enhanced Features
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import time

# --- Setup ---
st.set_page_config(page_title="🎀 BudgetBuddy", layout="wide")

# --- Inject Pink Theme ---
st.markdown("""
    <style>
        body {
            background-color: #fff0f5;
            color: #4b0082;
        }
        .css-1v0mbdj, .stApp, .block-container {
            background-color: #fff0f5;
        }
        .stButton>button {
            background-color: #ff69b4;
            color: white;
            border-radius: 10px;
        }
        .stTextInput>div>div>input, .stSelectbox>div>div {
            background-color: #ffe4e1;
            color: #4b0082;
        }
    </style>
""", unsafe_allow_html=True)

# --- Load GPT-2 Model ---
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
        return pd.read_csv("expenses.csv", names=["date", "category", "amount", "note", "recurring"])
    except:
        return pd.DataFrame(columns=["date", "category", "amount", "note", "recurring"])

def add_expense(category, amount, note, recurring):
    with st.spinner("Adding expense..."):
        time.sleep(1)
        df = pd.DataFrame([{
            "date": datetime.now().strftime("%Y-%m-%d"),
            "category": category,
            "amount": amount,
            "note": note,
            "recurring": recurring
        }])
        df.to_csv("expenses.csv", mode='a', header=False, index=False)
        st.success("🎉 Expense added!")

def predict_next_month():
    df = load_expenses()
    if df.empty:
        return 0.0
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
    return df.groupby("month")["amount"].sum().mean()

def filter_by_period(df, period):
    df["date"] = pd.to_datetime(df["date"])
    today = datetime.today()
    if period == "Weekly":
        start_date = today - timedelta(days=7)
    elif period == "Monthly":
        start_date = today - timedelta(days=30)
    else:
        start_date = today - timedelta(days=365)
    return df[df["date"] >= start_date]

# --- Fake Auth ---
if "is_logged_in" not in st.session_state:
    st.session_state.is_logged_in = False

def login_page():
    st.title("🎀 Track your every last cent... with BudgetBuddy!")
    st.markdown("<p style='font-size:18px;'>Login here 🐥</p>", unsafe_allow_html=True)
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    login = st.button("Login")
    if login and email and password:
        st.session_state.is_logged_in = True
        st.experimental_rerun()
    elif login:
        st.error("Please enter both email and password.")

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
        recurring = st.checkbox("Recurring?")
        submitted = st.form_submit_button("Add Expense")
        if submitted:
            add_expense(category, amount, note, recurring)

def dashboard_page():
    st.subheader("💸 Expense Dashboard")
    df = load_expenses()

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df["month"] = df["date"].dt.to_period("M")

        period = st.selectbox("View by time period:", ["Weekly", "Monthly", "Yearly"])
        filtered_df = filter_by_period(df, period)

        total_spent = filtered_df["amount"].sum()
        avg_spent = filtered_df.groupby("month")["amount"].sum().mean()
        top_category = filtered_df.groupby("category")["amount"].sum().idxmax()

        k1, k2, k3 = st.columns(3)
        k1.metric("💵 Total Spent", f"${total_spent:.2f}")
        k2.metric("📈 Avg per Month", f"${avg_spent:.2f}")
        k3.metric("🏆 Top Category", top_category)

        st.markdown("### 📊 Category Breakdown")
        st.plotly_chart(px.pie(filtered_df, names='category', values='amount'))

        st.markdown("### 📈 Spending Trend")
        st.plotly_chart(px.line(filtered_df.groupby(filtered_df["date"].dt.to_period("M"))["amount"].sum().reset_index(),
                                x="date", y="amount", title="Spending Over Time"))

        st.markdown("### 🔁 Recurring Expenses")
        st.dataframe(df[df["recurring"] == True])

        with st.expander("📋 View All Expenses"):
            st.dataframe(filtered_df.sort_values(by="date", ascending=False))
    else:
        st.info("No expenses found. Start by adding one!")

    st.subheader("💳 Budget Prediction")
    pred = predict_next_month()
    st.success(f"Estimated next month’s budget: ${pred:.2f}")

    st.markdown("### 🎯 Financial Goals")
    goal = st.text_input("Set a financial goal (e.g. Save $1000)")
    progress = st.slider("Progress toward goal", 0, 100, 25)
    if goal:
        st.success(f"You're {progress}% there to your goal: '{goal}'! Keep it up 💖")
        if progress == 100:
            st.balloons()
            st.success("🏅 You've reached your goal! Reward unlocked!")

# --- Sidebar Navigation ---
st.sidebar.image("https://img.icons8.com/external-flaticons-lineal-color-flat-icons/64/000000/external-budget-planning-flaticons-lineal-color-flat-icons.png", width=100)
st.sidebar.title("🎀 Budget Buddy 🎀")

if not st.session_state.is_logged_in:
    auth_choice = st.sidebar.radio("Choose an option", ["Login", "Register"])
    if auth_choice == "Login":
        login_page()
    else:
        register_page()
else:
    st.sidebar.success("✅ You're logged in!")
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

# --- Optional: Generate Fake Data for Testing ---
if st.sidebar.checkbox("Generate Fake Data"):
    import random
    cats = ["Food", "Transport", "Rent", "Utilities", "Entertainment", "Other"]
    df = pd.DataFrame([{
        "date": (datetime.today() - timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d"),
        "category": random.choice(cats),
        "amount": round(random.uniform(10, 300), 2),
        "note": "Sample note",
        "recurring": random.choice([True, False])
    } for _ in range(100)])
    df.to_csv("expenses.csv", index=False, header=False)
    st.success("✅ Fake data generated!")
