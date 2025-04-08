# 💖 Pink-Themed BudgetBuddy with All Enhanced Features
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import time
import random

# --- Setup ---
st.set_page_config(page_title="🎀 BudgetBuddy", layout="wide")

# --- Dark Mode Toggle ---
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

mode = st.sidebar.toggle("🌚 Dark Mode", value=st.session_state.dark_mode)
st.session_state.dark_mode = mode

primary_color = "#ffe4ec" if not mode else "#2c003e"
text_color = "#c2185b" if not mode else "#ffd6ef"
button_color = "#f8bbd0" if not mode else "#b40060"

# --- Styling ---
st.markdown(f"""
    <style>
    .stApp {{
        background-color: {primary_color};
        color: {text_color};
    }}
    h1, h2, h3, h4, h5, h6, .stMarkdown {{
        color: {text_color} !important;
    }}
    .stButton>button {{
        background-color: {button_color};
        color: white;
        border-radius: 8px;
    }}
    .stButton>button:hover {{
        background-color: #f48fb1;
        color: white;
    }}
    </style>
""", unsafe_allow_html=True)

# --- GPT-2 Model for Chatbot ---
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

# --- Loaders ---
def load_expenses():
    try:
        return pd.read_csv("expenses.csv", names=["date", "category", "amount", "note", "recurring"])
    except:
        return pd.DataFrame(columns=["date", "category", "amount", "note", "recurring"])

def load_savings():
    try:
        return pd.read_csv("savings_jars.csv")
    except:
        return pd.DataFrame(columns=["goal_name", "target_amount", "current_amount", "description"])

def load_reminders():
    try:
        return pd.read_csv("reminders.csv")
    except:
        return pd.DataFrame(columns=["name", "amount", "due_date", "remind_7", "remind_3"])

def predict_next_month():
    df = load_expenses()
    if df.empty:
        return 0.0
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M")
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
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    login = st.button("Login")
    if login and email and password:
        st.session_state.is_logged_in = True
        st.rerun()
    elif login:
        st.error("Please enter both email and password.")

def register_page():
    st.title("📝 Register for BudgetBuddy")
    username = st.text_input("Username")
    email = st.text_input("Email")
    phone = st.text_input("Phone Number")
    pic = st.file_uploader("Upload Profile Picture")
    if st.button("Register"):
        if username and email and phone:
            st.success("Registration successful! You can now log in.")
            st.balloons()
        else:
            st.error("Please complete all fields.")

# --- Pages ---
def dashboard_page():
    st.subheader("💸 Expense Dashboard")
    df = load_expenses()
    if df.empty:
        st.info("No expenses yet. Add some to get started!")
        return

    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M")

    period = st.selectbox("View by time period:", ["Weekly", "Monthly", "Yearly"])
    filtered_df = filter_by_period(df, period)

    if filtered_df.empty:
        st.warning("No expenses in this time frame.")
        return

    total_spent = filtered_df["amount"].sum()
    avg_spent = filtered_df.groupby("month")["amount"].sum().mean()
    top_category = filtered_df.groupby("category")["amount"].sum().idxmax()

    k1, k2, k3 = st.columns(3)
    k1.metric("💵 Total Spent", f"${total_spent:.2f}")
    k2.metric("📈 Avg per Month", f"${avg_spent:.2f}")
    k3.metric("🏆 Top Category", top_category)

    st.plotly_chart(px.pie(filtered_df, names='category', values='amount'))
    trend_df = filtered_df.copy()
    trend_df["month"] = trend_df["date"].dt.to_timestamp().dt.to_period("M").astype(str)
    st.plotly_chart(px.line(trend_df.groupby("month")["amount"].sum().reset_index(), x="month", y="amount"))

    st.subheader("🔁 Recurring Expenses")
    st.dataframe(df[df["recurring"] == True])

    st.subheader("💳 Budget Prediction")
    pred = predict_next_month()
    st.success(f"Estimated next month’s budget: ${pred:.2f}")

    st.subheader("🌟 Financial Goals")
    goal = st.text_input("Set a financial goal (e.g. Save $1000)")
    progress = st.slider("Progress toward goal", 0, 100, 25)
    if goal:
        st.success(f"You're {progress}% there to your goal: '{goal}'! Keep it up 💖")
        if progress == 100:
            st.balloons()
            st.success("🏅 You've reached your goal! Reward unlocked!")

def add_expense_page():
    st.subheader("🧳 Add an Expense")
    with st.form("expense_form"):
        category = st.selectbox("Category", ["Food", "Transport", "Rent", "Utilities", "Entertainment", "Other"])
        amount = st.number_input("Amount", min_value=0.0)
        note = st.text_input("Note (optional)")
        recurring = st.checkbox("Recurring?")
        submitted = st.form_submit_button("Add Expense")
        if submitted:
            df = pd.DataFrame([{ "date": datetime.now().strftime("%Y-%m-%d"),
                                "category": category,
                                "amount": amount,
                                "note": note,
                                "recurring": recurring }])
            df.to_csv("expenses.csv", mode='a', header=False, index=False)
            st.success("🎉 Expense added!")

def chat_page():
    st.subheader("💬 Chat with Your Financial Assistant")
    user_input = st.text_area("Ask anything about money, budgeting, etc.")
    if st.button("Send"):
        response = get_gpt_response(user_input)
        st.write(response)
        st.success("💡 Tip: You can meal prep to save more on food.")

def savings_jars_page():
    st.subheader("🎰 Savings Jars")
    df = load_savings()
    with st.form("savings_form"):
        goal_name = st.text_input("Goal Name")
        target_amount = st.number_input("Target Amount", min_value=0.0)
        current_amount = st.number_input("Current Saved", min_value=0.0)
        description = st.text_input("What is this for?")
        if st.form_submit_button("Add Jar"):
            new = pd.DataFrame([{"goal_name": goal_name, "target_amount": target_amount,
                                 "current_amount": current_amount, "description": description}])
            new.to_csv("savings_jars.csv", mode='a', header=False, index=False)
            st.success("Jar added!")
    if not df.empty:
        for _, row in df.iterrows():
            st.progress(int((row.current_amount / row.target_amount) * 100))
            st.write(f"{row.goal_name} ({row.description}): ${row.current_amount} / ${row.target_amount}")

def reminders_page():
    st.subheader("📅 Recurring Payment Reminders")
    df = load_reminders()
    with st.form("reminder_form"):
        name = st.text_input("Payment Name")
        amount = st.number_input("Amount", min_value=0.0)
        due_date = st.date_input("Due Date")
        remind_7 = st.checkbox("Remind me 7 days before")
        remind_3 = st.checkbox("Remind me 3 days before")
        if st.form_submit_button("Add Reminder"):
            new = pd.DataFrame([{"name": name, "amount": amount, "due_date": due_date,
                                 "remind_7": remind_7, "remind_3": remind_3}])
            new.to_csv("reminders.csv", mode='a', header=False, index=False)
            st.success("Reminder added!")
    if not df.empty:
        st.dataframe(df)
        calendar_df = df.copy()
        calendar_df["due_date"] = pd.to_datetime(calendar_df["due_date"])
        st.plotly_chart(px.timeline(calendar_df, x_start="due_date", x_end="due_date", y="name", title="Upcoming Reminders"))

def summary_page():
    st.subheader("📧 Weekly Summary Email Preview")
    df = load_expenses()
    if df.empty:
        st.info("No expenses found to generate summary.")
        return
    df["date"] = pd.to_datetime(df["date"])
    this_week = df[df["date"] >= datetime.now() - timedelta(days=7)]
    st.write(f"Total Spent: ${this_week['amount'].sum():.2f}")
    if not this_week.empty:
        top_categories = this_week.groupby("category")["amount"].sum().sort_values(ascending=False).head(3)
        st.write("Top Categories:")
        st.write(top_categories)

# --- Navigation ---
st.sidebar.image("https://img.icons8.com/external-flaticons-lineal-color-flat-icons/64/000000/external-budget-planning-flaticons-lineal-color-flat-icons.png", width=100)
st.sidebar.title("🎀 Budget Buddy 🎀")

if not st.session_state.is_logged_in:
    page = st.sidebar.radio("Choose an option", ["Login", "Register"])
    if page == "Login":
        login_page()
    else:
        register_page()
else:
    st.sidebar.success("✅ You're logged in!")
    page = st.sidebar.radio("Navigate", ["Dashboard", "Add Expense", "Chat Assistant", "Savings Jars", "Reminders", "Weekly Summary"])
    if page == "Dashboard":
        dashboard_page()
    elif page == "Add Expense":
        add_expense_page()
    elif page == "Chat Assistant":
        chat_page()
    elif page == "Savings Jars":
        savings_jars_page()
    elif page == "Reminders":
        reminders_page()
    elif page == "Weekly Summary":
        summary_page()
    if st.sidebar.button("Logout"):
        st.session_state.is_logged_in = False
        st.rerun()

# --- Generate Fake Data ---
if st.sidebar.checkbox("Generate Fake Data"):
    # Expenses
    cats = ["Food", "Transport", "Rent", "Utilities", "Entertainment", "Other"]
    df = pd.DataFrame([{
        "date": (datetime.today() - timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d"),
        "category": random.choice(cats),
        "amount": round(random.uniform(10, 300), 2),
        "note": "Sample note",
        "recurring": random.choice([True, False])
    } for _ in range(100)])
    df.to_csv("expenses.csv", index=False, header=False)

    # Savings Jars
    pd.DataFrame([
        {"goal_name": "Paris Trip", "target_amount": 1500, "current_amount": 400, "description": "Summer vacation"},
        {"goal_name": "Laptop Fund", "target_amount": 2000, "current_amount": 1200, "description": "For school work"},
    ]).to_csv("savings_jars.csv", index=False)

    # Reminders
    pd.DataFrame([
        {"name": "Rent", "amount": 800, "due_date": datetime.now().strftime("%Y-%m-%d"), "remind_7": True, "remind_3": True},
        {"name": "Spotify", "amount": 10.99, "due_date": (datetime.now()+timedelta(days=10)).strftime("%Y-%m-%d"), "remind_7": False, "remind_3": True},
    ]).to_csv("reminders.csv", index=False)

    st.success("✅ Fake data generated for all modules!")
