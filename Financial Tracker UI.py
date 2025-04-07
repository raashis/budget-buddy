import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import time

# --- Load GPT-2 Model ---
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

def get_gpt_response(user_input):
    # Encode the input text using the GPT-2 tokenizer
    input_ids = tokenizer.encode(user_input, return_tensors="pt")

    # Generate a response using the GPT-2 model
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

    # Decode the generated response and return it
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# --- Page Config ---
st.set_page_config(page_title="💰 BudgetBuddy", layout="wide")

# --- Sidebar ---
st.sidebar.image("https://img.icons8.com/external-flaticons-lineal-color-flat-icons/64/000000/external-budget-planning-flaticons-lineal-color-flat-icons.png", width=100)
st.sidebar.title("👋 Welcome to BudgetBuddy")
st.sidebar.markdown("Manage your expenses with AI help.")
st.sidebar.markdown("---")

# Dark Mode toggle
dark_mode = st.sidebar.checkbox("Dark Mode", value=False)
if dark_mode:
    st.markdown("""
        <style>
        body {
            background-color: #333333;
            color: white;
        }
        .css-1g6z4gm {
            background-color: #333333;
        }
        .css-1lcbn3d {
            background-color: #333333;
        }
        .css-1aeh0bm {
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        body {
            background-color: #fff0f5;
            color: black;
        }
        .css-1g6z4gm {
            background-color: #fff0f5;
        }
        </style>
    """, unsafe_allow_html=True)

# --- Helper Functions ---
def load_expenses():
    try:
        return pd.read_csv("expenses.csv", names=["date", "category", "amount", "note"])
    except:
        return pd.DataFrame(columns=["date", "category", "amount", "note"])

def add_expense(category, amount, note):
    # Simulating loading for progress bar
    with st.spinner("Adding expense..."):
        time.sleep(1)  # Simulate a delay in expense addition
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

# --- Registration ---
def user_registration():
    st.subheader("📝 Register for BudgetBuddy")
    username = st.text_input("Username")
    email = st.text_input("Email")
    phone_number = st.text_input("Phone Number")
    profile_pic = st.file_uploader("Upload your Profile Picture", type=["jpg", "png", "jpeg"])
    if st.button("Register"):
        if username and email and phone_number:
            st.session_state.username = username
            st.session_state.email = email
            st.session_state.phone_number = phone_number
            if profile_pic:
                st.image(profile_pic, caption="Profile Picture", width=150)
            st.success(f"Welcome {username}! Your profile has been created.")
        else:
            st.error("Please fill in all fields!")

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
        response = get_gpt_response(user_input)
        st.write(response)
        st.success("💡 Tip: You can meal prep to save more on food.")

# --- Add Expense ---
with col2:
    st.subheader("🧾 Add an Expense")
    with st.form("expense_form"):
        category = st.selectbox("Category", ["Food", "Transport", "Rent", "Utilities", "Entertainment", "Other"], key="category_select")
        amount = st.number_input(f"Amount", min_value=0.0, key="amount_input")
        note = st.text_input("Note (optional)", key="note_input")
        submitted = st.form_submit_button("Add Expense")
        if submitted:
            add_expense(category, amount, note)

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
    k1.metric("💵 Total Spent", f"${total_spent:.2f}")
    k2.metric("📈 Monthly Avg", f"${avg_spent:.2f}")
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
st.success(f"Estimated budget for next month: ${pred:.2f}")

# --- User Registration Page ---
if "username" not in st.session_state:
    user_registration()
else:
    st.write(f"Welcome back, {st.session_state.username}! 🎉")
