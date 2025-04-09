# 💖 Pink-Themed BudgetBuddy with Enhanced Features
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import time
import os
import random
import base64

# --- Setup ---
st.set_page_config(page_title="🎀 BudgetBuddy", layout="wide")

# --- Pink Theme Styling + Dark Mode Support ---
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

light_theme = """
    <style>
    body {
        background-color: #ffe4ec;
        color: #000000;
    }
    .stApp {
        background-color: #ffe4ec;
    }
    h1, h2, h3, h4, h5, h6, .stMarkdown, .css-1v0mbdj, .css-10trblm, .css-15zrgzn, .st-bb, .st-cb,
    label, .stTextInput label, .stFileUploader label, .stSelectbox label, .stCheckbox label {
        color: #000000 !important;
    }
    .stButton>button {
        background-color: #f8bbd0;
        color: #880e4f;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #f48fb1;
        color: white;
    }
    .stSelectbox>div>div>div {
        background-color: #fff0f5;
        color: #000000 !important;
    }

    </style>
"""

dark_theme = """
    <style>
    body {
        background-color: #1c1f26;
        color: #f8bbd0;
    }
    .stApp {
        background-color: #1c1f26;
    }
    h1, h2, h3, h4, h5, h6, .stMarkdown, .css-1v0mbdj, .css-10trblm, .css-15zrgzn, .st-bb, .st-cb,
    label, .stTextInput label, .stFileUploader label, .stSelectbox label, .stCheckbox label {
        color: #f8bbd0 !important;
    }
    .stButton>button {
        background-color: #3b4a70;
        color: #f8bbd0;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #5a6c98;
        color: white;
    }
    .stSelectbox>div>div>div {
        background-color: #3b4a70;
        color: #f8bbd0 !important;
    }

    .st-emotion-cache-1d3w5wq {
            color: #ff66b2 !important;  /* Pink text color for sidebar title in dark mode */
    }

    .css-1lcb9o6 {
            color: #ff66b2 !important;  /* Pink text color for "Navigate" in dark mode */
    }

    </style>
"""

sidebar_style = """
<style>
[data-testid="stSidebar"] {
    background-color: #1f1f29 !important; /* Always-dark sidebar */
}

/* Sidebar title */
[data-testid="stSidebar"] h1 {
    color: #f8bbd0 !important;
    font-weight: bold;
}

/* Sidebar checkbox styling */
[data-testid="stSidebar"] .stCheckbox > div {
    align-items: center;
    gap: 8px;
}
[data-testid="stSidebar"] .stCheckbox > div > label {
    color: #f8bbd0 !important;
    font-size: 16px;
    font-weight: 500;
}

/* Sidebar radio button labels */
[data-testid="stSidebar"] .stRadio > div > label {
    color: #f8bbd0 !important;
    font-size: 16px;
    font-weight: 500;
}

/* Make all sidebar labels white */
[data-testid="stSidebar"] label {
    color: white !important;
    font-weight: 500;
}


</style>
"""

st.markdown(sidebar_style, unsafe_allow_html=True)

# Apply the correct theme based on dark mode toggle
if st.session_state.dark_mode:
    st.markdown(dark_theme, unsafe_allow_html=True)
else:
    st.markdown(light_theme, unsafe_allow_html=True)

# --- Initialize session state ---
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "theme_toggled" not in st.session_state:
    st.session_state.theme_toggled = False

# --- Apply Theme ---
if st.session_state.dark_mode:
    st.markdown(dark_theme, unsafe_allow_html=True)
else:
    st.markdown(light_theme, unsafe_allow_html=True)

# --- Toggle Button ---
toggle = st.sidebar.checkbox("🌙 Toggle Dark Mode", value=st.session_state.dark_mode)

# --- Handle Toggle Logic ---
if toggle != st.session_state.dark_mode:
    st.session_state.dark_mode = toggle
    st.session_state.theme_toggled = True
    st.rerun()

# --- After rerun, reset the toggle flag ---
if st.session_state.theme_toggled:
    st.session_state.theme_toggled = False


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
    # Replace this path with the correct path to your CSV file
    try:
        df = pd.read_csv("expenses.csv", on_bad_lines='skip')
        df["date"] = pd.to_datetime(df["date"])
        return df
    except FileNotFoundError:
        st.error("The expenses.csv file was not found.")
        return pd.DataFrame()  # Return an empty DataFrame if file is not found

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
        file_exists = os.path.isfile("expenses.csv")
        df.to_csv("expenses.csv", mode='a', header=not file_exists, index=False)
        st.success("🎉 Expense added!")

# --- Fake Auth ---
if "is_logged_in" not in st.session_state:
    st.session_state.is_logged_in = False

def login_page():
    st.title("🎀 Track your every last cent... with BudgetBuddy!")
    st.markdown("<p style='font-size:18px;'>Login here 🐥</p>", unsafe_allow_html=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login = st.button("Login")
    
    if login and username and password:
        # Update session state for login status
        st.session_state.is_logged_in = True
        st.session_state.just_logged_in = True
        st.session_state.just_registered = False  # Make sure it's reset if re-login
        page = "Dashboard"  # Directly assign page to 'Dashboard'
    elif login:
        st.error("Please enter both username and password.")

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
            st.session_state.is_logged_in = True
            st.session_state.just_registered = True
            st.session_state.just_logged_in = False  # Make sure it's reset after registration
            page = "Add Expense"  # Redirect to Add Expense after registration
        else:
            st.error("Please fill in all required fields.")

st.sidebar.title("🎀 Budget Buddy 🎀")

# Show the login or register page if not logged in
if not st.session_state.is_logged_in:
    auth_choice = st.sidebar.radio("Choose an option", ["Login", "Register"])
    if auth_choice == "Login":
        login_page()  # Your login page logic here
    else:
        register_page()  # Your registration page logic here
else:
    # After logging in, immediately show the navigation sidebar options
    page = st.sidebar.radio("Navigate", ["Dashboard", "Add Expense", "Savings Jars", "Reminders", "Chat Assistant", "Weekly Summary"])

    # Show logged-in success message
    st.sidebar.success("✅ You're logged in!")

    # Handle logout
    if st.sidebar.button("Logout"):
        st.session_state.is_logged_in = False
        st.experimental_rerun()  # Rerun the app to show login/register page

    # Display selected page content
    if page == "Dashboard":
        st.header("📊 Dashboard")
        # Dashboard content here

    elif page == "Add Expense":
        st.header("➕ Add New Expense")
        # Add Expense content here

    elif page == "Savings Jars":
        st.header("🏦 Savings Jars")
        # Savings Jars content here

    elif page == "Reminders":
        st.header("🔔 Payment Reminders")
        # Reminders content here

    elif page == "Chat Assistant":
        st.header("🧠 Chat with BudgetBuddy")
        # Chat Assistant content here

    elif page == "Weekly Summary":
        st.header("📬 Weekly Summary Email Preview")
        # Weekly Summary content here


    # --- Page Routing Logic ---
    if page == "Dashboard":

        df = load_expenses()
        if not df.empty:
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
            df["date"] = pd.to_datetime(df["date"])
            spending_by_category = df.groupby("category")["amount"].sum().reset_index()

            # Pie Chart with only plot background Merlot
            fig = px.pie(spending_by_category, names="category", values="amount", title="Spending by Category")
            fig.update_layout(
                # plot_bgcolor="#500071",
                paper_bgcolor="#C8D8B9",
                font=dict(color="white"),
                title_font=dict(size=20, color="white"),
            )
            st.plotly_chart(fig)


            # # Line Chart with only plot background mint
            # trend = df.groupby(df["date"].dt.to_period("M")).sum(numeric_only=True).reset_index()
            # trend["date"] = trend["date"].astype(str)

            # fig2 = px.line(trend, x="date", y="amount", title="Spending Trend Over Time")
            # fig2.update_layout(
            #     # plot_bgcolor="#500071",
            #     paper_bgcolor='#C8D8B9',   # mint green
            #     font=dict(color="black"),   # Changed to black for better visibility
            #     xaxis_title="Date",         # Custom x-axis label
            #     yaxis_title="Amount",       # Custom y-axis label
            #     title_font=dict(size=20, color="black"),  # Adjusted title font color
            #     xaxis=dict(showgrid=True, gridcolor="white"),  # Optional: Show gridlines
            #     yaxis=dict(showgrid=True, gridcolor="white")   # Optional: Show gridlines
            # )

            # st.plotly_chart(fig2)
        else:
            st.info("No expenses yet. Add one under 'Add Expense'!")



    elif page == "Add Expense":
        category = st.selectbox("Category", ["Food", "Transport", "Shopping", "Rent", "Utilities", "Other"])
        amount = st.number_input("Amount", min_value=0.0, format="%.2f")
        note = st.text_input("Note")
        recurring = st.checkbox("Recurring?")
        if st.button("Add Expense"):
            add_expense(category, amount, note, recurring)



        # --- Savings Jars Page ---
    elif page == "Savings Jars":
        if "jars" not in st.session_state:
            st.session_state.jars = []

        # Jar setup
        jar_name = st.text_input("Jar Name")
        jar_goal = st.number_input("Financial Goal", min_value=0.0, format="%.2f")
        jar_description = st.text_input("What is this jar for?")
        jar_progress = st.slider("Progress", 0, 100, 0)

        # Coin drop animation function
        def coin_drop_animation():
            coin = "🪙"
            st.markdown("<p style='font-size:24px; text-align:center;'>Saving ongoing!</p>", unsafe_allow_html=True)
            
            # Create an empty container to hold the coin and update it with new positions
            coin_placeholder = st.empty()
            
            # Animate the coin falling (to create a smooth effect)
            for i in range(1, 11):  # Number of steps for the drop animation
                coin_placeholder.markdown(f"<p style='font-size:48px; text-align:center; padding-top:{i * 5}px;'>{coin}</p>", unsafe_allow_html=True)
                time.sleep(0.1)  # Short delay to animate the coin drop

        # Add Jar Button
        if st.button("Add Jar"):
            # Check if all fields are filled
            if jar_name and jar_goal > 0 and jar_description:
                # Create the new jar dictionary
                new_jar = {
                    "name": jar_name,
                    "goal": jar_goal,
                    "description": jar_description,
                    "progress": jar_progress
                }

                # Add the new jar to session state
                st.session_state.jars.append(new_jar)

                # Provide success feedback
                st.success(f"Jar '{jar_name}' has been added successfully!")

                # Trigger coin drop animation after adding the jar
                coin_drop_animation()

            else:
                # Provide warning if fields are missing
                st.warning("Please fill in all fields to add a jar.")

        # Read and encode the image to base64
        with open("image.png", "rb") as image_file:
            jar_image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

        # Display the image centered using HTML
        st.markdown(
            f"""
            <div style='text-align: center; margin-top: -5px;'>
                <img src='data:image/png;base64,{jar_image_base64}' width='150'>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Display progress next to the jar
        for jar in st.session_state.jars:
            st.markdown(f"<p style='font-size:16px; color:black; text-align:center;'>Progress: {jar['progress']}%</p>", unsafe_allow_html=True)


    elif page == "Reminders":
    # Define the reminders with due dates
# List to store reminders
        reminders = []

        # Ask for the number of reminders
        num_reminders = st.number_input("How many reminders would you like to add?", min_value=1, max_value=10, value=1)

        # Loop to add the reminders
        for i in range(num_reminders):
            st.subheader(f"Reminder {i+1}")

            # Get reminder details
            name = st.text_input(f"Name of reminder {i+1}")
            amount = st.number_input(f"Amount for {name}", min_value=0.0, format="%.2f")
            due_date = st.date_input(f"Due date for {name}")

            # Calculate remaining days
            days_left = (due_date - datetime.now().date()).days

            if st.button(f"Add Reminder {i+1}"):
                reminders.append({
                    "name": name,
                    "amount": amount,
                    "due": due_date,
                    "days_left": days_left
                })
                st.success(f"Reminder '{name}' has been added successfully!")

        # Display the reminders
        if reminders:
            st.subheader("Your Reminders")
            for r in reminders:
                st.markdown(f"<p style='color:#003366; font-size:16px;'>"
                            f"{r['name']} - ${r['amount']} due in {r['days_left']} days</p>",
                            unsafe_allow_html=True)
    elif page == "Chat Assistant":
        user_input = st.text_input("Ask me anything about budgeting!")
        if user_input:
            response = get_gpt_response(user_input)
            st.success(response)

    elif page == "Weekly Summary":
    # Function to load reminders from CSV
        def load_reminders():
            reminders_file = "reminders.csv"
            df = pd.read_csv(reminders_file)
            
            # Convert 'due_date' column to datetime if necessary
            df["due_date"] = pd.to_datetime(df["due_date"])
            
            return df
        
        # Function to get reminders from the past week
        def get_weekly_reminders():
            df = load_reminders()
            
            # Get today's date
            today = pd.to_datetime("today")

            # Filter for reminders due in the past week
            past_week_reminders = df[df["due_date"] >= today - pd.DateOffset(weeks=1)]

            return past_week_reminders

        # Get the past week's reminders
        weekly_reminders = get_weekly_reminders()

        # Display weekly reminders in the UI
        if not weekly_reminders.empty:
            st.dataframe(weekly_reminders)
        else:
            st.info("No reminders due in the past week.")
