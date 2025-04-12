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

#to remove error when i login
if "device_width" not in st.session_state:
    st.session_state["device_width"] = None


#for mobile responsiveness
def detect_device():
    detect_width = """
        <script>
        const width = window.innerWidth;
        const height = window.innerHeight;
        const streamlitInput = window.parent.document.querySelectorAll('input[data-testid="stTextInput"]')[0];
        streamlitInput.value = ${width};
        streamlitInput.dispatchEvent(new Event('input', { bubbles: true }));
        </script>
    """
    if st.session_state.device_width and str(st.session_state.device_width).isdigit():
    #safe to use the value
        st.markdown(detect_width, unsafe_allow_html=True)
        device_width = st.text_input("Screen width", label_visibility="collapsed", key="device_width")
        return int(device_width) if device_width.isdigit() else None

#setting up the global layout and browser title
st.set_page_config(page_title="🎀 BudgetBuddy", layout="wide")

#the theme styling + dark mode styling too
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
            color: #ff66b2 !important;  
    }

    .css-1lcb9o6 {
            color: #ff66b2 !important;  
    }

    </style>
"""

sidebar_style = """
<style>
[data-testid="stSidebar"] {
    background-color: #1f1f29 !important;  
}

/* sidebar title */
[data-testid="stSidebar"] h1 {
    color: #f8bbd0 !important;
    font-weight: bold;
}

/* sidebar checkbox styling */
[data-testid="stSidebar"] .stCheckbox > div {
    align-items: center;
    gap: 8px;
}
[data-testid="stSidebar"] .stCheckbox > div > label {
    color: #f8bbd0 !important;
    font-size: 16px;
    font-weight: 500;
}

/* sidebar radio button labels */
[data-testid="stSidebar"] .stRadio > div > label {
    color: #f8bbd0 !important;
    font-size: 16px;
    font-weight: 500;
}

/* making all sidebar labels white */
[data-testid="stSidebar"] label {
    color: white !important;
    font-weight: 500;
}


</style>
"""

st.markdown(sidebar_style, unsafe_allow_html=True)

#applying the correct theme based on the dark mode toggle
if st.session_state.dark_mode:
    st.markdown(dark_theme, unsafe_allow_html=True)
else:
    st.markdown(light_theme, unsafe_allow_html=True)

#initialising session state variables
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "theme_toggled" not in st.session_state:
    st.session_state.theme_toggled = False

#applyig the theme
if st.session_state.dark_mode:
    st.markdown(dark_theme, unsafe_allow_html=True)
else:
    st.markdown(light_theme, unsafe_allow_html=True)

#toggle button
toggle = st.sidebar.checkbox("🌙 Toggle Dark Mode", value=st.session_state.dark_mode)

if toggle != st.session_state.dark_mode:
    st.session_state.dark_mode = toggle
    st.session_state.theme_toggled = True
    st.rerun()

#making sure to reset the toggle button after reset
if st.session_state.theme_toggled:
    st.session_state.theme_toggled = False


#loading the GPT 2 model
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


#helper fns!
def load_expenses():
    try:
        df = pd.read_csv("expenses.csv", on_bad_lines='skip')
        df["date"] = pd.to_datetime(df["date"])
        return df
    except FileNotFoundError:
        st.error("The expenses.csv file was not found.")
        return pd.DataFrame()  #if file not found j return the empty one
    
def predict_expenses(df, period_type, horizon=1):
    if period_type == 'week':
        #just taking the avergae
        return df['amount'].mean()
    elif period_type == 'month':
        return df['amount'].sum()
    else:
        return 0

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

#fake auth
if "is_logged_in" not in st.session_state:
    st.session_state.is_logged_in = False

def login_page():
    st.title("🎀 Track your every last cent... with BudgetBuddy!")
    st.markdown("<p style='font-size:18px;'>Login here 🐥</p>", unsafe_allow_html=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login = st.button("Login")

    if login and username and password:
        #setting the login state
        st.session_state.is_logged_in = True
        st.session_state.just_logged_in = True
        st.session_state.just_registered = False  #do a reset if re-login

        #setting device width before rerun 
        st.session_state.device_width = detect_device()

        # setting default landing page after login as dashboard
        st.session_state.page = "Dashboard"

        #rerun immediately 
        st.rerun()

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
            st.session_state.just_logged_in = False  
            device_width = detect_device()
            st.session_state.device_width = device_width
            page = "Add Expense"  
        else:
            st.error("Please fill in all required fields.")

st.sidebar.title("🎀 Budget Buddy 🎀")

#showing the login or register page if not logged in
if not st.session_state.is_logged_in:
    auth_choice = st.sidebar.radio("Choose an option", ["Login", "Register"])
    if auth_choice == "Login":
        login_page()  
    else:
        register_page()  
else:
    
    page = st.sidebar.radio("Navigate", ["Dashboard", "Add Expense", "Weekly Summary", "Savings Jars", "Reminders", "Chat Assistant"])

    #logged-in success message
    st.sidebar.success("✅ You're logged in!")

    #handling logout
    if st.sidebar.button("Logout"):
        st.session_state.is_logged_in = False
        st.rerun()  

    if page == "Dashboard":
        st.header("📊 Dashboard")

    elif page == "Add Expense":
        st.header("➕ Add New Expense")

    elif page == "Savings Jars":
        st.header("🏦 Savings Jars")

    elif page == "Reminders":
        st.header("🔔 Payment Reminders")

    elif page == "Chat Assistant":
        st.header("🧠 Chat with BudgetBuddy")

    elif page == "Weekly Summary":
        st.header("📬 Weekly Summary of Expenses")


    #page routing logic
    if page == "Dashboard":        
        df = load_expenses()

        col1, col2 = st.columns([2, 1])

        with col1:
            if not df.empty:
                df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

                spending_by_category = df.groupby("category")["amount"].sum().reset_index()

                title_color = "white" if st.session_state.dark_mode else "black"

                st.markdown(
                    f"""
                    <div style='
                        background-color: {"#ffebf0" if not st.session_state.dark_mode else "#331c1c"};
                        padding: 20px;
                        border-radius: 20px;
                        margin-bottom: 20px;
                        border: 2px dashed {"#ff4081" if not st.session_state.dark_mode else "#ff80ab"};
                        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
                    '>
                        <h3 style='text-align: center; color: {title_color}; margin: 0;'>Total Spending</h3>
                        <h2 style='text-align: center; color: {title_color}; margin-top: 10px;'>${df['amount'].sum():.2f}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


                #pie chart
                fig = px.pie(
                    spending_by_category, 
                    names="category", 
                    values="amount",
                    color_discrete_sequence=px.colors.sequential.RdBu,
                    hole=0.4
                )

                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(
                        color=title_color,
                        size=14
                    ),
                    legend=dict(
                        font=dict(
                            color=title_color
                        )
                    ),
                    showlegend=True
                )

                st.markdown(
                    f"<h3 style='text-align:center; color:{title_color};'>Spending by Category</h3>", 
                    unsafe_allow_html=True
                )

                st.plotly_chart(fig, use_container_width=True)

            else:
                st.info("No expenses yet. Add one under 'Add Expense'!")

                sample_data = pd.DataFrame({
                    "category": ["Food", "Transport", "Shopping", "Rent"],
                    "amount": [300, 150, 200, 800]
                })

                fig = px.pie(
                    sample_data, 
                    names="category", 
                    values="amount", 
                    title="Sample Spending (Add your expenses to see your data)",
                    opacity=0.7,
                    color_discrete_sequence=px.colors.sequential.Tealgrn
                )

                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown(f"<h3 style='color:{title_color};'>📉 Expense Forecasts</h3>", unsafe_allow_html=True)

            if not df.empty and len(df) >= 2:
                try:
                    weekly_prediction = predict_expenses(df, 'week', horizon=1)
                    monthly_prediction = predict_expenses(df, 'month', horizon=1)

                    st.markdown(
                        f"""
                        <div style='color:{title_color}; font-size: 20px;'>
                            <p><strong>Next Week Forecast:</strong> ${weekly_prediction:.2f}</p>
                            <p><strong>Next Month Forecast:</strong> ${monthly_prediction:.2f}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                except Exception:
                    st.info("Need more data for predictions")

                tip_color = "white" if st.session_state.dark_mode else "black"
                tip_bg = "#222222" if st.session_state.dark_mode else "#f0f0f0"
                st.markdown(
                    f"""
                    <div style='background-color:{tip_bg}; padding:15px; border-radius:10px; color:{tip_color};'>
                    💡 Predictions improve with more expense data!
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.info("Add at least 2 expenses to see predictions")

            try:
                reminders_df = pd.read_csv("reminders.csv", parse_dates=["due_date"])
                upcoming_reminders = reminders_df[reminders_df["due_date"] >= pd.Timestamp.today()]

                if not upcoming_reminders.empty:
                    st.markdown(f"<h3 style='color:{title_color};'>🔔 Upcoming Payment Reminders</h3>", unsafe_allow_html=True)
                    for _, row in upcoming_reminders.iterrows():
                        st.markdown(
                            f"<div style='background-color:#ffe4ec; padding:10px; border-radius:10px; margin-bottom:10px; color:#880e4f;'>"
                            f"<strong>{row['name']}</strong> <br/>"
                            f"📅 Due on: {row['due_date'].strftime('%B %d, %Y')}"
                            f"</div>", unsafe_allow_html=True
                        )
                else:
                    st.info("No upcoming reminders!")

            except FileNotFoundError:
                st.info("No reminders saved yet.")


    elif page == "Add Expense":
        category = st.selectbox("Category", ["Food", "Transport", "Shopping", "Rent", "Utilities", "Other"])
        amount = st.number_input("Amount", min_value=0.0, format="%.2f")
        note = st.text_input("Note")
        recurring = st.checkbox("Recurring?")
        if st.button("Add Expense"):
            add_expense(category, amount, note, recurring)


    elif page == "Savings Jars":
        if "jars" not in st.session_state:
            st.session_state.jars = []

        #setting up the jar
        jar_name = st.text_input("Jar Name")
        jar_goal = st.number_input("Financial Goal", min_value=0.0, format="%.2f")
        jar_description = st.text_input("What is this jar for?")
        jar_progress = st.slider("Progress", 0, 100, 0)

        #coin drop animation fn
        def coin_drop_animation():
            coin = "🪙"
            st.markdown("<p style='font-size:24px; text-align:center;'>Saving ongoing!</p>", unsafe_allow_html=True)
            
            coin_placeholder = st.empty()
            
            for i in range(1, 11):  
                coin_placeholder.markdown(f"<p style='font-size:48px; text-align:center; padding-top:{i * 5}px;'>{coin}</p>", unsafe_allow_html=True)
                time.sleep(0.1)  

        if st.button("Add Jar"):
            #checking if all fields are filled
            if jar_name and jar_goal > 0 and jar_description:
                new_jar = {
                    "name": jar_name,
                    "goal": jar_goal,
                    "description": jar_description,
                    "progress": jar_progress
                }

                st.session_state.jars.append(new_jar)

                #success feedback
                st.success(f"Jar '{jar_name}' has been added successfully!")

                #coin drop animation after adding the jar
                coin_drop_animation()

            else:
                #warning msg if fields are missing
                st.warning("Please fill in all fields to add a jar.")

        with open("image.png", "rb") as image_file:
            jar_image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

        #tryna centre the image
        st.markdown(
            f"""
            <div style='text-align: center; margin-top: -5px;'>
                <img src='data:image/png;base64,{jar_image_base64}' width='150'>
            </div>
            """,
            unsafe_allow_html=True
        )

        for jar in st.session_state.jars:
            st.markdown(f"<p style='font-size:16px; color:black; text-align:center;'>Progress: {jar['progress']}%</p>", unsafe_allow_html=True)


    elif page == "Reminders":
        reminders = []

        num_reminders = st.number_input("How many reminders would you like to add?", min_value=1, max_value=10, value=1)

        #loop to add the reminders
        for i in range(num_reminders):
            st.subheader(f"Reminder {i+1}")

            name = st.text_input(f"Name of reminder {i+1}")
            amount = st.number_input(f"Amount for {name}", min_value=0.0, format="%.2f")
            due_date = st.date_input(f"Due date for {name}")

            #calculate the remaining days
            days_left = (due_date - datetime.now().date()).days

            if st.button(f"Add Reminder {i+1}"):
                reminders.append({
                    "name": name,
                    "amount": amount,
                    "due": due_date,
                    "days_left": days_left
                })
                st.success(f"Reminder '{name}' has been added successfully!")

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

        df = load_expenses()
        if not df.empty:
            #dropping unwanted columns if they exist
            columns_to_drop = ['due_date', 'remind_7', 'remind_3']
            df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

            st.markdown("Here’s a quick glance at your expenses this past week:")
            past_week = datetime.now() - timedelta(days=7)
            weekly_df = df[df["date"] >= past_week]

            if not weekly_df.empty:
                st.dataframe(weekly_df)
            else:
                st.info("You haven’t added any expenses in the past week.")
        else:
            st.warning("No expenses recorded yet!")
