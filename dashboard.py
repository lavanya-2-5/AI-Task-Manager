import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from smart_assigner import auto_assign_tasks, reset_assignments
import os
from datetime import datetime

st.set_page_config(page_title="AI Task Manager Dashboard", layout="wide")
st.title("📊 AI-Powered Task Management Dashboard")

# Load data
if os.path.exists("model_predictions.csv"):
    df = pd.read_csv("model_predictions.csv")
else:
    st.error("model_predictions.csv file not found.")
    st.stop()

# --- Deadline Urgency Tag ---
def get_deadline_status(deadline):
    try:
        deadline_date = datetime.strptime(deadline, "%Y-%m-%d")
        days_remaining = (deadline_date - datetime.today()).days
        if days_remaining < 0:
            return "Overdue"
        elif days_remaining <= 2:
            return "Due Soon"
        else:
            return "On Time"
    except:
        return "Invalid"

df["Deadline Status"] = df["Deadline"].apply(get_deadline_status)

# --- Current Tasks Table at Top ---
st.subheader("📌 Current Task Assignments")
# --- Current Tasks Table at Top ---
st.subheader("📌 Current Task Assignments")

# Sirf selected columns hi show karna
show_columns = ['TaskID', 'Description', 'Deadline', 'AssignedTo', 'Priority', 'Deadline Status']
st.dataframe(df[show_columns], use_container_width=True)

# --- Action Buttons Right Below Current Tasks ---
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
with col_btn1:
    if st.button("🚀 Auto Assign Tasks"):
        auto_assign_tasks()
        st.success("✅ Tasks auto-assigned successfully.")
        st.rerun()

with col_btn2:
    if st.button("🔄 Reset Assignments"):
        reset_assignments()
        st.success("🗑️ Assignments reset successfully.")
        st.rerun()

with col_btn3:
    st.download_button("⬇️ Download CSV", df.to_csv(index=False), "assigned_tasks.csv", "text/csv")

# --- Task Summary ---
st.markdown("### 📋 Task Summary")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Tasks", len(df))
with col2:
    overdue = len(df[df['Deadline Status'] == "Overdue"])
    st.metric("Overdue Tasks", overdue)
with col3:
    unassigned = df['AssignedTo'].isna().sum()
    st.metric("Unassigned Tasks", unassigned)

# --- Priority Distribution Pie Chart ---
st.markdown("### 🥇 Task Priority Distribution")
if 'Priority' in df.columns:
    priority_counts = df['Priority'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(priority_counts, labels=priority_counts.index, autopct='%1.1f%%')
    ax1.axis('equal')
    st.pyplot(fig1)
else:
    st.warning("Priority column not found.")

# --- Workload Distribution Chart ---
st.markdown("### 👥 Workload Distribution")
if 'AssignedTo' in df.columns:
    assigned_counts = df['AssignedTo'].value_counts()
    fig2, ax2 = plt.subplots()
    ax2.bar(assigned_counts.index, assigned_counts.values)
    ax2.set_ylabel("Number of Tasks")
    st.pyplot(fig2)

# --- Filters ---
st.markdown("### 🔍 Filter Tasks")
with st.expander("Click to Filter"):
    members = df['AssignedTo'].dropna().unique().tolist()
    priorities = df['Priority'].dropna().unique().tolist()

    selected_member = st.selectbox("Filter by Member", ["All"] + members)
    selected_priority = st.selectbox("Filter by Priority", ["All"] + priorities)

    filtered_df = df.copy()
    if selected_member != "All":
        filtered_df = filtered_df[filtered_df['AssignedTo'] == selected_member]
    if selected_priority != "All":
        filtered_df = filtered_df[filtered_df['Priority'] == selected_priority]

    st.dataframe(filtered_df, use_container_width=True)

# --- Task Completion Tracker ---
st.markdown("### ✅ Task Completion Tracker")
df['Completed'] = False
completed_tasks = st.multiselect("Select Completed Tasks", df['TaskID'].tolist())
df['Completed'] = df['TaskID'].isin(completed_tasks)

# --- Model Accuracy Placeholder ---
st.markdown("### 📈 Model Accuracy")

if 'Priority' in df.columns and 'predicted_priority' in df.columns:
    correct_preds = (df['Priority'] == df['predicted_priority']).sum()
    total_preds = df['predicted_priority'].notna().sum()
    if total_preds > 0:
        accuracy = correct_preds / total_preds * 100
        st.success(f"Prediction Accuracy: {accuracy:.2f}%")
    else:
        st.info("📌 No predictions available to evaluate accuracy.")
else:
    st.info("ℹ️ Model not applied yet.")