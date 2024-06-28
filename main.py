import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ctransformers import AutoModelForCausalLM

# Load the Mistral 7B model
model_path = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
model = AutoModelForCausalLM.from_pretrained(model_path, model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf", model_type="mistral", gpu_layers=0, context_length=2048)

# Set page title
st.title("Server Resource Usage and Log Analysis with Mistral 7B")

# Sidebar for user input
st.sidebar.header("Upload Files")
uploaded_file = st.sidebar.file_uploader("Choose a log file", type=["log", "txt"])
uploaded_csv = st.sidebar.file_uploader("Choose a CSV file for resource usage", type=["csv"])

# Process uploaded log file
if uploaded_file is not None:
    log_data = uploaded_file.read().decode("utf-8")
    st.subheader("Log File Content (First 1000 characters)")
    st.text(log_data[:1000])  # Display only the first 1000 characters

# Process uploaded CSV file
if uploaded_csv is not None:
    resource_data = pd.read_csv(uploaded_csv)
    st.subheader("Resource Usage Data (First 5 rows)")
    st.write(resource_data.head())  # Display only the first 5 rows

    # Display basic statistics
    st.subheader("Basic Statistics")
    st.write(resource_data.describe())

    # Plot resource usage
    st.subheader("Resource Usage Over Time")
    fig, ax = plt.subplots()
    sns.lineplot(data=resource_data, ax=ax)
    st.pyplot(fig)


# Load the Mistral 7B model
model_path = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
model = AutoModelForCausalLM.from_pretrained(model_path, model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
                                             model_type="mistral", gpu_layers=0, context_length=2048)

def analyze_logs_and_resources(logs, resources):
    # Prepare a prompt for the LLM
    log_sample = logs[:200]  # Reduce log sample to 200 characters
    resource_summary = resources.describe().to_string()
    resource_summary_lines = resource_summary.split('\n')[:5]  # Take only the first 5 lines of the summary
    resource_summary_truncated = '\n'.join(resource_summary_lines)

    prompt = f"""Analyze the following server logs and resource usage data to recommend cost-saving measures and identify issues. Provide a concise analysis, with specific recommendations.

Logs (sample):
{log_sample}

Resource Usage Data (summary):
{resource_summary_truncated}

Analysis and Recommendations:
"""

    # Generate the analysis using Mistral 7B
    result = model(prompt, max_new_tokens=150, temperature=0.7)
    return result


# Analysis and recommendations
if st.button("Analyze"):
    st.subheader("Analysis and Recommendations")
    if uploaded_file is not None and uploaded_csv is not None:
        analysis = analyze_logs_and_resources(log_data, resource_data)
        st.write(analysis)
    else:
        st.write("Please upload both log file and resource usage CSV file.")