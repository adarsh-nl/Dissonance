import pandas as pd
import plotly.express as px
import streamlit as st

# Constants
BASE_PATH = "C:\Adarsh work\Dissonance\\"

# Emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="Dashboard", page_icon=":hot_springs:", layout="wide")

# Main Page Introduction
st.title("Welcome to the Federated Learning Dashboard!")
st.write("""
This dashboard provides insights and visualizations related to the Dissonance algorithm.

Feel free to explore the different sections of the dashboard to understand the dataset, visualize data, or run the algorithm demo.

Navigate to the other pages using the sidebar on the left.
""")