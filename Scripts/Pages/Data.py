# import pandas as pd
# import plotly.express as px
# import streamlit as st

# # Constants
# BASE_PATH = "C:\Adarsh work\Dissonance\\"

# # Emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
# st.set_page_config(page_title="Dashboard", page_icon=":hot_springs:", layout="wide")

# data = pd.read_csv(BASE_PATH + "Data\Fused_cleaned_dataset.csv")

# # SIDEBAR
# st.sidebar.header("Apply Filters:")

# fault = st.sidebar.multiselect("Select Fault type", 
#                                options=data["Fault"].unique(),
#                                default=data["Fault"].unique())

# data_selection = data.query("Fault == @fault")

# # MAIN PAGE
# st.title(":bar_chart: Data Dashboard")

# with st.expander("To view the complete data. Click here :)"):
#     st.write("This is the complete data.")
#     st.dataframe(data_selection)

# # Create two columns
# col1, col2 = st.columns(2)

# # Plot the distribution of classes using a bar chart in the first column
# class_distribution = data_selection['Fault'].value_counts()
# fig_class_distribution = px.bar(class_distribution, x=class_distribution.index, y=class_distribution.values,
#                                 labels={'x': 'Fault', 'y': 'Count'}, title='Class Distribution')
# col1.plotly_chart(fig_class_distribution)

# # You can add more plots based on your requirements in the second column
# # Example 1: Scatter plot
# fig_scatter = px.scatter(data_selection, x='AirIn', y='WaterIn', color='Fault', title='Scatter Plot')
# col2.plotly_chart(fig_scatter)

# # Example 2: Histogram
# fig_histogram = px.histogram(data_selection, x='Air.T', color='Fault', title='Temperature Histogram')
# col2.plotly_chart(fig_histogram)

# # Example 3: Box plot
# fig_box_plot = px.box(data_selection, x='Fault', y='Water.level', points='all', title='Box Plot')
# col2.plotly_chart(fig_box_plot)

# # Example 4: Line chart
# fig_line_chart = px.line(data_selection, x=data_selection.index, y='AirIn', color='Fault', title='AirIn Over Time')
# col1.plotly_chart(fig_line_chart)

# # Example 5: Violin plot
# fig_violin = px.violin(data_selection, y='Water.T', x='Fault', box=True, points='all', title='Violin Plot')
# col2.plotly_chart(fig_violin)

# # Example 6: 3D Scatter plot
# fig_3d_scatter = px.scatter_3d(data_selection, x='AirIn', y='WaterIn', z='Air.T', color='Fault', title='3D Scatter Plot')
# col1.plotly_chart(fig_3d_scatter)

# # Example 7: Pair plot
# fig_pair_plot = px.scatter_matrix(data_selection, dimensions=['AirIn', 'WaterIn', 'Air.T', 'Water.T'], color='Fault', title='Pair Plot')
# col1.plotly_chart(fig_pair_plot)


import pandas as pd
import plotly.express as px
import streamlit as st

# Constants
BASE_PATH = "C:\Adarsh work\Dissonance\\"

# Emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="Dashboard", page_icon=":hot_springs:", layout="wide")

# Load data with st.cache_data
@st.cache_data(ttl=600)  # Set a time-to-live (TTL) of 10 minutes
def load_data():
    return pd.read_csv(BASE_PATH + "Data\Fused_cleaned_dataset.csv")

data = load_data()

# SIDEBAR
st.sidebar.header("Apply Filters:")

fault = st.sidebar.multiselect("Select Fault type", 
                               options=data["Fault"].unique(),
                               default=data["Fault"].unique())

data_selection = data.query("Fault == @fault")

# MAIN PAGE
st.title(":bar_chart: Data Dashboard")

with st.expander("To view the complete data. Click here :)"):
    st.write("This is the complete data.")
    st.dataframe(data_selection)

# Create two columns
col1, col2 = st.columns(2)

# Plot the distribution of classes using a bar chart in the first column
class_distribution = data_selection['Fault'].value_counts()
fig_class_distribution = px.bar(class_distribution, x=class_distribution.index, y=class_distribution.values,
                                labels={'x': 'Fault', 'y': 'Count'}, title='Class Distribution')
col1.plotly_chart(fig_class_distribution)

# You can add more plots based on your requirements in the second column
# Example 1: Scatter plot
@st.cache_data(ttl=600)  # Set a time-to-live (TTL) of 10 minutes
def scatter_plot():
    return px.scatter(data_selection, x='AirIn', y='WaterIn', color='Fault', title='Scatter Plot')

fig_scatter = scatter_plot()
col2.plotly_chart(fig_scatter)

# Example 2: Histogram
@st.cache_data(ttl=600)  # Set a time-to-live (TTL) of 10 minutes
def histogram_plot():
    return px.histogram(data_selection, x='Air.T', color='Fault', title='Temperature Histogram')

fig_histogram = histogram_plot()
col2.plotly_chart(fig_histogram)

# Example 3: Box plot
@st.cache_data(ttl=600)  # Set a time-to-live (TTL) of 10 minutes
def box_plot():
    return px.box(data_selection, x='Fault', y='Water.level', points='all', title='Box Plot')

fig_box_plot = box_plot()
col2.plotly_chart(fig_box_plot)

# Example 4: Line chart
@st.cache_data(ttl=600)  # Set a time-to-live (TTL) of 10 minutes
def line_chart():
    return px.line(data_selection, x=data_selection.index, y='AirIn', color='Fault', title='AirIn Over Time')

fig_line_chart = line_chart()
col1.plotly_chart(fig_line_chart)

# Example 5: Violin plot
@st.cache_data(ttl=600)  # Set a time-to-live (TTL) of 10 minutes
def violin_plot():
    return px.violin(data_selection, y='Water.T', x='Fault', box=True, points='all', title='Violin Plot')

fig_violin = violin_plot()
col2.plotly_chart(fig_violin)

# Example 6: 3D Scatter plot
@st.cache_data(ttl=600)  # Set a time-to-live (TTL) of 10 minutes
def scatter_3d_plot():
    return px.scatter_3d(data_selection, x='AirIn', y='WaterIn', z='Air.T', color='Fault', title='3D Scatter Plot')

fig_3d_scatter = scatter_3d_plot()
col1.plotly_chart(fig_3d_scatter)

# Example 7: Pair plot
@st.cache_data(ttl=600)  # Set a time-to-live (TTL) of 10 minutes
def pair_plot():
    return px.scatter_matrix(data_selection, dimensions=['AirIn', 'WaterIn', 'Air.T', 'Water.T'], color='Fault', title='Pair Plot')

fig_pair_plot = pair_plot()
col1.plotly_chart(fig_pair_plot)
