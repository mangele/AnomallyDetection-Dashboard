import streamlit as st
import yaml
from yaml.loader import SafeLoader
import pandas as pd
import altair as alt

import streamlit_authenticator as stauth
from datetime import datetime
import numpy as np
# Function to load data
@st.cache_resource
def load_data(file_path):
    # Load data from CSV file
    data = pd.read_csv(file_path)
    return data

# Function to get unique session_ids
def get_unique_session_ids(data1, data2):
    # Find common session_ids in both datasets
    common_ids = set(data1['session_id']).intersection(set(data2['session_id']))
    common = list(common_ids)
    return common


# Function to filter data by session_id
def filter_data_by_session_id(data, session_id):
    return data[data['session_id'] == session_id]

# Function to get feature importance ranking
def get_feature_importance(data, session_id):
    filtered_data = filter_data_by_session_id(data, session_id)
    return filtered_data[['Feature', 'Importance']].sort_values(by='Importance', ascending=False)

def save_text(text, filename="file.txt"):
    with open(filename, "a") as file:  # 'a' mode appends to the file without overwriting
        file.write(text + "\n")
def get_format_time(path="./"):
    # Get the current time
    now = datetime.now()
    
    # Format the time in the desired format
    formatted_time = now.strftime("%Y%m%d_%H_%M_%S")
    
    # Create the filename
    filename = f"{path}/{st.session_state['username']}_{formatted_time}.csv"
    return filename


# Function to convert session ID to a shorter format
def shorten_session_id(session_id):
    parts = session_id.split('/')
    # Extract the relevant parts and join them
    # Adjust the indices in `parts[]` as per your requirement
    shortened = '-'.join(parts[1:4])
    return shortened

def app_function():
    st.markdown("""
      <style>
        h1 {
          margin-top: -50px;
        }
      </style>
    """, unsafe_allow_html=True)
    st.title("Feature Importance Dashboard")
 #   st.header('', divider='rainbow')

    # Load the data
    data_permutation = load_data("test/median_file.csv")
    data_kfold = load_data("test/cross-validation_median_file.csv")

    # Find common session IDs between both datasets
    common_session_ids = get_unique_session_ids(data_permutation, data_kfold)

    # Create a mapping of shortened session IDs to full session IDs
    session_id_mapping = {shorten_session_id(sid): sid for sid in common_session_ids}


    # Two-column layout
    left_col, right_col = st.columns([2, 4])  # Adjust the ratio as needed

    # Left column for the selectbox
    with left_col:
        # Create a selectbox with shortened session IDs
        selected_short_session_id = st.selectbox("Select Session ID", list(session_id_mapping.keys()))
        selected_session_id = session_id_mapping[selected_short_session_id]
#        selected_session_id = st.selectbox("Select Session ID", selected_short_session_id)
        error_code = selected_session_id.split("-")[-1]
        date = selected_session_id.split("/")[-1].split("_")[0]

    permu_importance_data = get_feature_importance(data_permutation, selected_session_id)
    kfold_importance_data = get_feature_importance(data_kfold, selected_session_id)

    # Right column for the metrics
    with right_col:
        # Custom CSS for metrics
        st.markdown("""
            <style>
            .metric {
                border-radius: 5px;
                background-color: #ffffff;
                color: rainbow;
                padding: 10px;
                text-align: center;
                box-shadow: 0 4px 8px 4px rgba(0,0,0,0.2);  /* Box shadow */
                border-left: 8px solid #5eafd4;  /* Blue left border */
            }
            .metric h3 {
                margin: 0;
				color: #5eafd4;
                font-size: 1.2em;
            }
            .metric h2 {
                margin: 0;
                font-size: 1.6em;
            }
            </style>
            """, unsafe_allow_html=True)
        # Three metrics in the right column
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.markdown(f"""
                <div class="metric">
                    <h3>Charger Model</h3>
                    <h2>RT</h2>
                </div>
                """, unsafe_allow_html=True)
        with metric_col2:
            st.markdown(f"""
                <div class="metric">
                    <h3>Error Code</h3>
                    <h2>{error_code}</h2>
                </div>
                """, unsafe_allow_html=True)
        with metric_col3:
            st.markdown(f"""
                <div class="metric">
                    <h3>CanLog Date</h3>
                    <h2>{date}</h2>
                </div>
                """, unsafe_allow_html=True)

    chart1 = alt.Chart(permu_importance_data).mark_bar().encode(
        y=alt.Y('Feature:N', sort='-x'),
        x='Importance:Q'
    ).properties(
        title='Permutation Importance',
        height=365
    )

    chart2 = alt.Chart(kfold_importance_data).mark_bar().encode(
        y=alt.Y('Feature:N', sort='-x'),
        x='Importance:Q'
    ).properties(
        title='Kfold Importance',
        height=365
    )

    # Display the charts in Streamlit using columns
    col1, col2 = st.columns([2,4])
    with col1:
        st.header('Importance', divider='rainbow')
        st.altair_chart(chart2, use_container_width=True)
#        st.header('Permutation', divider='rainbow')
        st.altair_chart(chart1, use_container_width=True)
   
    with col2:
        st.header('Exploring Signal', divider='rainbow')
        permu_features = permu_importance_data['Feature'].tolist()
        kfold_features = kfold_importance_data['Feature'].tolist()
        
        all_feature = kfold_features.copy()
        all_feature.extend(x for x in permu_features if x not in all_feature)

        feature_option = st.selectbox("Select Signal", all_feature)
	    
        board = feature_option.split("_")[0].split("-")[-1]

        if board == "ccu":
            filename = f"/home/mgl/phd/AnomallyDetection-Dashboard/canLog/{board.upper()}/{selected_session_id.split('/')[-1]}_RT-{board.upper()}.csv"
        else:
            filename = f"/home/mgl/phd/AnomallyDetection-Dashboard/canLog/{board.upper()}/{selected_session_id.split('/')[-1]}_{board.upper()}.csv"

        try:
            canLog = load_data(filename)
            st.line_chart(canLog[feature_option], height=255)
        except FileNotFoundError:
            # Number of data points
            num_points = 300

            # Generate a time array (you can adjust the range as needed)
            t = np.linspace(0, 10, num_points)

            # Generate a signal with a random frequency
            frequency = np.random.uniform(0.1, 1.0)  # Random frequency between 0.1 and 1.0
            signal = np.sin(2 * np.pi * frequency * t) + 1.5*np.random.rand(300)

            # Create a DataFrame
            canLog = pd.DataFrame({feature_option: signal})            

            st.line_chart(canLog, height=255)
	
        st.header('TroubleShooting Steps', divider="rainbow")

        c_left, c_right = st.columns([2,4])
        with c_left:
            best_method = options = st.multiselect(
                'Best Method',
                ['None', 'Permutation', 'Kfold'])
        with c_right:
            best_signals = st.multiselect(
                'Best Signal', ['None'] + all_feature 
                )
        user_input = st.text_area("Enter your text here", height=150) 
        if st.button("Save Text"):
            save_text(f"{st.session_state}, {selected_session_id}, {best_method}, {best_signals},  {user_input}", get_format_time())
            st.success("Text saved successfully!")


# Main app
st.set_page_config(layout="wide", page_icon="logo.png" )

with open('./configAuth.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)
authenticator =  stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)
#authenticator._implement_logout()
name, authentication_status, username = authenticator.login("main")


if authentication_status == False:
    st.error("Username/password is incorrect")
if authentication_status == None:
    st.error("Please entere your username and password")
if authentication_status:
    # ---- SIDEBAR ----
    # First, display the welcome message
    authenticator.logout("Logout", "sidebar")

    # Add some space between the welcome message and the logout button
    st.sidebar.write("\n\n\n")

    # Place the logout button
    st.sidebar.title(f"Hi {name.split(' ')[0]}")

    # Continue with the rest of the app
    app_function() 
