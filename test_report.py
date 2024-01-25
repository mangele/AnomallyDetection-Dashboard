import streamlit as st
import yaml
from yaml.loader import SafeLoader
from pathlib import Path
import pickle
import pandas as pd
import altair as alt
import numpy as np
import dask.dataframe as dd

import streamlit_authenticator as stauth
from datetime import datetime

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
    return list(common_ids)

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
def get_format_time(path="/home/miguel/Data%20Analytics/da/canAnalyser/results"):
    # Get the current time
    now = datetime.now()
    
    # Format the time in the desired format
    formatted_time = now.strftime("%Y%m%d_%H_%M_%S")
    
    # Create the filename
    filename = f"{path}/{st.session_state['username']}_{formatted_time}.csv"
    return filename

@st.cache_resource
def read_and_filter_dask(file_path, session_id, signal):
    # Specify data types for columns where Dask's inference fails
    dtype = {
        'rt-ccu_controlmaster': 'object',
        'rt-ccu_currentchargetype': 'object',
        'rt-ccu_currentlimitflags': 'object',
        'rt-ccu_dynamicpowerlimit': 'object',
        'rt-ccu_powerlimitflags': 'object',
        'rt-ccu_saeresponcecode': 'object',
        'rt-ccu_voltagelimitflags': 'object'
    }

    # Read the CSV file using Dask with specified data types
    ddf = dd.read_csv(file_path, dtype=dtype)
    # Filter the DataFrame based on session_id and keep it as a DataFrame
    filtered_ddf = ddf[ddf['filename'] == session_id][[signal, "ticks"]]

    # Sort the filtered DataFrame by 'ticks'
    sorted_ddf = filtered_ddf.sort_values('ticks')  # Sorting by a single column

    # Compute the Dask DataFrame to get a Pandas DataFrame
    return sorted_ddf.compute()


def get_board_csvfile(board):
	files = {
			"ccu" : "/mnt/c/Users/miguel.retamozo/OneDrive - Tritium Pty Ltd/Documents/DA_2020_dataset/all2020.csv" ,
			"wiso" : "/mnt/c/Users/miguel.retamozo/OneDrive - Tritium Pty Ltd/Documents/DA_2020_dataset/data/iso_2020_after.csv",
			"ts" : "/mnt/c/Users/miguel.retamozo/OneDrive - Tritium Pty Ltd/Documents/DA_2020_dataset/data/ts_2020_after.csv",
			"wsgt" : "/mnt/c/Users/miguel.retamozo/OneDrive - Tritium Pty Ltd/Documents/DA_2020_dataset/data/wsgt_2020_after.csv",
			"q7" : "/mnt/c/Users/miguel.retamozo/OneDrive - Tritium Pty Ltd/Documents/DA_2020_dataset/data/q7_2020.csv",
			"fp" : "/mnt/c/Users/miguel.retamozo/OneDrive - Tritium Pty Ltd/Documents/DA_2020_dataset/data/fp_2020.csv",
			}
	return files[board]



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


    # Two-column layout
    left_col, right_col = st.columns([2, 4])  # Adjust the ratio as needed

    # Left column for the selectbox
    with left_col:
        selected_session_id = st.selectbox("Select Session ID", common_session_ids)
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

#------------
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
            filename = f"/home/miguel/Data%20Analytics/da/canLog/{board.upper()}/{selected_session_id.split('/')[-1]}_RT-{board.upper()}.csv"
        else:
            filename = f"/home/miguel/Data%20Analytics/da/canLog/{board.upper()}/{selected_session_id.split('/')[-1]}_{board.upper()}.csv"

        canLog = load_data(filename)
	
        st.line_chart(canLog[feature_option], height=255)

        st.header('TroubleShooting Steps', divider="rainbow")

        c_left, c_right = st.columns([2,4])
        with c_left:
            best_method = options = st.multiselect(
                'Best Method',
                ['None', 'Permutation', 'Kfold'])
#            st.write(permu_importance_data['Feature'].iloc[:])
        with c_right:
            best_signals = st.multiselect(
                'Best Signal', ['None'] + all_feature 
                )
        user_input = st.text_area("Enter your text here", height=150) 
        if st.button("Save Text"):
            save_text(f"{st.session_state}, {selected_session_id}, {best_method}, {best_signals},  {user_input}", get_format_time())
            st.success("Text saved successfully!")

print("Session state on refresh:", st.session_state)

# Main app
st.set_page_config(layout="wide", page_icon="/home/miguel/Data%20Analytics/da/canAnalyser/icon.png" )

with open('../configAuth.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)
authenticator =  stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)
#authenticator._implement_logout()
#print(f" MAIN before authentication -----{authenticator.credentials}-----")
name, authentication_status, username = authenticator.login("main")

#print(f" MAIN after login authentication -----{authenticator.credentials}-----")

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
    st.sidebar.title(f"   Hi {name.split(' ')[0]}")

    # Continue with the rest of the app
    app_function() 
    #st.text(f"2-----{authenticator.credentials}-----")

