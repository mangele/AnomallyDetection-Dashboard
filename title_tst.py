import streamlit as st

st.set_page_config(layout="wide")

padding_top = 0

#st.markdown(f"""
#    <style>
#        .reportview-container .main .block-container{{
#            padding-top: {padding_top}rem;
#        }}
#    </style>""",
#    unsafe_allow_html=True,
#)

st.title("My Next Best Actions")

st.sidebar.text_input("foo")
