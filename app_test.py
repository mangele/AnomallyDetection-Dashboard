import streamlit as st
import altair as alt
import pandas as pd

# Sample data and chart for demonstration
data = pd.DataFrame({'x': range(10), 'y': range(10)})
chart1 = alt.Chart(data).mark_line().encode(x='x', y='y')

def main():
    # Custom CSS for the metric class
    st.markdown("""
        <style>
        .metric-header {
            border-radius: 10px 10px 0 0;
            background-color: #ffffff;
            color: blue;
            padding: 15px;
            text-align: center;
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        }
        .chart-container {
            padding: 15px;
            padding-top: 0;
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        }
        </style>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
                <div class="metric-header">
                    <h3>Charger Model</h3>
                    <h2>RT</h2>
                </div>
                """, unsafe_allow_html=True)
        st.markdown(f"""
                <div class="metric-header">
                    <h3>Permutation</h3>
                </div>
                """, unsafe_allow_html=True)
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.altair_chart(chart1, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

