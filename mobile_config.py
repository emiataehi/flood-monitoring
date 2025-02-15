import streamlit as st

def set_mobile_responsive():
    """Configure mobile-responsive layout settings"""
    
    # CSS to handle mobile responsiveness
    mobile_css = """
        <style>
        /* Mobile-specific styles */
        @media screen and (max-width: 768px) {
            /* Make metric cards stack on mobile */
            [data-testid="stMetricValue"] {
                width: 100% !important;
            }
            
            /* Adjust chart sizes for mobile */
            [data-testid="stPlotlyChart"] {
                width: 100% !important;
                height: 300px !important;
            }
            
            /* Improve table scrolling on mobile */
            .stDataFrame {
                width: 100% !important;
                overflow-x: auto !important;
            }
            
            /* Better spacing for mobile */
            .block-container {
                padding-top: 1rem !important;
                padding-bottom: 1rem !important;
            }
            
            /* Make text more readable on mobile */
            p, li {
                font-size: 16px !important;
                line-height: 1.6 !important;
            }
            
            /* Improve button accessibility on mobile */
            .stButton > button {
                width: 100% !important;
                margin: 0.5rem 0 !important;
                padding: 0.5rem !important;
            }
        }
        </style>
    """
    
    # Inject CSS
    st.markdown(mobile_css, unsafe_allow_html=True)

def optimize_layout():
    """Configure layout optimization for mobile"""
    
    # Set page config for mobile
    st.set_page_config(
        page_title="Flood Monitoring",
        layout="wide",
        initial_sidebar_state="collapsed"  # Collapse sidebar by default on mobile
    )
    
    # Hide unnecessary elements on mobile
    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)