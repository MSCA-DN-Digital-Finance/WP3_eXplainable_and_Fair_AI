import streamlit as st

# ---------------------------------------------------------
# PAGE CONFIG (call from main app only)
# ---------------------------------------------------------
def set_page_config():
    st.set_page_config(
        page_title="TE–Carbon Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )



# ---------------------------------------------------------
# GLOBAL STYLES
# ---------------------------------------------------------
def apply_global_styles():

    st.markdown(
    """
    <style>
    div.block-container {
        padding-top: 1.5rem !important;
    }

    h1 {
        margin-top: 0.2rem !important;
        margin-bottom: 0.6rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


    st.markdown(
        """
        <style>
        /* -----------------------------------
           FIX SIDEBAR WIDTH ONLY WHEN OPEN
        ----------------------------------- */

        section[data-testid="stSidebar"][aria-expanded="true"] {
            min-width: 200px !important;
            max-width: 200px !important;
            width: 200px !important;
        }

        /* Prevent drag resize handle */
        section[data-testid="stSidebar"] > div {
            resize: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # (keep your other styles exactly as they are)

    st.markdown(
    """
    <style>
    /* Narrow selectbox width */
    div[data-baseweb="select"] {
        max-width: 230px;   /* tweak if needed */
    }
    </style>
    """,
    unsafe_allow_html=True
)


    st.markdown("""
    <style>
    div.block-container {
        max-width: 1500px;
        padding-left: 3rem;
        padding-right: 3rem;
        margin-left: auto;
        margin-right: auto;
    }

    .section-title {
        font-size: 22px;
        font-weight: 700;
        padding: 10px 18px;
        background-color: #111;
        border-left: 4px solid #6A5AE0;
        border-radius: 6px;
        margin-top: 30px;
        margin-bottom: 10px;
    }

    .metric-card {
        border: 1px solid #555;
        border-radius: 8px;
        padding: 10px 12px;
        margin: 4px;
        background-color: #1e1e1e;
    }

    .metric-label {
        font-size: 13px;
        font-weight: 500;
        color: #ddd;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .metric-value {
        font-size: 24px;
        font-weight: 700;
        margin-top: 4px;
        color: white;
    }

    .delta-pos { color: #00d26a; font-size: 14px; }
    .delta-neg { color: #ff4b4b; font-size: 14px; }

    /* Selectbox glow */
    div[data-baseweb="select"] span {
        color: #F2F2F2 !important;
        font-weight: 300 !important;
        text-shadow: 0 0 12px rgba(255,255,255,0.45) !important;
    }

    /* Radio glow */
    div[role="radio"][aria-checked="true"] ~ label {
        color: #F2F2F2 !important;
        font-weight: 300 !important;
        text-shadow: 0 0 12px rgba(255,255,255,0.45) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
<style>
/* Tooltip styles (GLOBAL) */
.tooltip-icon {
    background-color: #d0d0d0;
    color: #333;
    border-radius: 50%;
    width: 15px;
    height: 15px;
    font-size: 10px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    cursor: help;
    position: relative;
    flex: 0 0 auto;
}
.tooltip-icon:hover .tooltip-text {
    visibility: visible;
    opacity: 1;
}
.tooltip-text {
    visibility: hidden;
    opacity: 0;
    width: 220px;
    background-color: #555;
    color: #fff;
    text-align: left;
    border-radius: 6px;
    padding: 8px;
    position: absolute;
    z-index: 9999;
    bottom: 125%;
    left: 50%;
    margin-left: -110px;
    font-size: 12px;
    line-height: 1.3;
    transition: opacity 0.25s ease-in-out;
}
</style>
""", unsafe_allow_html=True)
