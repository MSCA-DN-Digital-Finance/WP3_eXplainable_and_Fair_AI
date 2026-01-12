import streamlit as st
import pandas as pd
import json, pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

@st.cache_data(show_spinner=False)
def load_json(rel_path: str):
    path = BASE_DIR / rel_path
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_pickle(rel_path: str):
    path = BASE_DIR / rel_path
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data(show_spinner=False)
def load_parquet(rel_path: str):
    path = BASE_DIR / rel_path
    return pd.read_parquet(path)
