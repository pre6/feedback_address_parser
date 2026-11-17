"""Address parser UI.

Run locally (API must already be running):
    streamlit run ui/app.py

In Docker, set API_URL=http://api:8000
"""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
FIELDS = {"address": "ADDRESS", "city": "CITY", "zip": "ZIP"}

st.set_page_config(page_title="Address Parser", page_icon="📍")
st.title("Address Parser")

# --- state ---------------------------------------------------------------- #
for key in ("parsed", "model_version", "last_text", "status"):
    st.session_state.setdefault(key, None)
for key in FIELDS:
    st.session_state.setdefault(f"f_{key}", "")


def run_parse(text: str):
    try:
        r = requests.post(f"{API_URL}/parse", json={"text": text}, timeout=10)
        r.raise_for_status()
    except requests.RequestException as e:
        st.error(f"Could not reach the API: {e}")
        return
    data = r.json()
    st.session_state.parsed = data
    st.session_state.model_version = data["model_version"]
    st.session_state.last_text = text
    st.session_state.status = None
    # prefill the editable fields with the first entity of each label
    for key, label in FIELDS.items():
        found = [e["text"] for e in data["entities"] if e["label"] == label]
        st.session_state[f"f_{key}"] = found[0] if found else ""


def send_correction():
    payload = {
        "text": st.session_state.last_text,
        "address": st.session_state.f_address,
        "city": st.session_state.f_city,
        "zip": st.session_state.f_zip,
    }
    try:
        r = requests.post(f"{API_URL}/corrections", json=payload, timeout=10)
        r.raise_for_status()
        st.session_state.status = "Correction saved. Thank you!"
    except requests.RequestException as e:
        st.session_state.status = f"Failed to save correction: {e}"


# --- input ---------------------------------------------------------------- #
text = st.text_input("Location string", placeholder="123 Main St, Springfield, IL 62704")
if st.button("Parse", type="primary") and text.strip():
    run_parse(text)

# --- results + correction form -------------------------------------------- #
if st.session_state.parsed:
    st.caption(f"Model: {st.session_state.model_version}")
    st.subheader("Parsed fields (edit if wrong)")
    st.text_input("Address", key="f_address")
    st.text_input("City", key="f_city")
    st.text_input("ZIP", key="f_zip")
    st.button("Submit correction", on_click=send_correction)

    with st.expander("Raw model output"):
        st.json(st.session_state.parsed)

if st.session_state.status:
    st.info(st.session_state.status)
