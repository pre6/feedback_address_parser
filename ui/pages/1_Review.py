"""Review queue: edit many parsed addresses in one table, save them all at once.

Streamlit shows this as a second page automatically because it lives in ui/pages/.
"""
import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
FIELDS = ["address", "city", "zip"]

st.set_page_config(page_title="Review queue", layout="wide")
st.title("Review queue")

with st.sidebar:
    reviewer = st.text_input("Your name", value="analyst")
    status = st.selectbox("Show", ["needs_review", "auto", "reviewed", "all"])
    page_size = st.selectbox("Rows per page", [25, 50, 100, 200], index=1)
    page = st.number_input("Page", min_value=1, value=1, step=1)
    if st.button("Refresh"):
        st.session_state.pop("df", None)

st.session_state.setdefault("ver", 0)
params = (status, page_size, page)


def load():
    r = requests.get(
        f"{API_URL}/review",
        params={"status": status, "limit": page_size, "offset": (page - 1) * page_size},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data["rows"], columns=[
        "source_id", "location", "address", "city", "zip", "status", "min_conf", "flags", "model_version"])
    df["flags"] = df["flags"].apply(lambda f: ", ".join(f) if isinstance(f, list) else "")
    for c in FIELDS:
        df[c] = df[c].fillna("")
    df.insert(0, "approve", False)
    st.session_state.df, st.session_state.total, st.session_state.params = df, data["total"], params
    st.session_state.ver += 1  # resets the editor's pending edits


if "df" not in st.session_state or st.session_state.get("params") != params:
    try:
        load()
    except requests.RequestException as e:
        st.error(f"Could not load the queue from the API: {e}")
        st.stop()

if st.session_state.get("flash"):
    st.success(st.session_state.pop("flash"))

df = st.session_state.df
st.caption(f"{st.session_state.total} rows with status '{status}'. Showing least confident first.")

if df.empty:
    st.info("Nothing here.")
    st.stop()

edited = st.data_editor(
    df,
    key=f"editor_{st.session_state.ver}",
    hide_index=True,
    disabled=["source_id", "location", "status", "min_conf", "flags", "model_version"],
    column_config={
        "approve": st.column_config.CheckboxColumn("OK as-is", help="Tick if the parse is already correct"),
        "min_conf": st.column_config.NumberColumn("Conf", format="%.3f"),
    },
)

edit_cols = edited[FIELDS].fillna("").astype(str).apply(lambda s: s.str.strip())
orig_cols = df[FIELDS].fillna("").astype(str).apply(lambda s: s.str.strip())
changed = (edit_cols != orig_cols).any(axis=1) | edited["approve"]
n = int(changed.sum())

if st.button(f"Save {n} row(s)", type="primary", disabled=n == 0):
    updates = [
        {"source_id": edited.loc[i, "source_id"], **{c: edit_cols.loc[i, c] for c in FIELDS}}
        for i in edited.index[changed]
    ]
    try:
        r = requests.post(f"{API_URL}/review/batch", json={"reviewer": reviewer, "updates": updates}, timeout=60)
        r.raise_for_status()
        res = r.json()
        st.session_state.flash = f"Saved {res['updated']} row(s)." + (
            f" Not found: {res['not_found']}" if res["not_found"] else "")
        st.session_state.pop("df", None)
        st.rerun()
    except requests.RequestException as e:
        st.error(f"Save failed: {e}")
