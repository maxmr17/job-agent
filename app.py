import base64
import datetime
import hashlib
import html
import json
import re
from io import BytesIO
from urllib.parse import quote_plus
import streamlit as st
import pandas as pd

from agent import run_agent, generate_job_search_profile, parse_and_score_jobs
from tools import (
    extract_job_from_url,
    extract_text_from_pdf,
    extract_pdf_style,
    export_cover_letter_pdf,
    export_resume_pdf,
    search_linkedin_jobs,
)
import database as db
import auth as auth_module

db.init_db()

st.set_page_config(
    page_title="JobAgent AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS  — design system
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

/* ── Reset & base ──────────────────────────────────────────────────────── */
html, body, [class*="css"], * {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
/* !! Restore Material Symbols Rounded for Streamlit icon spans — our * rule above would
   override the icon font, causing icon-name text ("keyboard_arrow_down" etc.) to render
   as literal Inter text instead of the glyph. These selectors target exactly the icon spans
   Streamlit generates via DynamicIcon. */
[data-testid="stIconMaterial"],
[data-testid="stExpanderIcon"],
[data-testid="stExpanderIconCheck"],
[data-testid="stExpanderIconError"],
[data-testid="stExpanderIconSpinner"],
span[translate="no"] {
    font-family: 'Material Symbols Rounded' !important;
    font-feature-settings: 'liga' !important;
    -webkit-font-feature-settings: 'liga' !important;
    font-size: inherit !important;
}
#MainMenu, footer, header { visibility: hidden !important; }
.stApp { background: #f1f5f9 !important; }
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
    max-width: 1120px !important;
}

/* ── Sidebar ───────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] > div > div,
section[data-testid="stSidebar"] > div > div > div,
section[data-testid="stSidebar"] > div > div > div > div,
section[data-testid="stSidebar"] > div > div > div > div > div,
[data-testid="stSidebarContent"],
[data-testid="stSidebarHeader"],
[data-testid="stSidebarUserContent"] {
    background: #0f172a !important;
    background-color: #0f172a !important;
}
section[data-testid="stSidebar"] {
    border-right: none !important;
    box-shadow: 1px 0 0 rgba(255,255,255,0.04) !important;
}
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label { color: #94a3b8 !important; }

/* ── Sidebar nav buttons (inactive) ───────────────────────────────────────── */
section[data-testid="stSidebar"] div.stButton > button {
    background: transparent !important;
    color: #94a3b8 !important;
    border: none !important;
    border-left: 2px solid transparent !important;
    border-radius: 8px !important;
    text-align: center !important;
    justify-content: center !important;
    font-weight: 500 !important;
    font-size: 13.5px !important;
    padding: 9px 16px !important;
    margin: 1px 0 !important;
    box-shadow: none !important;
    width: 100% !important;
    letter-spacing: normal !important;
    transform: none !important;
    line-height: 1.4 !important;
}
section[data-testid="stSidebar"] div.stButton > button:hover {
    background: rgba(79,70,229,0.12) !important;
    color: #a5b4fc !important;
    border-left-color: rgba(99,102,241,0.5) !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Sidebar nav active item (rendered as markdown div) ───────────────────── */
.sidebar-nav-active {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 9px 16px;
    margin: 1px 0;
    border-radius: 8px;
    font-size: 13.5px;
    font-weight: 600;
    color: #a5b4fc;
    background: rgba(79,70,229,0.18);
    box-shadow: inset 2px 0 0 #6366f1;
}

/* ── Sign Out button override (also in sidebar, should look distinct) ────── */
section[data-testid="stSidebar"] div.stButton > button[data-testid="stBaseButton-secondary"]:last-of-type {
    color: #64748b !important;
    font-size: 13px !important;
    justify-content: center !important;
    border-top: 1px solid rgba(255,255,255,0.06) !important;
    border-left: none !important;
    border-radius: 6px !important;
    margin-top: 4px !important;
}
section[data-testid="stSidebar"] div.stButton > button[data-testid="stBaseButton-secondary"]:last-of-type:hover {
    color: #f87171 !important;
    background: rgba(248,113,113,0.08) !important;
    border-left: none !important;
}

/* ── Sidebar expand button (shown when sidebar is collapsed) ──────────────── */
button[data-testid="stExpandSidebarButton"] {
    background: #1e293b !important;
    color: #a5b4fc !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 0 8px 8px 0 !important;
    box-shadow: 2px 0 8px rgba(0,0,0,0.3) !important;
    opacity: 1 !important;
    visibility: visible !important;
    pointer-events: all !important;
}
button[data-testid="stExpandSidebarButton"]:hover {
    background: #334155 !important;
    color: #c7d2fe !important;
}

/* ── Buttons ───────────────────────────────────────────────────────────── */
div.stButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 0.65rem 1.8rem !important;
    width: 100% !important;
    box-shadow: 0 2px 8px rgba(79,70,229,0.28), 0 1px 2px rgba(0,0,0,0.08) !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.01em !important;
    position: relative !important;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #4338ca 0%, #4f46e5 100%) !important;
    box-shadow: 0 4px 14px rgba(79,70,229,0.4), 0 2px 4px rgba(0,0,0,0.1) !important;
    transform: translateY(-1px) !important;
}
div.stButton > button:active { transform: translateY(0) !important; }

/* ── Download button ───────────────────────────────────────────────────── */
div.stDownloadButton > button {
    background: #fff !important;
    color: #334155 !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    width: auto !important;
    padding: 0.45rem 1.2rem !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
    transition: all 0.15s !important;
}
div.stDownloadButton > button:hover {
    background: #f8fafc !important;
    border-color: #cbd5e1 !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.07) !important;
    transform: translateY(-1px) !important;
}

/* ── Tabs — superseded by pill style in god-tier block below ───────────── */

/* ── Inputs ────────────────────────────────────────────────────────────── */
.stTextArea textarea, .stTextInput input {
    border-radius: 10px !important;
    border: 1.5px solid #e2e8f0 !important;
    font-size: 14px !important;
    background: #fff !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
    color: #0f172a !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: #4f46e5 !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.1) !important;
}

/* ── Password toggle (eye) button ──────────────────────────────────────── */
/* Streamlit renders the show/hide eye inside .stTextInput, outside div.stButton */
.stTextInput > div > div > button,
.stTextInput [data-testid*="BaseButton"],
[data-testid="stTextInputRootElement"] button {
    background: transparent !important;
    color: #64748b !important;
    border: none !important;
    box-shadow: none !important;
    width: auto !important;
    min-width: unset !important;
    padding: 4px 8px !important;
    border-radius: 6px !important;
    font-size: 16px !important;
    font-weight: normal !important;
    letter-spacing: 0 !important;
    transform: none !important;
    transition: color 0.15s !important;
}
.stTextInput > div > div > button:hover,
.stTextInput [data-testid*="BaseButton"]:hover,
[data-testid="stTextInputRootElement"] button:hover {
    color: #4f46e5 !important;
    background: rgba(79,70,229,0.08) !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── File uploader ─────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] { border-radius: 10px !important; }
/* Completely suppress any visible label above the dropzone */
[data-testid="stFileUploader"] > label,
[data-testid="stFileUploader"] label:not([data-testid]),
[data-testid="stFileUploader"] > div > label {
    display: none !important;
    height: 0 !important;
    overflow: hidden !important;
    margin: 0 !important;
    padding: 0 !important;
}
[data-testid="stFileUploaderDropzone"] {
    border-radius: 10px !important;
    border: 2px dashed #e2e8f0 !important;
    background: #fafbfd !important;
    transition: all 0.2s !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #4f46e5 !important;
    background: #eff6ff !important;
}
/* Override primary-button styles for the uploader's browse/upload button */
[data-testid="stFileUploader"] div.stButton > button,
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploaderDropzone"] div.stButton > button {
    background: #fff !important;
    color: #334155 !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 8px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
    width: auto !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 0.4rem 1rem !important;
    animation: none !important;
    transform: none !important;
    letter-spacing: 0 !important;
}
[data-testid="stFileUploader"] div.stButton > button:hover,
[data-testid="stFileUploaderDropzone"] button:hover,
[data-testid="stFileUploaderDropzone"] div.stButton > button:hover {
    background: #f8fafc !important;
    color: #1e293b !important;
    border-color: #cbd5e1 !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.07) !important;
    transform: none !important;
    animation: none !important;
}

/* ── Bordered containers (step cards) ──────────────────────────────────── */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 14px !important;
    border: 1.5px solid #e2e8f0 !important;
    background: #fff !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
    overflow: hidden !important;
    transition: box-shadow 0.15s !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    box-shadow: 0 2px 10px rgba(0,0,0,0.07) !important;
}

/* ── Expanders ─────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1.5px solid #e8ecf3 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    margin-bottom: 8px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
    background: #fff !important;
}
[data-testid="stExpander"] summary {
    background: #fafbfd !important;
    padding: 12px 18px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #334155 !important;
    transition: background 0.15s !important;
}
[data-testid="stExpander"] summary:hover { background: #f1f5f9 !important; }
[data-testid="stExpander"] summary svg { color: #94a3b8 !important; }

/* ── Metrics ───────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #fff;
    border: 1.5px solid #e2e8f0;
    border-radius: 14px;
    padding: 20px 24px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    transition: box-shadow 0.15s;
}
[data-testid="stMetric"]:hover { box-shadow: 0 3px 10px rgba(0,0,0,0.08); }
[data-testid="stMetricValue"] {
    font-size: 28px !important;
    font-weight: 800 !important;
    color: #0f172a !important;
    letter-spacing: -0.02em !important;
}
[data-testid="stMetricLabel"] {
    font-size: 11px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    color: #94a3b8 !important;
}

/* ── Dividers ──────────────────────────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1.5px solid #f1f5f9 !important;
    margin: 2rem 0 !important;
}

/* ── Alerts ────────────────────────────────────────────────────────────── */
[data-testid="stAlert"] { border-radius: 10px !important; font-size: 13.5px !important; }

/* ── Selectbox ─────────────────────────────────────────────────────────── */
.stSelectbox > div > div {
    border-radius: 10px !important;
    border: 1.5px solid #e2e8f0 !important;
    background: #fff !important;
}
/* Selectbox / Multiselect: force light backgrounds throughout */
[data-baseweb="select"] > div:first-child {
    background: #fff !important;
    border-color: #e2e8f0 !important;
    border-radius: 10px !important;
}
[data-baseweb="popover"],
[data-baseweb="menu"],
[data-baseweb="popover"] ul {
    background: #fff !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.08) !important;
}
[data-baseweb="menu"] li,
[data-baseweb="menu"] [role="option"] {
    background: #fff !important;
    color: #1e293b !important;
}
[data-baseweb="menu"] li:hover,
[data-baseweb="menu"] [role="option"]:hover,
[data-baseweb="menu"] [aria-selected="true"] {
    background: #eff6ff !important;
    color: #4f46e5 !important;
}
/* Multiselect tags */
[data-baseweb="tag"] {
    background: #eef2ff !important;
    color: #4f46e5 !important;
    border-color: #c7d2fe !important;
}
[data-baseweb="tag"] span { color: #4f46e5 !important; }
/* Multiselect clear/remove button on tags */
[data-baseweb="tag"] button {
    color: #6366f1 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    width: auto !important;
    min-width: unset !important;
    padding: 0 !important;
    font-size: 12px !important;
    transform: none !important;
    letter-spacing: 0 !important;
}

/* ── Slider ─────────────────────────────────────────────────────────────── */
[data-testid="stSlider"] [data-testid="stSliderThumbValue"],
[data-testid="stSlider"] output,
[data-testid="stSlider"] p {
    color: #334155 !important;
}
[data-testid="stSlider"] [role="slider"] {
    background: #4f46e5 !important;
    border-color: #4f46e5 !important;
}

/* ── Forms (Security / Danger Zone backgrounds) ──────────────────────────── */
[data-testid="stForm"] {
    background: #fff !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
}
/* Form submit buttons — primary */
[data-testid="stFormSubmitButton"] > button,
[data-testid="stBaseButton-primaryFormSubmit"],
button[kind="primaryFormSubmit"] {
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 0.65rem 1.8rem !important;
    width: 100% !important;
    box-shadow: 0 2px 8px rgba(79,70,229,0.28) !important;
    cursor: pointer !important;
}
/* Danger Zone delete button — keep it red */
[data-testid="stForm"] [data-testid="stFormSubmitButton"] > button[kind="primary"],
[data-testid="stForm"] [data-testid*="danger"] > button {
    background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%) !important;
}

/* ── Dataframe ─────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1.5px solid #e2e8f0 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
}

/* ── Reusable component classes ────────────────────────────────────────── */
.section-label {
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 10px;
    margin-top: 4px;
}
.section-card {
    background: #fff;
    border: 1.5px solid #e8ecf3;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.bullet-item {
    display: flex;
    gap: 10px;
    padding: 8px 0;
    border-bottom: 1px solid #f1f5f9;
    font-size: 13.5px;
    line-height: 1.6;
    color: #1e293b;
}
.bullet-item:last-child { border-bottom: none; }
.bullet-dot { flex-shrink: 0; margin-top: 2px; }
.tag {
    display: inline-block;
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 99px;
    padding: 3px 11px;
    font-size: 12px;
    color: #475569;
    font-weight: 500;
    margin: 2px 2px 2px 0;
    transition: background 0.1s;
}

/* ── Fit score ─────────────────────────────────────────────────────────── */
.fit-hero {
    display: flex;
    align-items: center;
    gap: 28px;
    padding: 28px 32px;
    border-radius: 16px;
    margin-bottom: 8px;
    position: relative;
    overflow: hidden;
}
.fit-score-num {
    font-size: 72px;
    font-weight: 900;
    line-height: 1;
    letter-spacing: -0.04em;
}
.fit-score-denom { font-size: 22px; font-weight: 500; color: #94a3b8; }
.fit-label { font-size: 18px; font-weight: 700; margin-bottom: 10px; }
.fit-bar-track {
    background: rgba(0,0,0,0.08);
    border-radius: 99px;
    height: 8px;
    width: 220px;
    overflow: hidden;
}
.fit-bar-fill { height: 8px; border-radius: 99px; transition: width 0.8s ease; }

/* ── Step header ───────────────────────────────────────────────────────── */
.step-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 18px;
    padding-bottom: 14px;
    border-bottom: 1px solid #f1f5f9;
}
.step-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 26px;
    height: 26px;
    background: linear-gradient(135deg, #4f46e5, #6366f1);
    color: #fff;
    border-radius: 50%;
    font-size: 12px;
    font-weight: 700;
    flex-shrink: 0;
    box-shadow: 0 2px 6px rgba(79,70,229,0.3);
}
.step-title {
    font-size: 15px;
    font-weight: 700;
    color: #0f172a;
}

/* ── Auth screen ───────────────────────────────────────────────────────── */
.auth-page {
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #eff6ff 0%, #f5f3ff 50%, #fdf4ff 100%);
}
.auth-logo-box {
    width: 56px; height: 56px;
    background: linear-gradient(135deg, #4f46e5, #818cf8);
    border-radius: 16px;
    margin: 0 auto 16px;
    display: flex; align-items: center; justify-content: center;
    font-size: 26px;
    box-shadow: 0 6px 20px rgba(79,70,229,0.35);
}
.auth-feature {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 6px 0;
    font-size: 13px;
    color: #475569;
    line-height: 1.5;
}
.auth-feature-dot {
    width: 18px; height: 18px;
    background: #eff6ff;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 10px;
    flex-shrink: 0;
    margin-top: 1px;
}

/* ── Result section heading ────────────────────────────────────────────── */
.result-section-head {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 18px;
}
.result-section-icon {
    font-size: 20px;
    line-height: 1;
}
.result-section-title {
    font-size: 17px;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.01em;
}

/* ── Job card ──────────────────────────────────────────────────────────── */
.job-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13.5px;
}
.job-table thead th {
    padding: 10px 14px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94a3b8;
    background: #f8fafc;
    border-bottom: 2px solid #e2e8f0;
    text-align: left;
}
.job-table tbody tr {
    border-bottom: 1px solid #f1f5f9;
    transition: background 0.1s;
}
.job-table tbody tr:hover { background: #fafbff !important; }

/* ── Dashboard status kanban colors ────────────────────────────────────── */
.status-pill {
    display: inline-flex;
    align-items: center;
    border-radius: 99px;
    padding: 3px 11px;
    font-size: 11.5px;
    font-weight: 600;
    letter-spacing: 0.01em;
    white-space: nowrap;
}

/* ── Progress bar ──────────────────────────────────────────────────────── */
.gen-progress {
    background: linear-gradient(90deg, #eff6ff, #f5f3ff);
    border: 1.5px solid #bfdbfe;
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 13.5px;
    color: #3730a3;
    display: flex;
    align-items: center;
    gap: 12px;
}
.gen-progress-dot {
    width: 8px; height: 8px;
    background: #4f46e5;
    border-radius: 50%;
    flex-shrink: 0;
    animation: pulse 1.2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.8); }
}

/* ── Page entrance animation ───────────────────────────────────────────── */
@keyframes fade-up {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.block-container { animation: fade-up 0.35s ease both !important; }

/* ── Gradient page titles ──────────────────────────────────────────────── */
.grad-title {
    background: linear-gradient(135deg, #0f172a 0%, #3730a3 80%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ── Section labels with left accent ──────────────────────────────────── */
.section-label {
    border-left: 3px solid #e2e8f0;
    padding-left: 8px !important;
}
.sl-indigo { border-left-color: #4f46e5 !important; }
.sl-green  { border-left-color: #15803d !important; }
.sl-amber  { border-left-color: #b45309 !important; }
.sl-slate  { border-left-color: #64748b !important; }
.sl-sky    { border-left-color: #0891b2 !important; }

/* ── Typed bullet-card left border ────────────────────────────────────── */
.section-card-indigo { border-left: 3px solid #4f46e5 !important; }
.section-card-green  { border-left: 3px solid #15803d !important; }
.section-card-amber  { border-left: 3px solid #b45309 !important; }
.section-card-slate  { border-left: 3px solid #94a3b8 !important; }
.section-card-sky    { border-left: 3px solid #0891b2 !important; }

/* ── Tag hover lift ────────────────────────────────────────────────────── */
.tag {
    transition: background 0.12s, transform 0.12s, box-shadow 0.12s !important;
}
.tag:hover {
    background: #e2e8f0 !important;
    border-color: #cbd5e1 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08) !important;
    cursor: default !important;
}

/* ── Logo pulse in sidebar ─────────────────────────────────────────────── */
@keyframes logo-glow {
    0%, 100% { box-shadow: 0 2px 8px rgba(79,70,229,0.35); }
    50%      { box-shadow: 0 2px 20px rgba(99,102,241,0.65), 0 0 0 4px rgba(79,70,229,0.1); }
}
.sidebar-logo { animation: logo-glow 3.5s ease-in-out infinite; }

/* ── Column tinting ────────────────────────────────────────────────────── */
.col-left  { background: #fafbfe; border-radius: 10px; padding: 12px !important; }
.col-right { background: #fdfaf6; border-radius: 10px; padding: 12px !important; }

/* ── Fit bar animated fill ─────────────────────────────────────────────── */
@keyframes bar-grow {
    from { width: 0%; }
}
.fit-bar-fill { animation: bar-grow 0.9s cubic-bezier(.22,1,.36,1) both !important; }
.dim-bar-fill { animation: bar-grow 0.7s cubic-bezier(.22,1,.36,1) both !important; }

/* ── Download chip style ───────────────────────────────────────────────── */
div.stDownloadButton > button {
    width: auto !important;
    border-radius: 99px !important;
    padding: 0.38rem 1.1rem !important;
    font-size: 12.5px !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
}

/* ── Elevated result card ──────────────────────────────────────────────── */
.result-card {
    background: #fff;
    border: 1.5px solid #e8ecf3;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 16px rgba(0,0,0,0.03);
}

/* ════════════════════════════════════════════════════════════════════════
   GOD-TIER LAYER — all additions below
   ════════════════════════════════════════════════════════════════════════ */

/* ── Animated mesh-gradient app background ─────────────────────────────── */
.stApp {
    background: linear-gradient(-45deg, #f8faff, #f4f0ff, #fff7f3, #f0fdf6) !important;
    background-size: 400% 400% !important;
    animation: mesh-shift 32s ease infinite !important;
}
@keyframes mesh-shift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ── Custom slim scrollbar ─────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: #94a3b8; }

/* ── Glassmorphism step / content cards ────────────────────────────────── */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255,255,255,0.88) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border-color: rgba(226,232,240,0.75) !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(79,70,229,0.09), 0 2px 8px rgba(0,0,0,0.05) !important;
}

/* ── Pill tabs (replaces underline) ────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(241,245,249,0.92) !important;
    backdrop-filter: blur(14px) !important;
    -webkit-backdrop-filter: blur(14px) !important;
    border-radius: 12px !important;
    padding: 5px !important;
    border: 1.5px solid #e8ecf3 !important;
    border-bottom: 1.5px solid #e8ecf3 !important;
    gap: 2px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px !important;
    border: none !important;
    border-bottom: none !important;
    margin-bottom: 0 !important;
    padding: 8px 14px !important;
    transition: all 0.18s ease !important;
    color: #64748b !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #334155 !important; }
.stTabs [aria-selected="true"] {
    background: #ffffff !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.10), 0 1px 2px rgba(0,0,0,0.06) !important;
    border-bottom: none !important;
    color: #4f46e5 !important;
    font-weight: 700 !important;
}
.stTabs [data-baseweb="tab-panel"] { padding: 22px 0 0 !important; }
/* Hide sliding indicator bar and bottom border — not needed in pill-tab design.
   Without these rules the indicator can bleed outside the tab-list and appear as
   a stray coloured bar overlapping the content below. */
.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
    height: 0 !important;
    visibility: hidden !important;
}
.stTabs [data-baseweb="tab-border"] {
    display: none !important;
    height: 0 !important;
    visibility: hidden !important;
}
.stTabs [data-baseweb="tab-list"] {
    overflow: hidden !important;
}

/* ── Metric cards — gradient top stripe + glass ────────────────────────── */
[data-testid="stMetric"] {
    position: relative !important;
    overflow: hidden !important;
    background: rgba(255,255,255,0.92) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
}
[data-testid="stMetric"]::after {
    content: '' !important;
    position: absolute !important;
    inset: 0 0 auto 0 !important;
    height: 3px !important;
    background: linear-gradient(90deg, #4f46e5, #818cf8, #c084fc) !important;
    border-radius: 14px 14px 0 0 !important;
}
/* staggered entrance */
[data-testid="column"]:nth-child(1) [data-testid="stMetric"] { animation: fade-up 0.35s 0.05s ease both !important; }
[data-testid="column"]:nth-child(2) [data-testid="stMetric"] { animation: fade-up 0.35s 0.10s ease both !important; }
[data-testid="column"]:nth-child(3) [data-testid="stMetric"] { animation: fade-up 0.35s 0.15s ease both !important; }
[data-testid="column"]:nth-child(4) [data-testid="stMetric"] { animation: fade-up 0.35s 0.20s ease both !important; }

/* ── Fit score hero — spring entrance ──────────────────────────────────── */
.fit-hero {
    animation: score-pop 0.55s cubic-bezier(.34,1.56,.64,1) both !important;
}
@keyframes score-pop {
    from { opacity: 0; transform: scale(0.90); }
    to   { opacity: 1; transform: scale(1); }
}

/* ── Strong match — persistent glow ────────────────────────────────────── */
.fit-hero-strong {
    animation: score-pop 0.55s cubic-bezier(.34,1.56,.64,1) both,
               match-glow 3s 0.6s ease-in-out infinite !important;
}
@keyframes match-glow {
    0%, 100% { box-shadow: 0 0 0 1px #15803d18, 0 6px 28px #15803d0f; }
    50%       { box-shadow: 0 0 0 2px #15803d32, 0 10px 52px #15803d1f; }
}

/* ── Celebration banner ─────────────────────────────────────────────────── */
@keyframes cel-slide {
    from { opacity:0; transform: translateY(-10px) scale(0.96); }
    to   { opacity:1; transform: translateY(0)    scale(1); }
}
.cel-banner { animation: cel-slide 0.45s cubic-bezier(.34,1.56,.64,1) both !important; }
@keyframes emoji-pop {
    0%   { transform: scale(0)    rotate(-20deg); }
    60%  { transform: scale(1.3)  rotate(10deg); }
    100% { transform: scale(1)    rotate(0deg); }
}
.cel-emoji { display:inline-block; animation: emoji-pop 0.65s 0.2s cubic-bezier(.34,1.56,.64,1) both; }

/* ── HR as fading gradient ──────────────────────────────────────────────── */
hr {
    border: none !important;
    border-top: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, #e2e8f0 25%, #e2e8f0 75%, transparent) !important;
    margin: 2rem 0 !important;
}

/* ── Section card — solid (no blur — backdrop-filter creates stacking contexts
   that can interfere with expander icon rendering) ────────────────────────── */
.section-card {
    background: #ffffff !important;
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
}

/* ── Sidebar tip card ───────────────────────────────────────────────────── */
.sidebar-tip {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 11px 14px;
    margin: 14px 8px 0;
}
.sidebar-tip-lbl {
    font-size: 10px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.09em !important;
    color: #4f46e5 !important;
    margin-bottom: 5px;
}
.sidebar-tip-txt {
    font-size: 11.5px !important;
    color: #64748b !important;
    line-height: 1.65 !important;
}

/* ── Generate CTA pulse ─────────────────────────────────────────────────── */
@keyframes gen-pulse {
    0%, 100% { box-shadow: 0 2px 8px rgba(79,70,229,0.30), 0 1px 2px rgba(0,0,0,0.08); }
    50%       { box-shadow: 0 4px 22px rgba(79,70,229,0.55), 0 0 0 4px rgba(79,70,229,0.10); }
}
.gen-cta div.stButton > button { animation: gen-pulse 3.5s ease-in-out infinite !important; }

/* ── Greeting header ────────────────────────────────────────────────────── */
.greeting-line {
    font-size: 13.5px;
    font-weight: 600;
    color: #64748b;
    margin-bottom: 6px;
    letter-spacing: 0.01em;
}

/* ── Streak badge ───────────────────────────────────────────────────────── */
.streak-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(251,191,36,0.12);
    border: 1px solid rgba(251,191,36,0.25);
    border-radius: 99px;
    padding: 4px 11px;
    font-size: 11.5px;
    font-weight: 700;
    color: #b45309;
    margin-top: 10px;
}

/* ── Dataframe highlight on hover ───────────────────────────────────────── */
[data-testid="stDataFrame"] { transition: box-shadow 0.2s !important; }
[data-testid="stDataFrame"]:hover { box-shadow: 0 4px 20px rgba(79,70,229,0.08) !important; }

/* ── Sidebar isolation — prevent god-tier CSS from leaking in ───────────── */
section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
    background: transparent !important;
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
    transform: none !important;
    box-shadow: none !important;
    border: none !important;
    animation: none !important;
    overflow: visible !important;
}
section[data-testid="stSidebar"] * {
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
}

/* ── Scope glassmorphism to main content only ───────────────────────────── */
section[data-testid="stSidebar"] ~ div [data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stMainBlockContainer"] [data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255,255,255,0.88) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# AUTH SCREEN
# ─────────────────────────────────────────────────────────────────────────────

def _render_auth_screen() -> None:
    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        st.markdown("""
        <div style="text-align:center;padding:48px 0 4px;">
            <div class="auth-logo-box">⚡</div>
            <div style="font-size:26px;font-weight:900;color:#0f172a;letter-spacing:-0.03em;
                        margin-bottom:6px;">JobAgent AI</div>
            <div style="font-size:14px;color:#64748b;line-height:1.6;">
                Land your dream job — faster, smarter, with AI that actually knows you.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

        tab_login, tab_register = st.tabs(["Sign In", "Create Account"])

        with tab_login:
            with st.form("login_form"):
                email_in    = st.text_input("Email", placeholder="you@example.com")
                password_in = st.text_input("Password", type="password", placeholder="••••••••")
                submitted   = st.form_submit_button("Sign In →", use_container_width=True)
            if submitted:
                if not email_in or not password_in:
                    st.error("Enter your email and password.")
                else:
                    user = auth_module.login_user(email_in, password_in)
                    if user:
                        st.session_state["user"] = user
                        st.rerun()
                    else:
                        st.error("Invalid email or password.")

        with tab_register:
            with st.form("register_form"):
                reg_name    = st.text_input("Full name", placeholder="Jane Smith")
                reg_email   = st.text_input("Email", placeholder="you@example.com", key="reg_email")
                reg_pw      = st.text_input("Password (8+ chars)", type="password", key="reg_pw",
                                             placeholder="••••••••")
                reg_pw_conf = st.text_input("Confirm password", type="password", key="reg_pw_conf",
                                             placeholder="••••••••")
                reg_submit  = st.form_submit_button("Create Account →", use_container_width=True)
            if reg_submit:
                try:
                    user = auth_module.register_user(reg_email, reg_name, reg_pw, reg_pw_conf)
                    st.session_state["user"] = user
                    st.success("Welcome aboard! Redirecting…")
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))

        st.markdown("""
        <div style="margin-top:28px;padding:18px 20px;background:#fff;border:1.5px solid #e2e8f0;
                    border-radius:12px;box-shadow:0 1px 4px rgba(0,0,0,0.05);">
            <div style="font-size:10.5px;font-weight:700;text-transform:uppercase;letter-spacing:0.09em;
                        color:#94a3b8;margin-bottom:12px;">What you get</div>
            <div class="auth-feature"><div class="auth-feature-dot">🎯</div>
                Fit score that tells you exactly how well you match — before you apply</div>
            <div class="auth-feature"><div class="auth-feature-dot">✉️</div>
                AI-written cover letter that sounds like you, not ChatGPT</div>
            <div class="auth-feature"><div class="auth-feature-dot">📄</div>
                Tailored resume with ATS keywords woven in naturally</div>
            <div class="auth-feature"><div class="auth-feature-dot">💬</div>
                LinkedIn outreach personalized to the actual recruiter</div>
            <div class="auth-feature"><div class="auth-feature-dot">🎤</div>
                Interview prep with exact questions and how to answer each one</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# AUTH GATE
# ─────────────────────────────────────────────────────────────────────────────
if "user" not in st.session_state or not st.session_state["user"]:
    _render_auth_screen()
    st.stop()

_user: dict = st.session_state["user"]
_user_id: str = _user["id"]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _copy_button(text: str, btn_id: str, label: str = "📋 Copy") -> None:
    b64 = base64.b64encode(text.encode("utf-8")).decode()
    st.markdown(
        f'<button id="{btn_id}" '
        f'onclick="navigator.clipboard.writeText(atob(\'{b64}\')).then(function(){{'
        f'var b=document.getElementById(\'{btn_id}\');'
        f'b.innerHTML=\'✓ Copied!\';b.style.color=\'#16a34a\';b.style.borderColor=\'#86efac\';'
        f'setTimeout(function(){{b.innerHTML=\'{label}\';b.style.color=\'\';'
        f'b.style.borderColor=\'\';}},2000);}}).catch(function(){{}})" '
        f'style="background:#fff;border:1.5px solid #e2e8f0;border-radius:8px;'
        f'padding:6px 16px;font-size:12.5px;font-weight:500;color:#475569;'
        f'cursor:pointer;transition:all 0.15s;letter-spacing:0.01em;">{label}</button>',
        unsafe_allow_html=True,
    )


def _fit_colors(score: int) -> tuple[str, str, str]:
    if score >= 8:
        return "#15803d", "#f0fdf4", "Strong Match"
    if score >= 6:
        return "#b45309", "#fffbeb", "Solid Match"
    return "#dc2626", "#fef2f2", "Stretch Role"


def _bullet_html(items: list[str], dot: str = "·", dot_color: str = "#94a3b8",
                 card_class: str = "") -> str:
    rows = "".join(
        f'<div class="bullet-item">'
        f'<span class="bullet-dot" style="color:{dot_color};">{dot}</span>'
        f'<span>{item}</span></div>'
        for item in items
    )
    cls = f"section-card {card_class}" if card_class else "section-card"
    return f'<div class="{cls}">{rows}</div>'


def _tag_html(items: list[str]) -> str:
    return "".join(f'<span class="tag">{item}</span>' for item in items)


def _section_label(text: str, accent: str = "") -> None:
    cls = f"section-label {accent}" if accent else "section-label"
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


def _result_head(icon: str, title: str) -> None:
    st.markdown(
        f'<div class="result-section-head">'
        f'<span class="result-section-icon">{icon}</span>'
        f'<span class="result-section-title">{title}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _info_card(text: str, color: str = "#4f46e5", bg: str = "#eff6ff", border: str = "#bfdbfe") -> None:
    st.markdown(
        f'<div style="font-size:13.5px;line-height:1.7;color:{color};background:{bg};'
        f'border:1.5px solid {border};border-radius:10px;padding:12px 16px;margin-bottom:10px;">'
        f'{text}</div>',
        unsafe_allow_html=True,
    )


def _status_pill_html(status: str) -> str:
    colors = {
        "Applied":      ("#4f46e5", "#eef2ff"),
        "Phone Screen": ("#0891b2", "#ecfeff"),
        "Interview":    ("#7c3aed", "#f5f3ff"),
        "Offer":        ("#15803d", "#f0fdf4"),
        "Rejected":     ("#dc2626", "#fef2f2"),
        "Declined":     ("#64748b", "#f8fafc"),
        "Archived":     ("#94a3b8", "#f8fafc"),
    }
    c, bg = colors.get(status, ("#64748b", "#f8fafc"))
    return (
        f'<span class="status-pill" style="background:{bg};color:{c};'
        f'border:1px solid {c}22;">{status}</span>'
    )


def _greeting(name: str) -> str:
    h = datetime.datetime.now().hour
    if h < 12:   period = "Good morning"
    elif h < 17: period = "Good afternoon"
    else:        period = "Good evening"
    first = (name or "").split()[0]
    return f"{period}{', ' + first if first else ''} 👋"


_TIPS = [
    "Apply within 3 days of a posting going live — callbacks drop 40% after that.",
    "Mention the company name at least 3× in your cover letter.",
    "LinkedIn messages sent Tuesday–Thursday mornings get the highest response rates.",
    "A 7+ fit score means you're genuinely competitive. Don't skip strong matches.",
    "Tailoring your resume to a JD increases ATS pass-through by up to 50%.",
    "One quantified achievement per bullet beats three vague ones every time.",
    "Research the hiring manager before your first call. One shared detail builds rapport instantly.",
    "Send a follow-up email 5 business days after applying — most candidates never do.",
]

def _daily_tip() -> str:
    idx = int(hashlib.md5(str(datetime.date.today()).encode()).hexdigest(), 16) % len(_TIPS)
    return _TIPS[idx]


def _confetti_burst() -> None:
    """Fire a canvas-based confetti burst — fires once per render."""
    st.markdown("""
<canvas id="__cc__" style="position:fixed;top:0;left:0;width:100vw;height:100vh;
pointer-events:none;z-index:99999;"></canvas>
<script>
(function(){
  var c=document.getElementById('__cc__');
  if(!c)return;
  var ctx=c.getContext('2d');
  c.width=window.innerWidth; c.height=window.innerHeight;
  var cols=['#4f46e5','#818cf8','#c084fc','#15803d','#86efac',
            '#fbbf24','#fb923c','#f472b6','#06b6d4','#f43f5e'];
  var ps=Array.from({length:160},function(_,i){return{
    x:Math.random()*c.width, y:-20-Math.random()*320,
    r:3+Math.random()*6, col:cols[i%cols.length],
    vy:3+Math.random()*5.5, vx:(Math.random()-0.5)*5,
    rot:Math.random()*360, spin:(Math.random()-0.5)*11,
    rect:Math.random()<0.65, op:1
  };});
  var f=0;
  (function draw(){
    ctx.clearRect(0,0,c.width,c.height);
    var alive=false;
    ps.forEach(function(p){
      if(p.y<c.height+40&&p.op>0)alive=true;
      p.y+=p.vy; p.x+=p.vx; p.rot+=p.spin; p.vy+=0.13;
      if(p.y>c.height*0.62)p.op=Math.max(0,p.op-0.028);
      ctx.save();
      ctx.globalAlpha=p.op;
      ctx.translate(p.x,p.y);
      ctx.rotate(p.rot*Math.PI/180);
      ctx.fillStyle=p.col;
      if(p.rect){ctx.fillRect(-p.r/2,-p.r/2,p.r,p.r*1.9);}
      else{ctx.beginPath();ctx.arc(0,0,p.r/2,0,Math.PI*2);ctx.fill();}
      ctx.restore();
    });
    f++;
    if(alive&&f<420)requestAnimationFrame(draw);
    else{ctx.clearRect(0,0,c.width,c.height);try{c.remove();}catch(e){}}
  })();
})();
</script>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FIT SCORE RENDERER
# ─────────────────────────────────────────────────────────────────────────────

def render_fit_score(result: dict) -> None:
    score     = int(result.get("fit_score") or 0)
    scoring   = result.get("fit_scoring") or {}
    rationale = result.get("fit_rationale", "")
    color, bg, label = _fit_colors(score)
    bar_pct = score * 10

    DIMENSIONS = [
        ("must_have_requirements",  "Must-Have Requirements", 0.35),
        ("technical_skill_overlap", "Technical Skill Overlap", 0.25),
        ("years_and_seniority",     "Years & Seniority",      0.20),
        ("leadership_and_scope",    "Leadership & Scope",     0.10),
        ("domain_and_industry_fit", "Domain & Industry Fit",  0.10),
    ]

    _circ = 263.9  # 2π × r=42
    _fill = round(_circ * bar_pct / 100, 1)
    _hero_cls = "fit-hero fit-hero-strong" if score >= 8 else "fit-hero"

    # Confetti + celebration banner for outstanding matches
    if score >= 8:
        _confetti_burst()
        st.markdown(f"""
        <div class="cel-banner" style="display:flex;align-items:center;gap:16px;
             background:linear-gradient(135deg,#f0fdf4,#dcfce7);
             border:1.5px solid #86efac;border-radius:14px;
             padding:18px 22px;margin-bottom:14px;">
            <span class="cel-emoji" style="font-size:38px;line-height:1;">🎉</span>
            <div>
                <div style="font-size:16px;font-weight:800;color:#15803d;
                            letter-spacing:-0.01em;margin-bottom:4px;">
                    Outstanding Match — You Should Apply!</div>
                <div style="font-size:13px;color:#166534;line-height:1.65;">
                    You're highly competitive for this role. Your materials are fully
                    optimized — send that application today.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="{_hero_cls}" style="background:{bg};border:1.5px solid {color}30;">
        <div style="flex-shrink:0;position:relative;width:108px;height:108px;">
            <svg width="108" height="108" viewBox="0 0 108 108" style="display:block;">
                <circle cx="54" cy="54" r="42" fill="none"
                        stroke="rgba(0,0,0,0.07)" stroke-width="9" stroke-linecap="round"/>
                <circle cx="54" cy="54" r="42" fill="none"
                        stroke="{color}" stroke-width="9" stroke-linecap="round"
                        stroke-dasharray="{_fill} {_circ}"
                        transform="rotate(-90 54 54)"
                        style="transition:stroke-dasharray 0.9s cubic-bezier(.22,1,.36,1);"/>
                <text x="54" y="50" text-anchor="middle" dominant-baseline="central"
                      font-size="26" font-weight="900" fill="{color}"
                      font-family="Inter,-apple-system,sans-serif">{score}</text>
                <text x="54" y="72" text-anchor="middle"
                      font-size="11" fill="#94a3b8"
                      font-family="Inter,-apple-system,sans-serif">/10</text>
            </svg>
        </div>
        <div style="flex:1;">
            <div class="fit-label" style="color:{color};margin-bottom:6px;">{label}</div>
            <div style="font-size:12px;color:{color}bb;font-weight:500;line-height:1.6;">
                Scored across 5 weighted dimensions
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if rationale:
        st.markdown(
            f'<div style="font-size:13.5px;color:#475569;margin-bottom:16px;'
            f'padding:12px 16px;background:rgba(255,255,255,0.9);border:1.5px solid #e2e8f0;'
            f'border-radius:10px;line-height:1.7;box-shadow:0 1px 3px rgba(0,0,0,0.04);">'
            f'{rationale}</div>',
            unsafe_allow_html=True,
        )

    if scoring:
        with st.expander("📊 Score breakdown by dimension"):
            for key, dim_label, weight in DIMENSIONS:
                dim = scoring.get(key, {})
                dim_score = int(dim.get("score") or 0)
                dc, _, _ = _fit_colors(dim_score)
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:12px;'
                    f'padding:10px 0;border-bottom:1px solid #f1f5f9;">'
                    f'<div style="width:180px;font-size:13px;font-weight:600;color:#334155;flex-shrink:0;">'
                    f'{dim_label}</div>'
                    f'<div style="width:36px;font-size:15px;font-weight:800;color:{dc};flex-shrink:0;">'
                    f'{dim_score}</div>'
                    f'<div style="flex:1;background:#e8ecf3;border-radius:99px;height:6px;">'
                    f'<div class="dim-bar-fill" style="background:{dc};width:{dim_score*10}%;height:6px;border-radius:99px;"></div>'
                    f'</div>'
                    f'<div style="width:36px;font-size:11px;color:#94a3b8;flex-shrink:0;text-align:right;">'
                    f'{int(weight*100)}%</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if dim.get("rationale"):
                    st.markdown(
                        f'<div style="font-size:12px;color:#64748b;padding:4px 0 10px 228px;">'
                        f'{dim["rationale"]}</div>',
                        unsafe_allow_html=True,
                    )


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY EXPANDER
# ─────────────────────────────────────────────────────────────────────────────

def render_strategy_expander(strategy: dict, key_prefix: str, kind: str = "Cover Letter") -> None:
    if not strategy:
        return
    with st.expander(f"🧠  AI writing strategy — how this {kind.lower()} was crafted"):
        if kind == "Cover Letter":
            c1, c2 = st.columns(2)
            with c1:
                for lbl, k in [
                    ("Narrative angle", "candidate_narrative_angle"),
                    ("Tone", "tone_guidance"),
                    ("Company culture signal", "company_culture_signals"),
                ]:
                    val = strategy.get(k, "")
                    if val:
                        st.markdown(
                            f'<div style="margin-bottom:12px;">'
                            f'<div class="section-label">{lbl}</div>'
                            f'<div style="font-size:13px;color:#334155;line-height:1.6;">{val}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
            with c2:
                for lbl, k in [
                    ("Key themes", "key_themes"),
                    ("Language mirrored", "company_language_to_mirror"),
                    ("What was avoided", "what_to_avoid"),
                ]:
                    items = strategy.get(k) or []
                    if items:
                        _section_label(lbl)
                        if isinstance(items, list) and all(isinstance(x, str) for x in items):
                            st.markdown(_tag_html(items), unsafe_allow_html=True)
        else:
            c1, c2 = st.columns(2)
            with c1:
                for lbl, k in [("Platform context", "platform_context"), ("Conversation angle", "conversation_angle")]:
                    val = strategy.get(k, "")
                    if val:
                        st.markdown(
                            f'<div style="margin-bottom:12px;">'
                            f'<div class="section-label">{lbl}</div>'
                            f'<div style="font-size:13px;color:#334155;line-height:1.6;">{val}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
            with c2:
                for lbl, k in [("Tone", "tone_guidance"), ("CTA", "cta")]:
                    val = strategy.get(k, "")
                    if val:
                        st.markdown(
                            f'<div style="margin-bottom:12px;">'
                            f'<div class="section-label">{lbl}</div>'
                            f'<div style="font-size:13px;color:#334155;line-height:1.6;">{val}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                avoid = strategy.get("what_to_avoid") or []
                if avoid:
                    _section_label("What was avoided")
                    st.markdown(_tag_html(avoid), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# RENDER RESULT — tabbed layout
# ─────────────────────────────────────────────────────────────────────────────

def render_result(result: dict, key_suffix: str = "") -> None:

    tabs = st.tabs([
        "🎯  Fit Score",
        "📄  Resume",
        "✉️  Cover Letter",
        "💬  LinkedIn",
        "🏢  Company Intel",
        "🎤  Interview",
    ])

    # ── Tab 1: Fit Score ──────────────────────────────────────────────────────
    with tabs[0]:
        if result.get("fit_score") is not None:
            render_fit_score(result)

        st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)

        col_req, col_bul = st.columns(2)
        with col_req:
            _result_head("▸", "Key Requirements")
            st.markdown(_bullet_html(result.get("key_requirements", []), dot="▸",
                        dot_color="#64748b", card_class="section-card-slate"),
                        unsafe_allow_html=True)
        with col_bul:
            _result_head("★", "Resume Bullets to Highlight")
            st.markdown(_bullet_html(result.get("resume_bullets", []), dot="★",
                        dot_color="#4f46e5", card_class="section-card-indigo"),
                        unsafe_allow_html=True)

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        col_str, col_gap = st.columns(2)
        with col_str:
            _result_head("✅", "Strengths")
            st.markdown(_bullet_html(result.get("strengths", []), dot="✓",
                        dot_color="#15803d", card_class="section-card-green"),
                        unsafe_allow_html=True)
        with col_gap:
            _result_head("⚠️", "Gaps to Address")
            st.markdown(_bullet_html(result.get("gaps", []), dot="△",
                        dot_color="#b45309", card_class="section-card-amber"),
                        unsafe_allow_html=True)

    # ── Tab 2: Resume ─────────────────────────────────────────────────────────
    with tabs[1]:
        tailored = result.get("tailored_resume")
        if tailored:
            _result_head("📄", "Tailored Resume")

            summary = tailored.get("summary", "")
            if summary:
                st.markdown(
                    f'<div class="section-card section-card-indigo">'
                    f'<div style="font-size:10.5px;font-weight:700;text-transform:uppercase;'
                    f'letter-spacing:0.09em;color:#4f46e5;margin-bottom:10px;">Professional Summary</div>'
                    f'<div style="font-size:14px;line-height:1.75;color:#1e293b;">{summary}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            experience = tailored.get("experience") or []
            if experience:
                with st.expander("📋 Rewritten experience bullets"):
                    for exp in experience:
                        st.markdown(
                            f'<div style="margin-bottom:6px;padding-bottom:4px;">'
                            f'<span style="font-weight:700;font-size:14px;color:#0f172a;">'
                            f'{exp.get("company","")}</span>'
                            f'<span style="color:#94a3b8;font-size:13px;"> · {exp.get("location","")}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        roles = exp.get("roles") or []
                        if not roles and exp.get("title"):
                            roles = [{"title": exp.get("title",""), "dates": exp.get("dates",""),
                                      "bullets": exp.get("bullets") or []}]
                        for role in roles:
                            st.markdown(
                                f'<div style="font-size:13px;color:#475569;font-style:italic;'
                                f'margin:2px 0 6px;display:flex;justify-content:space-between;">'
                                f'<span>{role.get("title","")}</span>'
                                f'<span style="color:#94a3b8;">{role.get("dates","")}</span>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                            for b in role.get("bullets") or []:
                                st.markdown(
                                    f'<div class="bullet-item">'
                                    f'<span class="bullet-dot" style="color:#4f46e5;">▸</span>'
                                    f'<span style="font-size:13px;">{b}</span></div>',
                                    unsafe_allow_html=True,
                                )
                        st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)

            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
            try:
                resume_style = st.session_state.get("resume_style", {})
                resume_pdf = export_resume_pdf(tailored, style_params=resume_style)
                pdf_bytes = resume_pdf.getvalue()
                b64_pdf = base64.b64encode(pdf_bytes).decode()
                st.markdown(
                    f'<iframe src="data:application/pdf;base64,{b64_pdf}" width="100%" height="820"'
                    f' style="border:1.5px solid #e8ecf3;border-radius:10px;display:block;"></iframe>',
                    unsafe_allow_html=True,
                )
                st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
                candidate = tailored.get("candidate_name", "resume").replace(" ", "_").lower()
                st.download_button(
                    "⬇️  Download Tailored Resume (1-page PDF)",
                    data=pdf_bytes,
                    file_name=f"{candidate}_tailored_resume.pdf",
                    mime="application/pdf",
                    key=f"resume_dl_{key_suffix}",
                )
            except Exception as e:
                st.warning(f"PDF generation failed: {e}")

        optimization = result.get("resume_optimization")
        if optimization:
            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
            with st.expander("🔬 What the AI critic found & fixed"):
                brief = optimization.get("overall_optimization_brief", "")
                if brief:
                    _info_card(f"<strong>Optimization applied:</strong> {brief}",
                               "#15803d", "#f0fdf4", "#86efac")

                oc1, oc2 = st.columns(2)
                with oc1:
                    kw = optimization.get("missing_keywords") or []
                    if kw:
                        _section_label("Keywords added")
                        st.markdown(_tag_html(kw), unsafe_allow_html=True)
                    tf = optimization.get("untapped_transferable_skills") or []
                    if tf:
                        _section_label("Transferable skills surfaced")
                        st.markdown(_bullet_html(tf), unsafe_allow_html=True)
                with oc2:
                    hw = optimization.get("hidden_wins") or []
                    if hw:
                        _section_label("Hidden wins highlighted")
                        st.markdown(_bullet_html(hw), unsafe_allow_html=True)
                    lg = optimization.get("language_gaps") or []
                    if lg:
                        _section_label("Vocabulary aligned to JD")
                        rows_lg = "".join(
                            f'<div class="bullet-item">'
                            f'<span style="color:#94a3b8;font-size:12px;">"{g.get("resume_uses","")}"</span>'
                            f'<span style="color:#4f46e5;margin:0 8px;font-weight:700;">→</span>'
                            f'<span style="font-size:13px;">"{g.get("jd_uses","")}"</span>'
                            f'</div>'
                            for g in lg
                        )
                        st.markdown(f'<div class="section-card">{rows_lg}</div>', unsafe_allow_html=True)

                wb_list = optimization.get("weak_bullets") or []
                if wb_list:
                    with st.expander(f"✏️  {len(wb_list)} bullet(s) rewritten"):
                        for wb in wb_list:
                            st.markdown(
                                f'<div style="margin-bottom:12px;font-size:13px;">'
                                f'<div style="color:#dc2626;text-decoration:line-through;margin-bottom:3px;">'
                                f'{wb.get("original","")}</div>'
                                f'<div style="color:#94a3b8;font-size:12px;margin-bottom:3px;">'
                                f'Issue: {wb.get("issue","")}</div>'
                                f'<div style="color:#15803d;">↳ {wb.get("improvement_direction","")}</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

    # ── Tab 3: Cover Letter ───────────────────────────────────────────────────
    with tabs[2]:
        _result_head("✉️", "Cover Letter")
        render_strategy_expander(result.get("cover_letter_strategy"), key_suffix, "Cover Letter")

        cover_letter = result.get("cover_letter", "")
        if cover_letter:
            st.markdown(
                f'<div class="section-card section-card-indigo" style="font-size:14px;line-height:1.85;'
                f'color:#1e293b;white-space:pre-wrap;font-family:inherit;">'
                f'{html.escape(cover_letter)}</div>',
                unsafe_allow_html=True,
            )

            dl_c, cp_c, _ = st.columns([1.2, 1, 3])
            with dl_c:
                pdf_buf = export_cover_letter_pdf(cover_letter)
                _company = (result.get("cover_letter_strategy") or {}).get("company_name", "company")
                _safe = re.sub(r"[^\w\s-]", "", _company).strip().replace(" ", "_").lower() or "company"
                st.download_button(
                    "⬇️  Download PDF",
                    data=pdf_buf,
                    file_name=f"{_safe}_cover_letter.pdf",
                    mime="application/pdf",
                    key=f"dl_{key_suffix}",
                )
            with cp_c:
                st.markdown("<div style='margin-top:4px;'></div>", unsafe_allow_html=True)
                _copy_button(cover_letter, f"copy-cl-{key_suffix}", "📋 Copy")

        fc = result.get("fact_check_report")
        if fc:
            had_fab = fc.get("had_fabrications", False)
            fab_list = fc.get("fabrications_removed") or []
            icon  = "⚠️" if had_fab else "✅"
            fc_color = "#b45309" if had_fab else "#15803d"
            fc_bg    = "#fffbeb" if had_fab else "#f0fdf4"
            fc_bdr   = "#fde68a" if had_fab else "#bbf7d0"
            fc_label = f"Removed {len(fab_list)} fabricated claim(s)" if had_fab else "All claims verified against your resume"
            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
            with st.expander(f"{icon} Fact-check — {fc_label}"):
                _info_card(fc.get("summary",""), fc_color, fc_bg, fc_bdr)
                fc_c1, fc_c2 = st.columns(2)
                with fc_c1:
                    st.metric("Cover letter accuracy", f"{fc.get('cover_letter_accuracy_pct',100)}%")
                with fc_c2:
                    st.metric("LinkedIn accuracy", f"{fc.get('linkedin_accuracy_pct',100)}%")
                if fab_list:
                    _section_label("Claims removed")
                    st.markdown(_bullet_html(fab_list, dot="✕", dot_color="#dc2626"), unsafe_allow_html=True)

    # ── Tab 4: LinkedIn ───────────────────────────────────────────────────────
    with tabs[3]:
        _result_head("💬", "LinkedIn Outreach Message")

        recruiter_analysis = result.get("recruiter_analysis")
        if recruiter_analysis and recruiter_analysis.get("top_hook"):
            top_hook = recruiter_analysis.get("top_hook", "")
            brief_ra = recruiter_analysis.get("personalization_brief", "")
            secondary = recruiter_analysis.get("secondary_hooks") or []
            with st.expander("🤝 Recruiter personalization — shared connections used"):
                _info_card(f"<strong>Primary hook:</strong> {top_hook}")
                if brief_ra:
                    st.markdown(
                        f'<div style="font-size:13px;color:#475569;margin-bottom:12px;">{brief_ra}</div>',
                        unsafe_allow_html=True,
                    )
                rec_c1, rec_c2 = st.columns(2)
                with rec_c1:
                    for lbl, key in [("Shared schools","shared_schools"),
                                      ("Shared employers","shared_employers"),
                                      ("Shared locations","shared_locations")]:
                        items = recruiter_analysis.get(key) or []
                        if items:
                            _section_label(lbl)
                            rows_r = "".join(
                                f'<div class="bullet-item">'
                                f'<span class="bullet-dot" style="color:#4f46e5;">·</span>'
                                f'<span>{it.get("school") or it.get("company") or it.get("location","")}'
                                f'<span style="color:#94a3b8;font-size:12px;"> — {it.get("hook","")}'
                                f'</span></span></div>'
                                for it in items
                            )
                            st.markdown(f'<div class="section-card">{rows_r}</div>', unsafe_allow_html=True)
                with rec_c2:
                    for lbl, key in [("Shared interests","shared_interests_or_domains"),
                                      ("Career parallels","career_parallels"),
                                      ("Other","other_commonalities")]:
                        items = recruiter_analysis.get(key) or []
                        if items:
                            _section_label(lbl)
                            rows_r = "".join(
                                f'<div class="bullet-item">'
                                f'<span class="bullet-dot" style="color:#4f46e5;">·</span>'
                                f'<span>{it.get("topic") or it.get("parallel") or it.get("detail","")}'
                                f'<span style="color:#94a3b8;font-size:12px;"> — {it.get("hook","")}'
                                f'</span></span></div>'
                                for it in items
                            )
                            st.markdown(f'<div class="section-card">{rows_r}</div>', unsafe_allow_html=True)
                if secondary:
                    _section_label("Additional hooks woven in")
                    st.markdown(_tag_html(secondary), unsafe_allow_html=True)

        render_strategy_expander(result.get("linkedin_strategy"), key_suffix, "LinkedIn")

        li_msg = result.get("linkedin_message", "")
        if li_msg:
            st.markdown(
                f'<div class="section-card section-card-sky" style="font-size:14px;line-height:1.8;color:#1e293b;">'
                f'{li_msg.replace(chr(10), "<br>")}</div>',
                unsafe_allow_html=True,
            )
            _copy_button(li_msg, f"copy-li-{key_suffix}", "📋 Copy message")

    # ── Tab 5: Company Intel ──────────────────────────────────────────────────
    with tabs[4]:
        company_research = result.get("company_research")
        if not company_research:
            st.markdown("""
            <div style="text-align:center;padding:60px 20px;color:#94a3b8;">
                <div style="font-size:40px;margin-bottom:12px;">🏢</div>
                <div style="font-size:15px;font-weight:600;color:#64748b;">No company intel yet</div>
                <div style="font-size:13px;margin-top:6px;">
                    Company research is included with every new generation.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            _result_head("🏢", "Company Intel")

            snapshot = company_research.get("company_snapshot", "")
            if snapshot:
                st.markdown(
                    f'<div class="section-card section-card-indigo" style="font-size:14px;line-height:1.8;color:#1e293b;">'
                    f'{snapshot}</div>',
                    unsafe_allow_html=True,
                )

            ci_c1, ci_c2 = st.columns(2)
            with ci_c1:
                prob = company_research.get("what_problem_this_hire_solves", "")
                if prob:
                    _section_label("Problem this hire solves", "sl-amber")
                    st.markdown(
                        f'<div class="section-card" style="font-size:13px;line-height:1.65;">{prob}</div>',
                        unsafe_allow_html=True,
                    )
                culture = company_research.get("culture_read", "")
                if culture:
                    _section_label("Culture read", "sl-indigo")
                    st.markdown(
                        f'<div class="section-card" style="font-size:13px;line-height:1.65;">{culture}</div>',
                        unsafe_allow_html=True,
                    )
            with ci_c2:
                success = company_research.get("what_success_looks_like_90_days", "")
                if success:
                    _section_label("Success at 90 days", "sl-green")
                    st.markdown(
                        f'<div class="section-card" style="font-size:13px;line-height:1.65;">{success}</div>',
                        unsafe_allow_html=True,
                    )
                angle = company_research.get("interview_angle", "")
                if angle:
                    _section_label("Your interview angle")
                    _info_card(angle)

            tps = company_research.get("smart_talking_points") or []
            if tps:
                _section_label("Smart talking points for the interview", "sl-indigo")
                st.markdown(_bullet_html(tps, dot="💡", card_class="section-card-indigo"), unsafe_allow_html=True)

            rl = company_research.get("what_to_research_before_interview") or []
            if rl:
                _section_label("Research checklist", "sl-slate")
                st.markdown(_bullet_html(rl, dot="☐"), unsafe_allow_html=True)

            pq = company_research.get("questions_to_probe_in_interview") or []
            if pq:
                _section_label("Questions to probe (insight > enthusiasm)", "sl-indigo")
                st.markdown(_bullet_html(pq, dot="?", dot_color="#4f46e5", card_class="section-card-indigo"), unsafe_allow_html=True)

            rf = company_research.get("potential_red_flags_to_probe") or []
            if rf:
                _section_label("Red flags to probe", "sl-amber")
                st.markdown(_bullet_html(rf, dot="⚠", dot_color="#b45309", card_class="section-card-amber"), unsafe_allow_html=True)

    # ── Tab 6: Interview ──────────────────────────────────────────────────────
    with tabs[5]:
        interview_prep = result.get("interview_prep")
        follow_up = result.get("follow_up_email")

        if not interview_prep and not follow_up:
            st.markdown("""
            <div style="text-align:center;padding:60px 20px;color:#94a3b8;">
                <div style="font-size:40px;margin-bottom:12px;">🎤</div>
                <div style="font-size:15px;font-weight:600;color:#64748b;">Interview prep coming soon</div>
                <div style="font-size:13px;margin-top:6px;">
                    Included automatically in every new generation.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            if interview_prep:
                _result_head("🎤", "Interview Prep")

                elevator = interview_prep.get("elevator_pitch", "")
                if elevator:
                    _section_label("Your elevator pitch", "sl-green")
                    st.markdown(
                        f'<div style="font-size:14px;line-height:1.8;color:#1e293b;'
                        f'background:#f0fdf4;border:1.5px solid #86efac;border-radius:12px;'
                        f'padding:16px 20px;margin-bottom:8px;">{elevator}</div>',
                        unsafe_allow_html=True,
                    )
                    _copy_button(elevator, f"copy-elev-{key_suffix}", "📋 Copy pitch")

                likely_q = interview_prep.get("likely_questions") or []
                if likely_q:
                    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
                    _section_label("Likely questions & how to answer", "sl-indigo")
                    for qi, q_item in enumerate(likely_q):
                        q_text  = q_item.get("question", "")
                        why_ask = q_item.get("why_they_ask", "")
                        af      = q_item.get("answer_framework") or {}
                        with st.expander(f"Q: {q_text}"):
                            if why_ask:
                                st.markdown(
                                    f'<div style="font-size:12px;color:#64748b;font-style:italic;'
                                    f'margin-bottom:12px;">Why they ask: {why_ask}</div>',
                                    unsafe_allow_html=True,
                                )
                            for lbl, key_af, c, bg in [
                                ("Setup",        "situation_setup",  "#0ea5e9", "#f0f9ff"),
                                ("Key Evidence", "key_evidence",     "#15803d", "#f0fdf4"),
                                ("Strong Close", "strong_close",     "#4f46e5", "#eff6ff"),
                            ]:
                                text_af = af.get(key_af, "")
                                if text_af:
                                    st.markdown(
                                        f'<div style="background:{bg};border-left:3px solid {c};'
                                        f'border-radius:0 8px 8px 0;padding:10px 14px;margin-bottom:8px;'
                                        f'font-size:13px;line-height:1.65;color:#1e293b;">'
                                        f'<strong style="color:{c};font-size:10.5px;text-transform:uppercase;'
                                        f'letter-spacing:0.06em;">{lbl}</strong><br>{text_af}</div>',
                                        unsafe_allow_html=True,
                                    )
                            pts = af.get("talking_points") or []
                            if pts:
                                _section_label("Talking points")
                                st.markdown(_bullet_html(pts, dot="▸"), unsafe_allow_html=True)
                            pitfall = af.get("pitfall_to_avoid", "")
                            if pitfall:
                                st.markdown(
                                    f'<div style="background:#fffbeb;border:1.5px solid #fde68a;'
                                    f'border-radius:8px;padding:10px 14px;font-size:12.5px;color:#92400e;">'
                                    f'⚠ Pitfall to avoid: {pitfall}</div>',
                                    unsafe_allow_html=True,
                                )

                dz_list = interview_prep.get("danger_zones") or []
                if dz_list:
                    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
                    _section_label("Danger zones — questions to reframe", "sl-amber")
                    for dz in dz_list:
                        with st.expander(f"⚠ {dz.get('question','')}"):
                            why_d = dz.get("why_dangerous", "")
                            reframe = dz.get("reframe_strategy", "")
                            bridge = dz.get("bridging_evidence", "")
                            if why_d:
                                st.markdown(
                                    f'<div style="color:#b45309;font-size:13px;margin-bottom:8px;">'
                                    f'{why_d}</div>',
                                    unsafe_allow_html=True,
                                )
                            if reframe:
                                st.markdown(
                                    f'<div style="background:#fffbeb;border-left:3px solid #f59e0b;'
                                    f'padding:10px 14px;border-radius:0 8px 8px 0;font-size:13px;'
                                    f'color:#1e293b;margin-bottom:6px;">'
                                    f'<strong>Reframe:</strong> {reframe}</div>',
                                    unsafe_allow_html=True,
                                )
                            if bridge:
                                st.markdown(
                                    f'<div style="font-size:12px;color:#64748b;">Bridge: {bridge}</div>',
                                    unsafe_allow_html=True,
                                )

                qa_list = interview_prep.get("questions_to_ask") or []
                if qa_list:
                    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
                    _section_label("Power questions to ask them", "sl-indigo")
                    rows_qa = ""
                    for qa in qa_list:
                        why_p = qa.get("why_powerful", "")
                        rows_qa += (
                            f'<div class="bullet-item">'
                            f'<span class="bullet-dot" style="color:#4f46e5;font-weight:700;">→</span>'
                            f'<span><strong style="font-size:13px;color:#0f172a;">'
                            f'{qa.get("question","")}</strong>'
                            + (f'<br><span style="color:#94a3b8;font-size:12px;">{why_p}</span>'
                               if why_p else "")
                            + '</span></div>'
                        )
                    st.markdown(f'<div class="section-card">{rows_qa}</div>', unsafe_allow_html=True)

                checklist = interview_prep.get("preparation_checklist") or []
                if checklist:
                    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
                    _section_label("Preparation checklist", "sl-slate")
                    st.markdown(_bullet_html(checklist, dot="☐"), unsafe_allow_html=True)

            if follow_up:
                st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
                _result_head("📧", "Follow-Up Email")
                subject = follow_up.get("subject_line", "")
                body    = follow_up.get("body", "")
                timing  = follow_up.get("send_timing", "")
                note    = follow_up.get("personalization_note", "")

                if subject:
                    st.markdown(
                        f'<div class="section-label">Subject line</div>'
                        f'<div style="font-size:14px;font-weight:600;color:#0f172a;background:#fff;'
                        f'border:1.5px solid #e2e8f0;border-radius:10px;padding:10px 16px;'
                        f'margin-bottom:12px;box-shadow:0 1px 3px rgba(0,0,0,0.04);">{subject}</div>',
                        unsafe_allow_html=True,
                    )
                if body:
                    st.markdown(
                        f'<div class="section-card" style="font-size:13.5px;line-height:1.8;'
                        f'color:#1e293b;white-space:pre-wrap;">{body}</div>',
                        unsafe_allow_html=True,
                    )
                    _copy_button(
                        f"Subject: {subject}\n\n{body}",
                        f"copy-fu-{key_suffix}",
                        "📋 Copy email",
                    )
                meta = []
                if timing:
                    meta.append(f"🕒 <strong>When to send:</strong> {timing}")
                if note:
                    meta.append(f"💡 {note}")
                for m in meta:
                    st.markdown(
                        f'<div style="font-size:12.5px;color:#64748b;margin-top:8px;">{m}</div>',
                        unsafe_allow_html=True,
                    )


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand
    st.markdown("""
    <div style="padding:28px 20px 20px;border-bottom:1px solid rgba(255,255,255,0.06);
                margin-bottom:8px;">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:18px;">
            <div class="sidebar-logo" style="width:34px;height:34px;background:linear-gradient(135deg,#4f46e5,#818cf8);
                        border-radius:9px;display:flex;align-items:center;justify-content:center;
                        font-size:16px;flex-shrink:0;">⚡</div>
            <div style="font-size:17px;font-weight:800;color:#f8fafc;letter-spacing:-0.02em;">
                JobAgent AI</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Navigation ────────────────────────────────────────────────────────
    # Native st.button() per nav item — guaranteed clickable, no JS needed.
    # Active item is a styled markdown div (non-clickable); inactive = button.
    _NAV_ITEMS = [
        ("⚡  Generate",   "Generate"),
        ("📊  Dashboard",  "Dashboard"),
        ("💼  Jobs",       "Jobs"),
        ("⚙️  Account",    "Account"),
    ]
    if "page" not in st.session_state:
        st.session_state["page"] = "Generate"

    for _label, _key in _NAV_ITEMS:
        if st.session_state["page"] == _key:
            st.markdown(
                f'<div class="sidebar-nav-active">{_label}</div>',
                unsafe_allow_html=True,
            )
        else:
            if st.button(_label, key=f"nav_{_key}", use_container_width=True):
                st.session_state["page"] = _key
                st.rerun()

    page = st.session_state["page"]

    # User card + stats
    _app_count = db.application_count(_user_id)
    _name_initial = _user.get("name", "?")[0].upper()
    st.markdown(f"""
    <div style="margin-top:16px;padding:0 8px;">
        <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.07);
                    border-radius:12px;padding:14px 16px;margin-bottom:8px;">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
                <div style="width:30px;height:30px;border-radius:50%;
                            background:linear-gradient(135deg,#4f46e5,#818cf8);
                            display:flex;align-items:center;justify-content:center;
                            font-size:12px;font-weight:700;color:#fff;flex-shrink:0;">
                    {_name_initial}</div>
                <div>
                    <div style="font-size:13px;font-weight:600;color:#e2e8f0;
                                white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
                                max-width:140px;">{_user.get("name","")}</div>
                    <div style="font-size:11px;color:#475569;word-break:break-all;
                                max-width:140px;overflow:hidden;text-overflow:ellipsis;
                                white-space:nowrap;">{_user.get("email","")}</div>
                </div>
            </div>
            <div style="display:flex;align-items:center;justify-content:space-between;">
                <div>
                    <div style="font-size:22px;font-weight:800;color:#f1f5f9;
                                letter-spacing:-0.03em;">{_app_count}</div>
                    <div style="font-size:10.5px;color:#475569;font-weight:600;
                                text-transform:uppercase;letter-spacing:0.06em;">
                        application{"s" if _app_count != 1 else ""} generated</div>
                </div>
                <div style="font-size:28px;opacity:0.15;">📄</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Streak badge ──────────────────────────────────────────────────────
    if _app_count > 0:
        st.markdown(
            f'<div style="padding:0 8px;">'
            f'<span class="streak-badge">🔥 {_app_count} application{"s" if _app_count != 1 else ""} generated</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Daily tip ─────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="sidebar-tip">'
        f'<div class="sidebar-tip-lbl">💡 Pro Tip</div>'
        f'<div class="sidebar-tip-txt">{_daily_tip()}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
    if st.button("Sign out", use_container_width=True):
        st.session_state.clear()
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE PAGE
# ─────────────────────────────────────────────────────────────────────────────
if "Generate" in page:

    st.markdown(f"""
    <div style="margin-bottom:32px;">
        <div class="greeting-line">{_greeting(_user.get('name',''))}</div>
        <div class="grad-title" style="font-size:30px;font-weight:900;letter-spacing:-0.025em;
                    line-height:1.15;margin-bottom:10px;">
            Build Your Application
        </div>
        <div style="font-size:14.5px;color:#64748b;line-height:1.65;max-width:600px;">
            Paste a job posting and your resume — get a tailored cover letter, optimized resume,
            LinkedIn outreach, company intel, and full interview prep in ~45 seconds.
        </div>
    </div>
    """, unsafe_allow_html=True)

    left_col, right_col = st.columns([3, 2], gap="large")

    with left_col:
        # Step 1 — Resume
        with st.container(border=True):
            st.markdown("""
            <div class="step-header">
                <span class="step-num">1</span>
                <span class="step-title">Resume</span>
            </div>
            """, unsafe_allow_html=True)

            uploaded_resume  = st.file_uploader("Resume PDF", type=["pdf"], label_visibility="collapsed")
            default_resume   = db.load_default_resume(_user_id)
            use_saved        = st.checkbox("Use saved resume", value=bool(default_resume))
            resume_text      = resume_name = None

            if uploaded_resume:
                file_bytes       = uploaded_resume.read()
                resume_text      = extract_text_from_pdf(BytesIO(file_bytes))
                resume_name      = uploaded_resume.name
                st.session_state["resume_name"]  = resume_name
                st.session_state["resume_style"] = extract_pdf_style(file_bytes)
                db.save_default_resume(_user_id, resume_text, resume_name)
            elif use_saved and default_resume:
                resume_text = default_resume["text"]
                resume_name = default_resume["name"]

            if resume_text:
                st.markdown(
                    f'<div style="background:#eff6ff;border:1.5px solid #bfdbfe;border-radius:8px;'
                    f'padding:10px 14px;color:#1d4ed8;font-weight:600;font-size:13px;margin-top:4px;">'
                    f'📄 &nbsp;{resume_name}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div style="background:#fffbeb;border:1.5px solid #fde68a;border-radius:8px;'
                    'padding:10px 14px;color:#92400e;font-size:13px;margin-top:4px;">'
                    '⚠️ &nbsp;Upload a PDF or enable "Use saved resume"</div>',
                    unsafe_allow_html=True,
                )

        # Step 2 — Job Description
        with st.container(border=True):
            st.markdown("""
            <div class="step-header">
                <span class="step-num">2</span>
                <span class="step-title">Job Description</span>
            </div>
            """, unsafe_allow_html=True)

            input_mode    = st.radio("Input type", ["Paste Text", "Paste URL"],
                                     horizontal=True, label_visibility="collapsed")
            job_description = None

            if input_mode == "Paste Text":
                job_description = st.text_area(
                    "Job description",
                    height=200,
                    placeholder="Paste the full job description here — the more detail, the better the output…",
                    label_visibility="collapsed",
                )
                if job_description and job_description.strip():
                    _jd_words = len(job_description.split())
                    _jd_color = "#15803d" if _jd_words >= 50 else "#b45309"
                    _jd_bg    = "#f0fdf4" if _jd_words >= 50 else "#fefce8"
                    _jd_bdr   = "#86efac" if _jd_words >= 50 else "#fde68a"
                    _jd_icon  = "✓" if _jd_words >= 50 else "⚠"
                    _jd_note  = "ready to generate" if _jd_words >= 50 else "try pasting more of the posting for best results"
                    st.markdown(
                        f'<div style="font-size:12.5px;color:{_jd_color};background:{_jd_bg};'
                        f'border:1px solid {_jd_bdr};border-radius:8px;padding:7px 12px;margin-top:4px;">'
                        f'{_jd_icon} {_jd_words:,} words detected — {_jd_note}.</div>',
                        unsafe_allow_html=True,
                    )
            else:
                job_url = st.text_input(
                    "Job URL",
                    placeholder="https://jobs.lever.co/company/role-id",
                    label_visibility="collapsed",
                )
                if job_url and job_url.strip():
                    st.markdown(
                        '<div style="font-size:12.5px;color:#15803d;background:#f0fdf4;'
                        'border:1px solid #86efac;border-radius:8px;padding:7px 12px;margin-top:4px;">'
                        '✓ URL detected — job description will be fetched automatically when you generate.</div>',
                        unsafe_allow_html=True,
                    )

    with right_col:
        # Step 3 — Recruiter (optional)
        with st.container(border=True):
            st.markdown("""
            <div class="step-header">
                <span class="step-num">3</span>
                <span class="step-title">Recruiter LinkedIn
                    <span style="margin-left:8px;font-size:11px;font-weight:500;color:#64748b;
                                 background:#f1f5f9;border-radius:99px;padding:2px 9px;">optional</span>
                </span>
            </div>
            <div style="font-size:13px;color:#64748b;margin-bottom:14px;line-height:1.6;">
                Paste the recruiter's LinkedIn URL to get a message personalized with genuine
                shared connections — school, employer, location, interests.
            </div>
            """, unsafe_allow_html=True)

            recruiter_url = st.text_input(
                "LinkedIn URL",
                placeholder="https://www.linkedin.com/in/their-name/",
                label_visibility="collapsed",
                key="recruiter_url",
            )
            recruiter_profile_text = ""

            if recruiter_url and recruiter_url.strip():
                st.markdown(
                    '<div style="font-size:12.5px;color:#15803d;background:#f0fdf4;'
                    'border:1px solid #86efac;border-radius:8px;padding:7px 12px;margin-top:4px;">'
                    '✓ Profile will be scraped and analyzed automatically.</div>',
                    unsafe_allow_html=True,
                )

        # Step 4 — Additional Materials (optional)
        with st.container(border=True):
            st.markdown("""
            <div class="step-header">
                <span class="step-num">4</span>
                <span class="step-title">Additional Materials
                    <span style="margin-left:8px;font-size:11px;font-weight:500;color:#64748b;
                                 background:#f1f5f9;border-radius:99px;padding:2px 9px;">optional</span>
                </span>
            </div>
            <div style="font-size:13px;color:#64748b;margin-bottom:14px;line-height:1.6;">
                Paste or upload anything else that supports your candidacy — performance reviews,
                notable project write-ups, LinkedIn About section, personal projects, awards, etc.
                Everything here is scanned and used to strengthen your fit score, resume, cover letter,
                and interview prep.
            </div>
            """, unsafe_allow_html=True)

            extra_file = st.file_uploader(
                "Upload PDF",
                type=["pdf"],
                label_visibility="collapsed",
                key="extra_materials_file",
            )
            extra_text = st.text_area(
                "Or paste text",
                height=120,
                placeholder="Paste performance review highlights, project summaries, LinkedIn About, personal projects…",
                label_visibility="collapsed",
                key="extra_materials_text",
            )

            additional_context = ""
            if extra_file:
                additional_context += extract_text_from_pdf(BytesIO(extra_file.read()))
            if extra_text and extra_text.strip():
                additional_context = (additional_context + "\n\n" + extra_text).strip()

            if additional_context:
                _ac_words = len(additional_context.split())
                st.markdown(
                    f'<div style="font-size:12.5px;color:#15803d;background:#f0fdf4;'
                    f'border:1px solid #86efac;border-radius:8px;padding:7px 12px;margin-top:4px;">'
                    f'✓ {_ac_words:,} words of additional context will be included.</div>',
                    unsafe_allow_html=True,
                )

        # Step 5 — Generate
        with st.container(border=True):
            st.markdown("""
            <div class="step-header">
                <span class="step-num">5</span>
                <span class="step-title">Generate Everything</span>
            </div>
            <div style="font-size:13px;color:#64748b;margin-bottom:16px;line-height:1.6;">
                Takes ~45 seconds. Produces a fit score, tailored resume, cover letter,
                LinkedIn message, company research, and full interview prep — all in one run.
            </div>
            """, unsafe_allow_html=True)

            has_recruiter = bool(recruiter_url and recruiter_url.strip())

            st.markdown('<div class="gen-cta">', unsafe_allow_html=True)
            _do_gen = st.button("⚡  Generate My Application", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            score = ""  # pre-init so it's always defined in this scope
            if _do_gen:
                if input_mode == "Paste URL":
                    if not job_url:
                        st.error("Enter a job URL before generating.")
                        st.stop()
                    with st.spinner("Fetching job posting…"):
                        job_description = extract_job_from_url(job_url)
                    # Guard: require at least 50 meaningful words (error pages / nav boilerplate fail this)
                    _jd_word_count = len((job_description or "").split())
                    if _jd_word_count < 50:
                        st.error(
                            "⚠️ No job description could be scraped from that URL — "
                            "the page may require a login or block scrapers. "
                            "Please copy and paste the job description text directly using **Paste Text** mode."
                        )
                        st.stop()

                if not resume_text:
                    st.error("Upload a resume or enable 'Use saved resume'.")
                    st.stop()
                if not job_description or not job_description.strip():
                    st.error("Paste a job description or URL.")
                    st.stop()

                resolved_recruiter = ""
                if recruiter_url and recruiter_url.strip():
                    with st.spinner("Fetching recruiter profile…"):
                        try:
                            resolved_recruiter = extract_job_from_url(recruiter_url)
                        except Exception:
                            st.warning(
                                "LinkedIn blocked the profile scrape — personalization skipped."
                            )

                _prog = st.empty()
                _steps_log: list[str] = []

                def _on_step(msg: str) -> None:
                    _steps_log.append(msg)
                    try:
                        _prog.markdown(
                            f'<div class="gen-progress">'
                            f'<div class="gen-progress-dot"></div>'
                            f'<div><strong>Step {len(_steps_log)} of 10</strong>'
                            f'{"&nbsp;·&nbsp; (incl. additional materials)" if additional_context and len(_steps_log) == 1 else ""}'
                            f' &nbsp;·&nbsp; {msg}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    except Exception:
                        pass

                result = run_agent(
                    job_description,
                    resume_text,
                    recruiter_profile=resolved_recruiter,
                    additional_context=additional_context,
                    on_step=_on_step,
                )
                _prog.empty()

                db.save_application(_user_id, job_description, resume_text, result)
                st.session_state["result"] = result

                company = (result.get("cover_letter_strategy") or {}).get("company_name", "")
                score   = result.get("fit_score", "")
                color, _, label = _fit_colors(int(score)) if score else ("#64748b", "", "")
                _done_bg   = "linear-gradient(135deg,#f0fdf4,#dcfce7)" if int(score or 0) >= 8 else "#f0fdf4"
                _done_icon = "🎉" if int(score or 0) >= 8 else "✅"
                st.markdown(
                    f'<div style="background:{_done_bg};border:1.5px solid #86efac;'
                    f'border-radius:12px;padding:14px 18px;font-size:14px;font-weight:700;'
                    f'color:#15803d;margin-top:4px;display:flex;align-items:center;gap:10px;'
                    f'animation:cel-slide 0.4s ease both;">'
                    f'<span style="font-size:20px;">{_done_icon}</span>'
                    f'<span>{"<strong>" + company + "</strong> — " if company else ""}'
                    f'<span style="color:{color};">{label} · {score}/10</span>'
                    f'&nbsp;<span style="font-size:12px;font-weight:500;color:#166534;">'
                    f'Materials ready below ↓</span>'
                    f'</span></div>',
                    unsafe_allow_html=True,
                )

    # Results
    result = st.session_state.get("result")
    if result:
        company_name = (result.get("cover_letter_strategy") or {}).get("company_name", "")
        score_v = result.get("fit_score", "")
        score_color, _, score_label = _fit_colors(int(score_v)) if score_v else ("#94a3b8", "", "")

        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:14px;margin-bottom:4px;">'
            f'<div style="font-size:22px;font-weight:900;color:#0f172a;letter-spacing:-0.02em;">'
            + (f"{company_name} Application" if company_name else "Application Materials")
            + f'</div>'
            + (f'<div style="background:{score_color}18;color:{score_color};'
               f'border:1.5px solid {score_color}30;border-radius:99px;padding:4px 14px;'
               f'font-size:13px;font-weight:700;">{score_label} · {score_v}/10</div>'
               if score_v else "")
            + f'</div>',
            unsafe_allow_html=True,
        )
        render_result(result, key_suffix="gen")


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD PAGE
# ─────────────────────────────────────────────────────────────────────────────
elif "Dashboard" in page:

    st.markdown("""
    <div style="margin-bottom:32px;">
        <div class="grad-title" style="font-size:28px;font-weight:900;letter-spacing:-0.02em;
                    line-height:1.15;margin-bottom:8px;">Application Tracker</div>
        <div style="font-size:14.5px;color:#64748b;line-height:1.6;">
            Every application you've generated — scored, tracked, and ready to review.
        </div>
    </div>
    """, unsafe_allow_html=True)

    apps = db.load_applications(_user_id)

    if not apps:
        st.markdown("""
        <div style="text-align:center;padding:80px 20px;background:#fff;border-radius:16px;
                    border:1.5px solid #e2e8f0;box-shadow:0 1px 4px rgba(0,0,0,0.05);">
            <div style="font-size:56px;margin-bottom:16px;">🚀</div>
            <div class="grad-title" style="font-size:22px;font-weight:800;letter-spacing:-0.01em;
                        margin-bottom:10px;">Your tracker starts here</div>
            <div style="font-size:14px;color:#64748b;max-width:380px;margin:0 auto 24px;line-height:1.7;">
                Generate your first application and it'll appear here — complete with fit score,
                status tracking, and full materials.
            </div>
            <a href="javascript:void(0)" onclick="
                var radios=window.parent.document.querySelectorAll('input[type=radio]');
                for(var r of radios){{if(r.value&&r.value.includes('Generate')){{r.click();break;}}}}
            " style="display:inline-block;background:linear-gradient(135deg,#4f46e5,#6366f1);
            color:#fff;padding:11px 28px;border-radius:10px;font-weight:700;font-size:14px;
            text-decoration:none;box-shadow:0 2px 8px rgba(79,70,229,0.3);letter-spacing:0.01em;">
                ⚡ Generate Your First Application
            </a>
        </div>
        """, unsafe_allow_html=True)
    else:
        scores = [
            e.get("result", {}).get("fit_score", 0)
            for e in apps
            if e.get("result", {}).get("fit_score") is not None
        ]
        avg_score  = round(sum(scores) / len(scores), 1) if scores else 0
        best_score = max(scores) if scores else 0
        best_entry = next(
            (e for e in reversed(apps) if e.get("result", {}).get("fit_score") == best_score), None
        )
        best_company = (
            (best_entry.get("result", {}).get("cover_letter_strategy") or {}).get("company_name", "—")
            if best_entry else "—"
        )
        strong_count = sum(1 for s in scores if s >= 8)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Applications", len(apps))
        m2.metric("Avg Fit Score", f"{avg_score}/10")
        m3.metric("Strong Matches", strong_count)
        m4.metric("Best Match", f"{best_company}")

        st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

        # Build dataframe
        rows = []
        for pos, entry in enumerate(apps):
            r        = entry.get("result", {})
            strategy = r.get("cover_letter_strategy", {})
            ts       = entry.get("timestamp", "")
            score    = int(r.get("fit_score") or 0)
            _, _, lbl = _fit_colors(score)
            rows.append({
                "_pos":    pos,
                "_app_id": entry.get("id", ""),
                "Company": strategy.get("company_name", "Unknown"),
                "Status":  entry.get("status", "Applied"),
                "Score":   score,
                "Match":   lbl,
                "Date":    ts[:10] if ts else "—",
            })
        df = pd.DataFrame(rows)

        _STATUS_COLORS = {
            "Applied":      ("#4f46e5", "#eef2ff"),
            "Phone Screen": ("#0891b2", "#ecfeff"),
            "Interview":    ("#7c3aed", "#f5f3ff"),
            "Offer":        ("#15803d", "#f0fdf4"),
            "Rejected":     ("#dc2626", "#fef2f2"),
            "Declined":     ("#64748b", "#f8fafc"),
            "Archived":     ("#94a3b8", "#f8fafc"),
        }

        # Filter bar
        st.markdown(
            '<div style="background:#fff;border:1.5px solid #e2e8f0;border-radius:12px;'
            'padding:14px 18px 8px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,0.04);">',
            unsafe_allow_html=True,
        )
        fc1, fc2, fc3 = st.columns([3, 2, 2])
        with fc1:
            search = st.text_input("Search", placeholder="🔍  Filter by company…",
                                   label_visibility="collapsed")
        with fc2:
            match_filter = st.multiselect("Match", options=["Strong Match", "Solid Match", "Stretch Role"],
                                          placeholder="All match levels", label_visibility="collapsed")
        with fc3:
            min_score = st.slider("Min fit score", 1, 10, 1)
        st.markdown("</div>", unsafe_allow_html=True)

        mask = pd.Series([True] * len(df))
        if search:
            mask &= df["Company"].str.contains(search, case=False, na=False)
        if match_filter:
            mask &= df["Match"].isin(match_filter)
        mask &= df["Score"] >= min_score

        filtered = df[mask].reset_index(drop=True)
        display  = filtered.drop(columns=["_pos", "_app_id"])

        def _style_score(val: int) -> str:
            if val >= 8:
                return "background:#dcfce7;color:#15803d;font-weight:700;"
            if val >= 6:
                return "background:#fef9c3;color:#b45309;font-weight:700;"
            return "background:#fee2e2;color:#dc2626;font-weight:700;"

        if filtered.empty:
            st.info("No applications match the current filters.")
        else:
            styled = display.style.map(_style_score, subset=["Score"])
            event = st.dataframe(
                styled,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                column_config={
                    "Company": st.column_config.TextColumn("Company", width="medium"),
                    "Status":  st.column_config.TextColumn("Status",  width="small"),
                    "Score":   st.column_config.NumberColumn("Fit Score", format="%d /10"),
                    "Match":   st.column_config.TextColumn("Match Level", width="medium"),
                    "Date":    st.column_config.TextColumn("Date", width="small"),
                },
            )
            st.markdown(
                '<div style="font-size:12px;color:#94a3b8;margin-top:6px;">'
                'Click a column header to sort · Click any row to view full materials'
                '</div>',
                unsafe_allow_html=True,
            )

            if event.selection.rows:
                sel_pos    = int(filtered.iloc[event.selection.rows[0]]["_pos"])
                sel_app_id = str(filtered.iloc[event.selection.rows[0]]["_app_id"])
                entry      = apps[sel_pos]
                r          = entry.get("result", {})
                strategy   = r.get("cover_letter_strategy", {})
                company    = strategy.get("company_name", "Application")
                score      = r.get("fit_score", "—")
                cur_status = entry.get("status", "Applied")
                cur_notes  = entry.get("notes", "")
                s_color, s_bg = _STATUS_COLORS.get(cur_status, ("#64748b", "#f8fafc"))
                f_color, _, f_label = (
                    _fit_colors(int(score)) if isinstance(score, (int, float)) else ("#94a3b8","","")
                )

                st.divider()

                hdr_c1, hdr_c2 = st.columns([3, 2])
                with hdr_c1:
                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;'
                        f'margin-bottom:8px;">'
                        f'<div style="font-size:22px;font-weight:900;color:#0f172a;'
                        f'letter-spacing:-0.02em;">{company}</div>'
                        f'<div style="background:{f_color}18;color:{f_color};'
                        f'border:1.5px solid {f_color}28;border-radius:99px;padding:4px 14px;'
                        f'font-size:12.5px;font-weight:700;">{f_label} · {score}/10</div>'
                        f'{_status_pill_html(cur_status)}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with hdr_c2:
                    with st.form(f"status_form_{sel_pos}"):
                        new_status = st.selectbox(
                            "Update status",
                            options=db.APP_STATUSES,
                            index=db.APP_STATUSES.index(cur_status)
                            if cur_status in db.APP_STATUSES else 0,
                            label_visibility="collapsed",
                        )
                        if st.form_submit_button("Save status", use_container_width=True):
                            db.update_application_status(sel_app_id, _user_id, new_status)
                            st.rerun()

                with st.expander("📝 Notes", expanded=bool(cur_notes)):
                    with st.form(f"notes_form_{sel_pos}"):
                        new_notes = st.text_area(
                            "Notes", value=cur_notes, height=90,
                            placeholder="Interview notes, feedback, contacts, next steps…",
                            label_visibility="collapsed",
                        )
                        if st.form_submit_button("Save notes", use_container_width=True):
                            db.update_application_notes(sel_app_id, _user_id, new_notes)
                            st.rerun()

                render_result(r, key_suffix=f"dash_{sel_pos}")


# ─────────────────────────────────────────────────────────────────────────────
# JOBS PAGE
# ─────────────────────────────────────────────────────────────────────────────
elif "Jobs" in page:

    st.markdown("""
    <div style="margin-bottom:32px;">
        <div class="grad-title" style="font-size:28px;font-weight:900;letter-spacing:-0.02em;
                    line-height:1.15;margin-bottom:8px;">Job Recommendations</div>
        <div style="font-size:14.5px;color:#64748b;line-height:1.6;">
            Upload your resume and we'll scan LinkedIn for the best-fit open roles,
            score each one, and explain why it's a match or a stretch.
        </div>
    </div>
    """, unsafe_allow_html=True)

    jleft, jright = st.columns([3, 2], gap="large")

    with jleft:
        with st.container(border=True):
            st.markdown("""
            <div class="step-header">
                <span class="step-num">1</span>
                <span class="step-title">Resume</span>
            </div>
            """, unsafe_allow_html=True)

            jobs_uploaded = st.file_uploader("Resume PDF", type=["pdf"], label_visibility="collapsed",
                                              key="jobs_resume_upload")
            jobs_default  = db.load_default_resume(_user_id)
            jobs_use_saved = st.checkbox("Use saved resume", value=bool(jobs_default), key="jobs_use_saved")

            jobs_resume_text = jobs_resume_name = None
            if jobs_uploaded:
                jobs_resume_text = extract_text_from_pdf(jobs_uploaded)
                jobs_resume_name = jobs_uploaded.name
            elif jobs_use_saved and jobs_default:
                jobs_resume_text = jobs_default["text"]
                jobs_resume_name = jobs_default["name"]

            if jobs_resume_text:
                st.markdown(
                    f'<div style="background:#eff6ff;border:1.5px solid #bfdbfe;border-radius:8px;'
                    f'padding:10px 14px;color:#1d4ed8;font-weight:600;font-size:13px;margin-top:4px;">'
                    f'📄 &nbsp;{jobs_resume_name}</div>',
                    unsafe_allow_html=True,
                )

    with jright:
        with st.container(border=True):
            st.markdown("""
            <div class="step-header">
                <span class="step-num">2</span>
                <span class="step-title">Location <span style="font-size:11px;font-weight:500;
                color:#64748b;background:#f1f5f9;border-radius:99px;padding:2px 9px;
                margin-left:6px;">optional</span></span>
            </div>
            """, unsafe_allow_html=True)
            jobs_location = st.text_input(
                "Location", placeholder="New York, NY  ·  Remote  ·  San Francisco",
                label_visibility="collapsed", key="jobs_location",
            )

        with st.container(border=True):
            st.markdown("""
            <div class="step-header">
                <span class="step-num">3</span>
                <span class="step-title">Find Jobs</span>
            </div>
            """, unsafe_allow_html=True)

            if st.button("🔍  Scan LinkedIn for Best-Fit Roles"):
                if not jobs_resume_text:
                    st.error("Upload a resume or enable 'Use saved resume'.")
                    st.stop()
                with st.spinner("Analyzing your profile…"):
                    jobs_profile = generate_job_search_profile(jobs_resume_text)
                st.session_state["jobs_profile"] = jobs_profile

                with st.spinner("Scanning LinkedIn (may take ~30 seconds)…"):
                    scraped = search_linkedin_jobs(
                        jobs_profile.get("search_queries", []),
                        location=jobs_location,
                    )
                st.session_state["jobs_scraped"] = scraped

                with st.spinner("Scoring and ranking listings…"):
                    scored_jobs = parse_and_score_jobs(jobs_resume_text, scraped)
                st.session_state["jobs_scored"] = scored_jobs

    # Results
    jobs_profile = st.session_state.get("jobs_profile")
    jobs_scraped = st.session_state.get("jobs_scraped")
    scored_jobs  = st.session_state.get("jobs_scored")

    if jobs_profile:
        st.divider()
        titles = jobs_profile.get("target_titles") or []
        skills = jobs_profile.get("key_skills") or []
        level  = jobs_profile.get("experience_level", "")

        prof_c1, prof_c2 = st.columns(2)
        with prof_c1:
            _section_label("Target Roles Detected", "sl-indigo")
            st.markdown(_tag_html(titles), unsafe_allow_html=True)
        with prof_c2:
            _section_label("Key Skills", "sl-slate")
            st.markdown(_tag_html(skills[:12]), unsafe_allow_html=True)
        if level:
            st.markdown(
                f'<div style="font-size:13px;color:#64748b;margin:8px 0 4px;">'
                f'Experience level detected: <strong style="color:#334155;">{level}</strong></div>',
                unsafe_allow_html=True,
            )

    if jobs_scraped and any(r.get("blocked") for r in jobs_scraped):
        with st.expander("⚠️ LinkedIn blocked some searches — open manually"):
            for r in jobs_scraped:
                if r.get("blocked"):
                    st.markdown(f"- [{r['query']}]({r['url']})")

    if scored_jobs:
        st.markdown(
            f'<div style="font-size:20px;font-weight:800;color:#0f172a;letter-spacing:-0.02em;'
            f'margin:24px 0 16px;">Best-Fit Openings</div>',
            unsafe_allow_html=True,
        )

        def _fit_badge(score: int) -> str:
            if score >= 8:
                return (f'<span style="background:#dcfce7;color:#15803d;font-weight:700;'
                        f'padding:3px 10px;border-radius:99px;font-size:12px;">{score}/10</span>')
            if score >= 6:
                return (f'<span style="background:#fef9c3;color:#b45309;font-weight:700;'
                        f'padding:3px 10px;border-radius:99px;font-size:12px;">{score}/10</span>')
            return (f'<span style="background:#fee2e2;color:#dc2626;font-weight:700;'
                    f'padding:3px 10px;border-radius:99px;font-size:12px;">{score}/10</span>')

        def _salary_cell(j: dict) -> str:
            sal   = j.get("salary_range", "") or ""
            src   = j.get("salary_source", "") or ""
            basis = j.get("salary_estimate_basis", "") or ""
            if not sal:
                return '<span style="color:#94a3b8;font-size:12px;">—</span>'
            if src == "listed":
                return f'<span style="color:#15803d;font-weight:600;">{sal}</span>'
            tip = basis or "Market estimate based on role, level & location"
            return (
                f'<span title="{tip}" style="color:#6366f1;font-weight:600;cursor:help;">{sal}</span>'
                f'<sup title="{tip}" style="color:#94a3b8;font-size:10px;cursor:help;">est.</sup>'
            )

        def _work_cell(j: dict) -> str:
            w = j.get("work_arrangement", "") or ""
            if w.lower().startswith("remote"):
                return f'<span style="color:#0891b2;font-size:12px;">🌐 {w}</span>'
            if w.lower().startswith("hybrid"):
                return f'<span style="color:#7c3aed;font-size:12px;">🏢 {w}</span>'
            if w.lower().startswith("on"):
                return f'<span style="color:#b45309;font-size:12px;">🏢 {w}</span>'
            return f'<span style="color:#94a3b8;font-size:12px;">{w or "—"}</span>'

        def _fit_reasons_cell(j: dict) -> str:
            reasons = j.get("top_fit_reasons") or []
            gaps    = j.get("top_gap_reasons") or []
            parts   = []
            for r in reasons[:2]:
                parts.append(f'<span style="color:#15803d;">✓</span> {r}')
            for g in gaps[:1]:
                parts.append(f'<span style="color:#dc2626;">✗</span> {g}')
            if not parts:
                return f'<span style="color:#64748b;font-size:12px;">{j.get("fit_reason","")}</span>'
            return "<br>".join(
                f'<span style="font-size:12px;color:#334155;line-height:1.7;">{p}</span>'
                for p in parts
            )

        header_html = (
            '<div style="border:1.5px solid #e2e8f0;border-radius:12px;overflow:hidden;'
            'overflow-x:auto;box-shadow:0 1px 4px rgba(0,0,0,0.05);">'
            '<table class="job-table">'
            '<thead><tr>'
            '<th>Company</th><th>Role</th><th>Fit</th>'
            '<th>Salary</th><th>Work</th><th>Why It Fits</th>'
            '</tr></thead><tbody>'
        )
        rows_html = ""
        for i, j in enumerate(scored_jobs):
            title   = j.get("title", "—")
            company = j.get("company", "")
            url     = j.get("apply_url", "") or ""
            if not url:
                url = (f"https://www.linkedin.com/jobs/search/?"
                       f"keywords={quote_plus(title + ' ' + company)}")
            role_cell = (
                f'<a href="{url}" target="_blank" style="color:#4f46e5;font-weight:600;'
                f'text-decoration:none;">{title} ↗</a>'
            )
            bg = "#ffffff" if i % 2 == 0 else "#fafbfd"
            rows_html += (
                f'<tr style="background:{bg};vertical-align:top;">'
                f'<td style="padding:10px 14px;color:#334155;font-weight:500;'
                f'white-space:nowrap;">{company or "—"}</td>'
                f'<td style="padding:10px 14px;">{role_cell}</td>'
                f'<td style="padding:10px 14px;white-space:nowrap;">'
                f'{_fit_badge(int(j.get("fit_score") or 0))}</td>'
                f'<td style="padding:10px 14px;white-space:nowrap;">{_salary_cell(j)}</td>'
                f'<td style="padding:10px 14px;white-space:nowrap;">{_work_cell(j)}</td>'
                f'<td style="padding:10px 14px;max-width:300px;">{_fit_reasons_cell(j)}</td>'
                f'</tr>'
            )
        st.markdown(
            header_html + rows_html + '</tbody></table></div>',
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
        job_options = ["— select a role to view details —"] + [
            f"{j.get('title','—')} @ {j.get('company','—')}" for j in scored_jobs
        ]
        selected_label = st.selectbox(
            "View details", options=job_options, label_visibility="collapsed",
            key="jobs_detail_select",
        )

        if selected_label != job_options[0]:
            sel_idx = job_options.index(selected_label) - 1
            job     = scored_jobs[sel_idx]
            jc, jbg, jl = _fit_colors(int(job.get("fit_score") or 0))

            st.divider()
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:14px;margin-bottom:16px;flex-wrap:wrap;">'
                f'<div style="font-size:20px;font-weight:800;color:#0f172a;letter-spacing:-0.01em;">'
                f'{job.get("title","Role")} — {job.get("company","")}</div>'
                f'<div style="background:{jc}18;color:{jc};border:1.5px solid {jc}28;'
                f'border-radius:99px;padding:4px 14px;font-size:13px;font-weight:700;">'
                f'{jl} · {job.get("fit_score","—")}/10</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            salary  = job.get("salary_range", "") or ""
            sal_src = job.get("salary_source", "") or ""
            sal_bas = job.get("salary_estimate_basis", "") or ""
            work    = job.get("work_arrangement", "") or ""
            location = job.get("location", "") or ""
            posted  = job.get("posted", "") or ""
            why     = job.get("fit_reason", "")
            fit_r   = job.get("top_fit_reasons") or []
            gap_r   = job.get("top_gap_reasons") or []
            matches = job.get("matching_skills") or []
            gaps    = job.get("gaps") or []
            url_j   = job.get("apply_url", "")

            # Meta strip
            meta_parts = []
            if salary:
                if sal_src == "listed":
                    meta_parts.append(
                        f'<span style="color:#15803d;font-weight:700;">{salary}</span>'
                        f'<span style="font-size:11px;color:#15803d;margin-left:4px;">listed</span>'
                    )
                else:
                    tip = sal_bas or "Market estimate based on role, level & location"
                    meta_parts.append(
                        f'<span title="{tip}" style="color:#6366f1;font-weight:700;cursor:help;">'
                        f'{salary}</span>'
                        f'<span title="{tip}" style="font-size:11px;color:#6366f1;margin-left:4px;'
                        f'cursor:help;">Zestimate™</span>'
                    )
            if work:
                meta_parts.append(f'<span style="color:#475569;">🏢 {work}</span>')
            if location:
                meta_parts.append(f'<span style="color:#64748b;">📍 {location}</span>')
            if posted:
                meta_parts.append(f'<span style="color:#94a3b8;">🕒 {posted}</span>')
            if meta_parts:
                st.markdown(
                    '<div style="display:flex;flex-wrap:wrap;gap:18px;font-size:13px;'
                    'background:#fff;border:1.5px solid #e2e8f0;border-radius:10px;'
                    'padding:12px 18px;margin-bottom:14px;box-shadow:0 1px 3px rgba(0,0,0,0.04);">'
                    + " · ".join(meta_parts) + "</div>",
                    unsafe_allow_html=True,
                )

            if sal_src != "listed" and sal_bas and salary:
                st.markdown(
                    f'<div style="font-size:12px;color:#6366f1;background:#eef2ff;'
                    f'border:1px solid #c7d2fe;border-radius:8px;padding:7px 12px;margin-bottom:12px;">'
                    f'💡 Salary Zestimate basis: {sal_bas}</div>',
                    unsafe_allow_html=True,
                )

            if why:
                st.markdown(
                    f'<div style="font-size:14px;line-height:1.75;color:#1e293b;background:#fff;'
                    f'border:1.5px solid #e2e8f0;border-radius:10px;padding:14px 18px;'
                    f'margin-bottom:14px;box-shadow:0 1px 3px rgba(0,0,0,0.04);">{why}</div>',
                    unsafe_allow_html=True,
                )

            if fit_r or gap_r:
                dc1, dc2 = st.columns(2)
                with dc1:
                    if fit_r:
                        _section_label("Why It's a Strong Fit", "sl-green")
                        for r_item in fit_r:
                            st.markdown(
                                f'<div style="font-size:13px;color:#15803d;line-height:1.65;'
                                f'margin-bottom:5px;">✓ {r_item}</div>',
                                unsafe_allow_html=True,
                            )
                with dc2:
                    if gap_r:
                        _section_label("Honest Gaps / Concerns", "sl-amber")
                        for g_item in gap_r:
                            st.markdown(
                                f'<div style="font-size:13px;color:#b91c1c;line-height:1.65;'
                                f'margin-bottom:5px;">✗ {g_item}</div>',
                                unsafe_allow_html=True,
                            )

            sk_c1, sk_c2 = st.columns(2)
            with sk_c1:
                if matches:
                    _section_label("Matching Skills", "sl-green")
                    st.markdown(_tag_html(matches), unsafe_allow_html=True)
            with sk_c2:
                if gaps:
                    _section_label("Skill Gaps", "sl-amber")
                    st.markdown(_tag_html(gaps), unsafe_allow_html=True)

            if url_j:
                st.markdown(
                    f'<a href="{url_j}" target="_blank" style="display:inline-block;margin-top:16px;'
                    f'background:linear-gradient(135deg,#4f46e5,#6366f1);color:#fff;'
                    f'padding:10px 22px;border-radius:10px;font-weight:600;font-size:14px;'
                    f'text-decoration:none;box-shadow:0 2px 8px rgba(79,70,229,0.3);">'
                    f'View & Apply on LinkedIn ↗</a>',
                    unsafe_allow_html=True,
                )

    elif jobs_profile and not scored_jobs:
        st.markdown(
            '<div style="text-align:center;padding:50px 20px;background:#fff;border-radius:14px;'
            'border:1.5px solid #e2e8f0;box-shadow:0 1px 4px rgba(0,0,0,0.05);margin-top:20px;">'
            '<div style="font-size:44px;margin-bottom:12px;">🔍</div>'
            '<div style="font-size:16px;font-weight:700;color:#334155;">No scoreable results yet</div>'
            '<div style="font-size:13px;color:#64748b;margin-top:6px;">'
            'LinkedIn may have blocked the search — try the links above.</div>'
            '</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# ACCOUNT PAGE
# ─────────────────────────────────────────────────────────────────────────────
elif "Account" in page:

    st.markdown(f"""
    <div style="margin-bottom:32px;">
        <div class="grad-title" style="font-size:28px;font-weight:900;letter-spacing:-0.02em;
                    line-height:1.15;margin-bottom:8px;">Account Settings</div>
        <div style="font-size:14.5px;color:#64748b;line-height:1.6;">
            Manage your profile, security, resume, and data.
        </div>
    </div>
    """, unsafe_allow_html=True)

    acc_tab1, acc_tab2, acc_tab3, acc_tab4 = st.tabs(
        ["👤  Profile", "🔒  Security", "📦  My Data", "🗑️  Danger Zone"]
    )

    with acc_tab1:
        st.markdown("#### Display name")
        with st.form("rename_form"):
            new_name = st.text_input("Name", value=_user.get("name", ""))
            if st.form_submit_button("Save name", use_container_width=True):
                try:
                    auth_module.rename_user(_user_id, new_name)
                    st.session_state["user"]["name"] = new_name.strip()
                    st.success("Name updated.")
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))

        st.markdown("---")
        st.markdown("#### Account info")
        last    = _user.get("last_login") or "—"
        created = _user.get("created_at", "")[:10] if _user.get("created_at") else "—"
        st.markdown(
            f'<div style="font-size:13.5px;color:#475569;line-height:2.2;">'
            f'<b>Email:</b> {_user.get("email","")}<br>'
            f'<b>Member since:</b> {created}<br>'
            f'<b>Last sign-in:</b> {last[:19].replace("T"," ") if last != "—" else "—"}'
            f'</div>',
            unsafe_allow_html=True,
        )

    with acc_tab2:
        st.markdown("#### Change password")
        with st.form("change_pw_form"):
            old_pw  = st.text_input("Current password", type="password")
            new_pw  = st.text_input("New password (8+ chars)", type="password")
            conf_pw = st.text_input("Confirm new password", type="password")
            if st.form_submit_button("Update password", use_container_width=True):
                try:
                    auth_module.change_password(_user_id, old_pw, new_pw, conf_pw)
                    st.success("Password changed successfully.")
                except ValueError as e:
                    st.error(str(e))

    with acc_tab3:
        st.markdown("#### Your applications")
        app_count = db.application_count(_user_id)
        st.markdown(
            f'<div style="font-size:14px;color:#1e293b;margin-bottom:16px;">'
            f'You have <strong>{app_count}</strong> saved application{"s" if app_count != 1 else ""}.</div>',
            unsafe_allow_html=True,
        )
        if app_count:
            export_data = db.export_user_data(_user_id)
            st.download_button(
                "⬇️  Export all my data (JSON)",
                data=json.dumps(export_data, indent=2, ensure_ascii=False),
                file_name="jobagent_my_data.json",
                mime="application/json",
            )

        st.markdown("---")
        st.markdown("#### Saved resume")
        saved_resume = db.load_default_resume(_user_id)
        if saved_resume:
            st.markdown(
                f'<div style="background:#eff6ff;border:1.5px solid #bfdbfe;border-radius:10px;'
                f'padding:12px 16px;color:#1d4ed8;font-size:13.5px;">'
                f'📄 &nbsp;<strong>{saved_resume["name"]}</strong></div>',
                unsafe_allow_html=True,
            )
            with st.expander("Preview saved resume text"):
                st.text(saved_resume["text"][:2000]
                        + ("…" if len(saved_resume["text"]) > 2000 else ""))
        else:
            st.info("No resume saved yet. Upload a PDF on the Generate page to save it automatically.")

    with acc_tab4:
        st.markdown(
            '<div style="background:#fef2f2;border:1.5px solid #fecaca;border-radius:12px;'
            'padding:22px 26px;margin-bottom:20px;">'
            '<div style="font-size:15px;font-weight:700;color:#dc2626;margin-bottom:6px;">'
            '⚠️  Delete account</div>'
            '<div style="font-size:13.5px;color:#7f1d1d;line-height:1.7;">'
            'This will permanently deactivate your account. Your applications and resume '
            'will be retained for 30 days and then purged. This action cannot be undone.'
            '</div></div>',
            unsafe_allow_html=True,
        )
        with st.form("delete_account_form"):
            del_pw = st.text_input("Confirm your password to delete", type="password")
            if st.form_submit_button("Delete my account", type="primary"):
                try:
                    auth_module.delete_account(_user_id, del_pw)
                    st.session_state.clear()
                    st.success("Account deleted. Signing you out.")
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))
