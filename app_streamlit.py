"""
arXiv 学术论文检索 — Streamlit 前端
暖色学术风格 · TF-IDF / BM25 / Sentence-BERT
"""

import os, sys, json
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
from search_backend import load_jsonl, load_from_database, make_title_abstract_field, SearchEngine
from database.clean_text import clean_text
from ai_search import AISearchEngine, AISearchConfig, DEFAULT_DEEPSEEK_BASE_URL, DEFAULT_DEEPSEEK_MODEL

DEFAULT_DATA = "data/sample_arxiv_20000.jsonl"
DEFAULT_AI_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL)
DEFAULT_AI_MODEL = os.environ.get("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)


def safe_html(value, limit=None):
    text = "" if value is None else str(value)
    if limit is not None:
        text = text[:limit]
    return escape(text, quote=True)

# ============================================================
st.set_page_config(page_title="arXiv Research Search", page_icon=":mag:", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=Noto+Serif+SC:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ============================================================
   DESIGN TOKENS — Research Workspace
   ============================================================ */
:root {
    --bg-root: #f6f9fb;
    --bg-surface: rgba(255,255,255,0.72);
    --bg-elevated: rgba(246,249,251,0.92);
    --border-default: rgba(119,137,153,0.22);
    --border-accent: rgba(79,111,143,0.34);
    --text-primary: #202936;
    --text-secondary: #52606d;
    --text-muted: #7c8792;
    --accent-blue: #4f6f8f;
    --accent-blue-dark: #395872;
    --accent-yellow: #d7c37c;
    --accent-sage: #55766b;
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 14px;
    --shadow-card: 0 14px 38px rgba(53,77,102,0.08);
    --shadow-card-hover: 0 22px 54px rgba(53,77,102,0.14);
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        scroll-behavior: auto !important;
        transition-duration: 0.01ms !important;
    }
}

/* ============================================================
   BASE & TYPOGRAPHY
   ============================================================ */
.stApp {
    background:
        radial-gradient(circle at 12% 4%, rgba(209,226,238,0.72) 0, rgba(209,226,238,0.30) 22%, transparent 46%),
        radial-gradient(circle at 88% 12%, rgba(232,222,184,0.42) 0, rgba(232,222,184,0.20) 20%, transparent 42%),
        linear-gradient(145deg, #fbfcfd 0%, #f4f8fb 42%, #eef4f8 100%);
    background-attachment: fixed;
}
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background-image:
        linear-gradient(rgba(79,111,143,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(79,111,143,0.025) 1px, transparent 1px);
    background-size: 44px 44px;
    mask-image: linear-gradient(to bottom, rgba(0,0,0,0.70), rgba(0,0,0,0.14));
}
[data-testid="stAppViewContainer"],
[data-testid="stHeader"] {
    background: transparent !important;
}
.main .block-container {
    position: relative;
    z-index: 1;
    max-width: 1180px;
}
body, .stMarkdown, .stText {
    font-family: 'IBM Plex Sans', 'Microsoft YaHei', -apple-system, sans-serif;
    color: #202936;
}
code, .mono { font-family: 'JetBrains Mono', monospace; }

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stDeployButton"] {
    display: none !important;
}
[data-testid="stToolbar"] {
    right: 0.5rem !important;
}
[data-testid="stAppDeployButton"] button,
[data-testid="stToolbar"] .stAppDeployButton button {
    height: 28px !important;
    padding: 0 0.8rem !important;
    border-radius: 999px !important;
    border: 1px solid rgba(119,137,153,0.24) !important;
    background: rgba(255,255,255,0.58) !important;
    color: #395872 !important;
    font-family: 'IBM Plex Sans', 'Microsoft YaHei', -apple-system, sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    box-shadow: 0 8px 22px rgba(53,77,102,0.08) !important;
    backdrop-filter: blur(14px);
    transition: background-color 0.16s ease, border-color 0.16s ease, box-shadow 0.16s ease, transform 0.16s ease !important;
}
[data-testid="stAppDeployButton"] button:hover,
[data-testid="stToolbar"] .stAppDeployButton button:hover {
    transform: translateY(-1px);
    background: rgba(255,255,255,0.78) !important;
    border-color: rgba(79,111,143,0.38) !important;
    color: #395872 !important;
    box-shadow: 0 10px 26px rgba(53,77,102,0.11) !important;
}

[data-testid="stMainBlockContainer"],
.main .block-container {
    padding-top: 0.75rem !important;
}

/* ============================================================
   HEADER BAR
   ============================================================ */
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0.65rem;
    z-index: 30;
    padding: 0.8rem 1rem;
    border: 1px solid rgba(119,137,153,0.20);
    border-radius: 12px;
    margin-bottom: 1.05rem;
    background: rgba(255,255,255,0.58);
    backdrop-filter: blur(18px);
    box-shadow: 0 10px 34px rgba(53,77,102,0.06);
}
.app-logo {
    font-family: 'Noto Serif SC', 'IBM Plex Sans', serif;
    font-size: 1.05rem; font-weight: 700; color: #202936;
    letter-spacing: 0.01em;
}
.app-logo span { color: #4f6f8f; }
.app-stats {
    font-size: 0.76rem; color: #7c8792;
    font-family: 'JetBrains Mono', monospace;
}
.app-header-subtitle {
    display: none;
    font-size: 0.72rem;
    color: #7c8792;
    margin-top: 0.15rem;
}

/* ============================================================
   HERO SECTION
   ============================================================ */
.hero-container {
    text-align: left;
    padding: 1.35rem 1.35rem 1.2rem;
    border: 1px solid rgba(119,137,153,0.18);
    border-radius: 14px;
    margin-bottom: 1.1rem;
    position: relative;
    overflow: hidden;
    background:
        linear-gradient(135deg, rgba(255,255,255,0.78), rgba(243,248,252,0.56)),
        radial-gradient(circle at 85% 0%, rgba(215,195,124,0.20), transparent 38%);
    box-shadow: 0 18px 48px rgba(53,77,102,0.08);
    animation: fadeInUp 0.42s ease both;
}
.hero-icon { display: none; }
.hero-kicker {
    color: #4f6f8f;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    margin-bottom: 0.35rem;
    text-transform: uppercase;
}
.hero-container::after {
    content: "";
    position: absolute;
    left: 1.35rem;
    right: 1.35rem;
    bottom: 0;
    height: 2px;
    background: linear-gradient(90deg, rgba(79,111,143,0.52), rgba(215,195,124,0.46), transparent);
}
.hero-title {
    font-family: 'Noto Serif SC', 'IBM Plex Sans', serif;
    font-size: 1.58rem; font-weight: 700; color: #202936;
    letter-spacing: 0.01em; margin-bottom: 0.55rem;
}
.hero-desc {
    font-size: 0.9rem; color: #52606d;
    max-width: 760px; margin: 0; line-height: 1.6;
}
.hero-accent { color: #4f6f8f; font-weight: 600; }

.suggestion-chip {
    display: inline-block;
    padding: 6px 18px; margin: 4px;
    background: rgba(255,255,255,0.62); border: 1px solid rgba(119,137,153,0.22);
    border-radius: 999px; font-size: 0.82rem; color: #52606d;
    cursor: pointer; transition: transform 0.18s ease, background-color 0.18s ease, border-color 0.18s ease, color 0.18s ease, box-shadow 0.18s ease;
}
.suggestion-chip:hover {
    transform: translateY(-1px);
    background: rgba(247,250,252,0.9); border-color: rgba(79,111,143,0.38); color: #395872;
    box-shadow: 0 8px 20px rgba(53,77,102,0.08);
}

/* ============================================================
   SEARCH BAR
   ============================================================ */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.78) !important;
    border: 1px solid rgba(119,137,153,0.24) !important;
    border-radius: 8px !important;
    color: #202936 !important;
    padding: 0.7rem 1rem !important;
    font-size: 0.95rem !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    transition: border-color 0.18s ease, box-shadow 0.18s ease, background-color 0.18s ease;
}
.stTextInput > div > div > input:focus {
    border-color: #4f6f8f !important;
    background: rgba(255,255,255,0.95) !important;
    box-shadow: 0 0 0 3px rgba(79,111,143,0.12), 0 10px 26px rgba(53,77,102,0.08) !important;
}
.stTextInput > div > div > input::placeholder { color: #7c8792 !important; }

/* ============================================================
   BUTTONS — override all Streamlit button variants
   Use .stApp prefix to beat Streamlit's dynamically-injected CSS
   ============================================================ */
/* Default & secondary — warm light bg + dark text */
.stApp .stButton > button,
.stApp .stButton > button[kind="secondary"] {
    background: rgba(255,255,255,0.62) !important; color: #202936 !important;
    border: 1px solid rgba(119,137,153,0.24) !important; border-radius: 7px !important;
    font-size: 0.84rem !important; font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 500 !important; padding: 0.45rem 1.2rem !important;
    box-shadow: 0 5px 16px rgba(53,77,102,0.045) !important;
    transition: transform 0.18s ease, box-shadow 0.18s ease, background-color 0.18s ease, border-color 0.18s ease !important;
}
.stApp .stButton > button:hover,
.stApp .stButton > button[kind="secondary"]:hover {
    transform: translateY(-1px);
    background: rgba(255,255,255,0.86) !important; border-color: rgba(79,111,143,0.36) !important;
    color: #395872 !important;
    box-shadow: 0 10px 26px rgba(53,77,102,0.09) !important;
}
.stApp .stButton > button:active,
.stApp .stButton > button[kind="secondary"]:active {
    transform: translateY(0);
    background: rgba(235,242,247,0.95) !important; border-color: rgba(79,111,143,0.38) !important;
    color: #202936 !important;
}
/* Primary button (active/selected state) */
.stApp .stButton > button[kind="primary"] {
    background-color: #466984 !important;
    background-image: linear-gradient(135deg, #4f6f8f, #3f617d) !important;
    color: #f6f9fb !important;
    border-color: rgba(79,111,143,0.70) !important; font-weight: 600 !important;
    box-shadow: 0 12px 28px rgba(79,111,143,0.20) !important;
    transition: transform 0.16s ease, box-shadow 0.16s ease, filter 0.16s ease, border-color 0.16s ease !important;
}
.stApp .stButton > button[kind="primary"]:hover {
    background-color: #42647f !important;
    background-image: linear-gradient(135deg, #4a6a86, #3b5b74) !important;
    border-color: rgba(79,111,143,0.82) !important;
    color: #f6f9fb !important;
    transform: translateY(-1px);
    filter: brightness(0.98) saturate(0.96);
    box-shadow: 0 14px 30px rgba(79,111,143,0.22) !important;
}
.stApp .stButton > button[kind="primary"]:active {
    background-color: #344f67 !important;
    background-image: linear-gradient(135deg, #3e5d76, #344f67) !important;
    border-color: rgba(57,88,114,0.88) !important;
    color: #f6f9fb !important;
    transform: translateY(0);
    box-shadow: 0 8px 20px rgba(79,111,143,0.18) !important;
}
/* Focus ring override */
.stApp .stButton > button:focus,
.stApp .stButton > button:focus-visible {
    outline: 2px solid rgba(79,111,143,0.45) !important;
    outline-offset: 2px !important;
    box-shadow: 0 0 0 3px rgba(79,111,143,0.18) !important;
}
.stApp .stButton > button[kind="primary"]:focus,
.stApp .stButton > button[kind="primary"]:focus-visible {
    background-color: #466984 !important;
    background-image: linear-gradient(135deg, #4f6f8f, #3f617d) !important;
    color: #f6f9fb !important;
    box-shadow: 0 0 0 3px rgba(79,111,143,0.16), 0 12px 28px rgba(79,111,143,0.20) !important;
}

button[data-testid="stBaseButton-primary"],
button[data-testid="stBaseButton-primary"]:hover,
button[data-testid="stBaseButton-primary"]:focus,
button[data-testid="stBaseButton-primary"]:focus-visible {
    background-color: #466984 !important;
    background-image: linear-gradient(135deg, #4f6f8f, #3f617d) !important;
    color: #f6f9fb !important;
}

/* ============================================================
   RESULT CARDS
   ============================================================ */
.paper-card {
    background:
        linear-gradient(180deg, rgba(255,255,255,0.88), rgba(252,253,254,0.74)),
        radial-gradient(circle at 100% 0%, rgba(215,195,124,0.10), transparent 32%);
    border: 1px solid rgba(119,137,153,0.20);
    border-radius: 10px;
    padding: 1.05rem 1.35rem 0.05rem;
    margin-bottom: 0.82rem;
    position: relative;
    backdrop-filter: blur(16px);
    box-shadow: var(--shadow-card);
    transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease, background-color 0.18s ease;
    animation: fadeInUp 0.42s cubic-bezier(.2,.75,.25,1) both;
}
.paper-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-card-hover);
    border-color: rgba(79,111,143,0.32);
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

.card-title {
    font-family: 'Noto Serif SC', 'IBM Plex Sans', serif;
    font-size: 1.04rem; font-weight: 700; color: #202936;
    line-height: 1.38; margin: 0.18rem 5.5rem 0.45rem 0;
    text-wrap: pretty;
}
.card-cats { margin-bottom: 0.5rem; }
.card-meta {
    font-size: 0.8rem; color: #52606d; margin-bottom: 0.3rem;
    overflow-wrap: anywhere;
}
.card-divider {
    height: 1px; background: linear-gradient(90deg, rgba(119,137,153,0.18), rgba(215,195,124,0.22), transparent); margin: 0.6rem 0;
}
.card-abstract {
    font-size: 0.83rem; color: #52606d; line-height: 1.6;
    margin-bottom: 0.8rem;
    overflow-wrap: anywhere;
}
.rank-label {
    color: #4f6f8f;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    margin-right: 0.45rem;
}

.score-badge {
    position: absolute;
    top: 1rem; right: 1rem;
    background: rgba(216,194,122,0.22);
    color: #6f5e2b;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.73rem; font-weight: 600;
    padding: 2px 10px; border-radius: 999px;
    border: 1px solid rgba(216,194,122,0.42);
}

.cat-pill {
    display: inline-block;
    padding: 2px 8px; margin: 0 4px 4px 0;
    border-radius: 4px; font-size: 0.68rem; font-weight: 600;
    font-family: 'JetBrains Mono', monospace; letter-spacing: 0.02em;
}
.cat-cs-CL  { background: rgba(180,60,60,0.1);   color: #9b3c3c; border: 1px solid rgba(180,60,60,0.2); }
.cat-cs-AI  { background: rgba(50,90,160,0.1);   color: #3b5998; border: 1px solid rgba(50,90,160,0.2); }
.cat-cs-LG  { background: rgba(120,60,160,0.1);  color: #7c3ba0; border: 1px solid rgba(120,60,160,0.2); }
.cat-cs-CV  { background: rgba(180,120,40,0.1);  color: #a0702a; border: 1px solid rgba(180,120,40,0.2); }
.cat-cs-IR  { background: rgba(60,130,90,0.1);   color: #3d7a50; border: 1px solid rgba(60,130,90,0.2); }
.cat-cs-NE  { background: rgba(190,70,130,0.1);  color: #b04a7a; border: 1px solid rgba(190,70,130,0.2); }
.cat-cs-RO  { background: rgba(190,100,50,0.1);  color: #b06a30; border: 1px solid rgba(190,100,50,0.2); }
.cat-cs-DC  { background: rgba(40,140,150,0.1);  color: #2d8a90; border: 1px solid rgba(40,140,150,0.2); }
.cat-cs-DS  { background: rgba(60,140,110,0.1);  color: #3d8a6a; border: 1px solid rgba(60,140,110,0.2); }
.cat-cs-SE  { background: rgba(170,140,40,0.1);  color: #a08820; border: 1px solid rgba(170,140,40,0.2); }
.cat-default { background: rgba(155,139,115,0.1); color: #8b7d68; border: 1px solid rgba(155,139,115,0.25); }

.card-action-btn > button {
    background: transparent !important; color: #4f6f8f !important;
    border: none !important; padding: 0.3rem 0 !important;
    font-size: 0.8rem !important; font-weight: 600 !important;
}
.card-action-btn > button:hover {
    color: #3f5f7d !important; background: transparent !important;
}

/* Results info bar */
.results-bar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.6rem 1rem;
    background: rgba(255,255,255,0.62); border: 1px solid rgba(119,137,153,0.20);
    border-radius: 8px; margin-bottom: 1rem;
    font-size: 0.85rem; color: #52606d;
    box-shadow: 0 10px 28px rgba(53,77,102,0.055);
    backdrop-filter: blur(14px);
}
.results-toolbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    margin: 1.1rem 0 0.65rem;
}
.section-label {
    color: #7c8792;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.section-title {
    color: #202936;
    font-family: 'Noto Serif SC', 'IBM Plex Sans', serif;
    font-size: 1.12rem;
    font-weight: 700;
    margin-top: 0.1rem;
}
.results-summary {
    color: #52606d;
    font-size: 0.82rem;
    text-align: right;
}
.search-row-note {
    color: #7c8792;
    font-size: 0.76rem;
    margin: -0.25rem 0 0.75rem;
}

/* ============================================================
   DETAIL PAGE
   ============================================================ */
.breadcrumb {
    font-size: 0.82rem; color: #7c8792; margin-bottom: 0.75rem;
    display: flex; gap: 0.4rem; align-items: center; flex-wrap: wrap;
}

.detail-hero {
    background:
        linear-gradient(135deg, rgba(255,255,255,0.82), rgba(240,247,251,0.76)),
        radial-gradient(circle at 92% 0%, rgba(215,195,124,0.16), transparent 36%);
    border: 1px solid rgba(119,137,153,0.20);
    border-radius: 10px;
    padding: 1.7rem 2rem;
    margin-bottom: 0.85rem;
    position: relative; overflow: hidden;
    box-shadow: 0 18px 48px rgba(53,77,102,0.08);
    backdrop-filter: blur(16px);
}
.detail-hero::before {
    content: '';
    position: absolute; top: 0; left: 0;
    width: 100%; height: 3px;
    background: linear-gradient(90deg, #4f6f8f, #d8c27a);
}
.detail-hero-title {
    font-family: 'Noto Serif SC', 'IBM Plex Sans', serif;
    font-size: 1.48rem; font-weight: 700; color: #202936;
    line-height: 1.35; margin-bottom: 0.75rem;
    text-wrap: pretty;
}
.detail-hero-meta {
    font-size: 0.85rem; color: #52606d;
    display: flex; gap: 0.85rem; flex-wrap: wrap;
}
.source-link-row {
    margin: 0.25rem 0 1.15rem;
}

.abstract-panel {
    background: rgba(255,255,255,0.76); border: 1px solid rgba(119,137,153,0.20);
    border-radius: 8px; padding: 1.5rem; margin-bottom: 1rem;
    height: 100%;
    box-shadow: 0 12px 34px rgba(53,77,102,0.06);
    backdrop-filter: blur(14px);
}
.abstract-panel h3,
.metadata-panel h3 {
    font-size: 0.85rem; font-weight: 600; color: #4f6f8f;
    text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.75rem;
}
.abstract-panel .paper-abstract { font-size: 0.88rem; color: #2d3742; line-height: 1.7; }
.metadata-panel {
    background: rgba(255,255,255,0.68);
    border: 1px solid rgba(119,137,153,0.20);
    border-radius: 8px;
    padding: 1.25rem;
    box-shadow: 0 10px 28px rgba(53,77,102,0.055);
    backdrop-filter: blur(14px);
}
.metadata-empty {
    color: #7c8792;
    font-size: 0.82rem;
    line-height: 1.55;
}

.meta-item {
    background: rgba(255,255,255,0.72); border: 1px solid rgba(119,137,153,0.20);
    border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 0.5rem;
    box-shadow: 0 8px 24px rgba(53,77,102,0.05);
}
.meta-item .meta-label {
    font-size: 0.68rem; text-transform: uppercase;
    letter-spacing: 0.05em; color: #7c8792;
}
.meta-item .meta-value { font-size: 0.85rem; color: #202936; margin-top: 0.2rem; }

.similar-card {
    background: rgba(255,255,255,0.72); border: 1px solid rgba(119,137,153,0.20);
    border-radius: 8px; padding: 1rem 1.2rem; margin-bottom: 0.5rem;
    transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
    box-shadow: 0 8px 24px rgba(53,77,102,0.045);
}
.similar-card:hover { transform: translateY(-2px); border-color: #4f6f8f; box-shadow: 0 14px 34px rgba(53,77,102,0.08); }
.similar-card-title {
    font-size: 0.88rem; font-weight: 600; color: #202936; line-height: 1.4;
}
.similar-card-score {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem; color: #4f6f8f;
}

/* ============================================================
   SIDEBAR
   ============================================================ */
[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, rgba(245,249,252,0.94), rgba(236,244,249,0.88)),
        radial-gradient(circle at 0% 0%, rgba(215,195,124,0.13), transparent 42%) !important;
    border-right: 1px solid rgba(119,137,153,0.22) !important;
    backdrop-filter: blur(18px);
}
/* Sidebar text — override Streamlit dark defaults */
[data-testid="stSidebar"],
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] caption {
    color: #202936 !important;
}
/* Sidebar input fields */
[data-testid="stSidebar"] input[type="text"],
[data-testid="stSidebar"] input[type="password"],
[data-testid="stSidebar"] input[type="number"],
[data-testid="stSidebar"] textarea {
    background: rgba(255,255,255,0.72) !important;
    border: 1px solid rgba(119,137,153,0.24) !important;
    color: #202936 !important;
}
[data-testid="stSidebar"] input::placeholder {
    color: #7c8792 !important;
}
/* Sidebar number input stepper buttons */
[data-testid="stSidebar"] [data-testid="stNumberInput"] button {
    background: #ffffff !important;
    color: #202936 !important;
    border: 1px solid rgba(119,137,153,0.22) !important;
}
[data-testid="stSidebar"] [data-testid="stNumberInput"] button:hover {
    background: rgba(235,242,247,0.88) !important;
    border-color: #aaa194 !important;
}
[data-testid="stSidebar"] [data-testid="stNumberInput"] button svg {
    fill: #202936 !important;
    color: #202936 !important;
}
/* Sidebar expander */
[data-testid="stSidebar"] [data-testid="stExpander"] details {
    background: rgba(255,255,255,0.28) !important;
    border: 1px solid rgba(119,137,153,0.18) !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    background: rgba(255,255,255,0.50) !important;
    color: #202936 !important;
    border-radius: 7px !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] .stMarkdown {
    background: transparent !important;
}
/* Sidebar toggle / checkbox / radio */
[data-testid="stSidebar"] .st-bx,
[data-testid="stSidebar"] .st-cb,
[data-testid="stSidebar"] .st-d8,
[data-testid="stSidebar"] [data-baseweb="checkbox"],
[data-testid="stSidebar"] [data-baseweb="radio"] {
    background: transparent !important;
}
[data-testid="stSidebar"] [data-baseweb="checkbox"] label,
[data-testid="stSidebar"] [data-baseweb="radio"] label,
[data-testid="stSidebar"] [data-baseweb="toggle"] label {
    color: #202936 !important;
}
[data-testid="stSidebar"] label[data-baseweb="checkbox"] > div {
    background: transparent !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] label[data-baseweb="checkbox"] > span:first-child {
    border-color: rgba(119,137,153,0.34) !important;
    background: rgba(255,255,255,0.72) !important;
}
[data-testid="stSidebar"] label[data-baseweb="checkbox"]:has(input[type="checkbox"]:checked) > span:first-child {
    background-color: #4f6f8f !important;
    border-color: #4f6f8f !important;
    box-shadow: 0 0 0 3px rgba(79,111,143,0.12) !important;
}
[data-testid="stSidebar"] label[data-baseweb="checkbox"]:has(input[type="checkbox"]:checked) > span:first-child svg {
    color: #ffffff !important;
    fill: #ffffff !important;
}
[data-testid="stSidebar"] [data-testid="stCheckbox"] p {
    background: transparent !important;
}
[data-testid="stSidebar"] button[aria-label^="Help for"] {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 0.95rem !important;
    height: 0.95rem !important;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] [data-testid="stTooltipIcon"] {
    color: #4f6f8f !important;
}
[data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg,
[data-testid="stSidebar"] button[aria-label^="Help for"] svg {
    width: 0.88rem !important;
    height: 0.88rem !important;
    color: #4f6f8f !important;
    fill: none !important;
    stroke: #4f6f8f !important;
}
[data-testid="stSidebar"] .data-mode-status {
    margin: 0.45rem 0 0.35rem;
    padding: 0.55rem 0.68rem;
    border-radius: 8px;
    border: 1px solid rgba(79,111,143,0.28);
    background: rgba(235,242,247,0.66);
    color: #395872;
    font-size: 0.78rem;
    line-height: 1.45;
}
[data-testid="stSidebar"] .data-mode-status strong {
    color: #202936;
    font-weight: 650;
}
[data-testid="stSidebar"] [data-baseweb="toggle"] [role="switch"] {
    background: #aaa194 !important;
}
[data-testid="stSidebar"] [data-baseweb="toggle"] [aria-checked="true"] [role="switch"] {
    background: #4f6f8f !important;
}
/* Sidebar select slider */
[data-testid="stSidebar"] [data-baseweb="slider"] {
    background: transparent !important;
}
[data-testid="stSidebar"] [data-baseweb="slider"] div {
    color: #202936 !important;
}
/* Sidebar buttons */
[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.54) !important;
    color: #202936 !important;
    border: 1px solid rgba(119,137,153,0.26) !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.82) !important;
}
.sidebar-brand {
    font-family: 'Noto Serif SC', 'IBM Plex Sans', serif;
    font-size: 1.08rem; font-weight: 700; margin-bottom: 0.75rem;
}

.sidebar-stat {
    display: flex; align-items: center; gap: 0.4rem;
    font-size: 0.78rem; color: #52606d; padding: 0.2rem 0;
    font-family: 'JetBrains Mono', monospace;
}
.sidebar-stat .dot { width: 6px; height: 6px; border-radius: 50%; display: inline-block; }
.dot-green { background: #4f7259; }
.dot-gray  { background: #aaa194; }

/* ============================================================
   MISC
   ============================================================ */
.stDivider { border-color: rgba(119,137,153,0.22) !important; }
hr { border-color: rgba(119,137,153,0.22) !important; }

.skeleton-card {
    background: rgba(255,255,255,0.62); border: 1px solid rgba(119,137,153,0.22);
    border-radius: 8px; height: 130px; margin-bottom: 1rem;
    animation: pulse 1.5s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 0.5; }
    50% { opacity: 0.85; }
}

/* Radio tweaks */
.stRadio > div { gap: 0.3rem; }
.stRadio label {
    padding: 0.35rem 0.75rem !important; border-radius: 6px !important;
}

/* Expander tweaks */
.streamlit-expanderHeader {
    font-size: 0.85rem !important; font-weight: 600 !important;
    color: #202936 !important;
}
/* Main area expanders */
[data-testid="stExpander"] details {
    border: 1px solid rgba(119,137,153,0.20) !important;
    border-radius: 10px !important;
    background: rgba(255,255,255,0.52) !important;
    box-shadow: 0 10px 26px rgba(53,77,102,0.045);
    backdrop-filter: blur(14px);
}
[data-testid="stExpander"] summary {
    color: #202936 !important;
}

button[data-testid^="stBaseButton-segmented_control"],
button[data-testid^="stBaseButton-pills"] {
    background: rgba(255,255,255,0.68) !important;
    border: 1px solid rgba(119,137,153,0.24) !important;
    color: #202936 !important;
    border-radius: 999px !important;
    box-shadow: 0 5px 16px rgba(53,77,102,0.045) !important;
}
button[data-testid="stBaseButton-segmented_controlActive"],
button[data-testid="stBaseButton-pillsActive"] {
    background: rgba(79,111,143,0.12) !important;
    border-color: rgba(79,111,143,0.52) !important;
    color: #395872 !important;
    box-shadow: 0 8px 22px rgba(79,111,143,0.12) !important;
}

/* Search spinner */
.search-spinner {
    width: 40px; height: 40px;
    border: 3px solid rgba(119,137,153,0.22);
    border-top: 3px solid #4f6f8f;
    border-radius: 50%;
    margin: 0 auto;
    animation: spin 0.8s linear infinite;
}
@keyframes spin {
    to { transform: rotate(360deg); }
}

/* ============================================================
   SEARCH MODE SWITCH
   ============================================================ */
.mode-switch-container {
    display: flex;
    justify-content: flex-start;
    width: 100%;
    margin-bottom: 1.2rem;
}
.mode-switch {
    display: inline-flex;
    background: rgba(255,255,255,0.48);
    border: 1px solid rgba(119,137,153,0.20);
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 10px 28px rgba(53,77,102,0.06);
    backdrop-filter: blur(14px);
}
.mode-switch-btn {
    padding: 0.6rem 1.8rem;
    font-size: 0.9rem;
    font-weight: 500;
    color: #52606d;
    background: transparent;
    border: none;
    cursor: pointer;
    transition: color 0.2s ease, background-color 0.2s ease, box-shadow 0.2s ease;
    white-space: nowrap;
}
.mode-switch-btn.active {
    background: rgba(255,255,255,0.92);
    color: #202936;
    font-weight: 700;
    box-shadow: 0 6px 18px rgba(53,77,102,0.10);
    border-radius: 8px;
    margin: 2px;
}
.mode-switch-btn:not(.active):hover {
    color: #4f6f8f;
    background: rgba(79,111,143,0.07);
}
.mode-switch-icon {
    font-size: 1.1rem;
    margin-right: 0.3rem;
}

/* ============================================================
   AI ONBOARDING CARD
   ============================================================ */
.ai-onboarding {
    background:
        linear-gradient(135deg, rgba(255,255,255,0.78), rgba(238,246,250,0.72)),
        radial-gradient(circle at 100% 0%, rgba(215,195,124,0.18), transparent 40%);
    border: 1px solid rgba(119,137,153,0.20);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1.5rem;
    width: 100%;
    max-width: none;
    margin-left: 0;
    margin-right: 0;
    box-shadow: 0 18px 48px rgba(53,77,102,0.08);
    backdrop-filter: blur(16px);
}
.ai-onboarding-title {
    font-family: 'Noto Serif SC', 'IBM Plex Sans', serif;
    font-size: 0.95rem;
    font-weight: 700;
    color: #4f6f8f;
    margin-bottom: 0.6rem;
}
.ai-onboarding-desc {
    font-size: 0.82rem;
    color: #52606d;
    line-height: 1.6;
}
.ai-onboarding-features {
    display: flex;
    gap: 1rem;
    margin-top: 0.8rem;
    flex-wrap: wrap;
}
.ai-feature-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: rgba(255,255,255,0.64);
    border: 1px solid rgba(119,137,153,0.20);
    border-radius: 6px;
    padding: 0.35rem 0.7rem;
    font-size: 0.78rem;
    color: #202936;
}
.ai-feature-badge .feat-icon {
    font-size: 1rem;
}

/* ============================================================
   AI ANSWER CARD
   ============================================================ */
.ai-answer-card {
    background: rgba(255,255,255,0.76);
    border: 1px solid rgba(119,137,153,0.20);
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 16px 40px rgba(53,77,102,0.07);
    backdrop-filter: blur(16px);
}
.ai-answer-card-header {
    font-size: 0.8rem;
    color: #4f6f8f;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.8rem;
    font-weight: 700;
}
.ai-answer-loading {
    background: rgba(255,255,255,0.72);
    border: 1px solid rgba(119,137,153,0.20);
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    box-shadow: 0 14px 34px rgba(53,77,102,0.06);
    backdrop-filter: blur(14px);
}
.ai-answer-loading .loading-text {
    color: #52606d;
    font-size: 0.9rem;
}

@media (max-width: 760px) {
    [data-testid="stMainBlockContainer"],
    .main .block-container {
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
    }
    .app-header {
        align-items: flex-start;
        padding: 0.7rem 0.8rem;
        top: 0.35rem;
    }
    .app-stats {
        display: none;
    }
    .app-header-subtitle {
        display: block;
    }
    .hero-container {
        padding: 1.05rem;
    }
    .hero-title {
        font-size: 1.28rem;
    }
    .detail-hero {
        padding: 1.2rem;
    }
    .detail-hero-title {
        font-size: 1.22rem;
    }
    .card-title {
        margin-right: 0;
        font-size: 0.98rem;
    }
    .score-badge {
        position: static;
        display: inline-block;
        margin-bottom: 0.45rem;
    }
    .results-toolbar {
        display: block;
    }
    .results-summary {
        text-align: left;
        margin-top: 0.35rem;
    }
}
</style>
""", unsafe_allow_html=True)


# ===== INIT SESSION STATE =====
for key, default in [
    ("docs", None), ("engine", None), ("last_results", []),
    ("last_query", ""), ("last_algo", "TF-IDF + Cosine"),
    ("last_advanced_signature", ""),
    ("selected_paper", None), ("page", "search"), ("similar", {}),
    ("sort_by", "Relevance"),
    # AI features
    ("ai_enabled", False), ("ai_use_enhance", True),
    ("ai_use_answer", True), ("ai_use_chat", True),
    ("ai_enhanced_keywords", ""), ("ai_enhance_explanation", ""),
    ("ai_answer", ""), ("ai_chat_history", []),
    ("ai_answer_length", "标准"),
    ("data_loaded_once", False),
    ("pending_clear_search", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ===== CACHED DATA LOADER =====
@st.cache_resource
def load_engine(data_source, max_docs):
    if data_source == "__mysql__":
        docs = load_from_database(max_docs=None if max_docs == 0 else int(max_docs))
    else:
        docs = load_jsonl(data_source, max_docs=None if max_docs == 0 else int(max_docs))
    docs = make_title_abstract_field(docs)
    for d in docs:
        d["title_abstract"] = clean_text(d["title_abstract"])
    engine = SearchEngine(docs)
    engine.build_tfidf()
    engine.build_bm25()
    return docs, engine


def _contains_all(text, terms):
    text = (text or "").lower()
    return all(term in text for term in terms)


def _split_terms(value):
    normalized = (value or "").replace(";", ",")
    return [part.strip().lower() for part in normalized.split(",") if part.strip()]


def apply_advanced_filters(docs, filters):
    title_terms = _split_terms(filters.get("title"))
    author_terms = _split_terms(filters.get("author"))
    abstract_terms = _split_terms(filters.get("abstract"))
    exclude_terms = _split_terms(filters.get("exclude"))
    category = (filters.get("category") or "").strip().lower()
    date_from = (filters.get("date_from") or "").strip()
    date_to = (filters.get("date_to") or "").strip()

    filtered = []
    for doc in docs:
        title = doc.get("title", "")
        authors = doc.get("authors", "")
        abstract = doc.get("abstract", "")
        categories = doc.get("categories", "")
        update_date = doc.get("update_date", "")
        full_text = " ".join([title, authors, abstract, categories]).lower()

        if title_terms and not _contains_all(title, title_terms):
            continue
        if author_terms and not _contains_all(authors, author_terms):
            continue
        if abstract_terms and not _contains_all(abstract, abstract_terms):
            continue
        if exclude_terms and any(term in full_text for term in exclude_terms):
            continue
        if category and category not in categories.lower():
            continue
        if date_from and update_date and update_date < date_from:
            continue
        if date_to and update_date and update_date > date_to:
            continue
        filtered.append(doc)

    return filtered


def advanced_filters_active(filters):
    return any((value or "").strip() for value in filters.values())


def run_search_with_filters(base_docs, base_engine, query, method, limit, model_name, filters):
    active = advanced_filters_active(filters)
    docs_for_search = apply_advanced_filters(base_docs, filters) if active else base_docs

    if not docs_for_search:
        return []

    if active:
        search_engine = SearchEngine(docs_for_search)
        search_engine.build_tfidf()
        search_engine.build_bm25()
    else:
        search_engine = base_engine

    query = (query or "").strip()
    if not query:
        return [(idx, 1.0, doc) for idx, doc in enumerate(docs_for_search[:limit])]

    if method == "TF-IDF + Cosine":
        return search_engine.search_tfidf(query, top_k=limit)
    if method == "BM25":
        return search_engine.search_bm25(query, top_k=limit)
    return search_engine.search_sbert(query, top_k=limit, model_name=model_name)


def go_to_search_results():
    st.session_state.page = "search"
    st.session_state.selected_paper = None


def open_paper_detail(paper):
    st.session_state.selected_paper = paper
    st.session_state.page = "detail"


# ===== SIDEBAR =====
with st.sidebar:
    st.markdown('<div class="sidebar-brand">arXiv Research Search</div>', unsafe_allow_html=True)

    with st.expander("数据源", expanded=True):
        use_mysql = st.checkbox("使用数据库", value=False, help="启用后优先连接 MySQL；连接失败时自动回退到本地 SQLite 数据库。")
        if use_mysql:
            data_source = "__mysql__"
            st.markdown(
                '<div class="data-mode-status"><strong>数据库模式已启用</strong><br>优先 MySQL，失败后自动回退 SQLite。</div>',
                unsafe_allow_html=True
            )
            max_docs = st.number_input("最大文档数", 0, 50000, 0, 100, help="0 表示加载全部文档；数据库模式下会对实际载入数量生效。")
        else:
            data_file = st.text_input("数据路径", DEFAULT_DATA, label_visibility="collapsed")
            data_source = data_file
            max_docs = st.number_input("最大文档数", 0, 50000, 500, 100, help="0 表示加载全部文档；默认读取本地 JSONL 样本文档。")

    with st.expander("检索算法", expanded=True):
        algo = st.radio("检索算法", ["TF-IDF + Cosine", "BM25", "Sentence-BERT"],
                        index=0, label_visibility="collapsed")
        if algo == "Sentence-BERT":
            sbert_model = st.text_input("模型名称", "sentence-transformers/all-MiniLM-L6-v2",
                                       label_visibility="collapsed")
        else:
            sbert_model = None

    with st.expander("增强检索设置", expanded=False):
        st.caption("配置模型接口，用于查询改写、结果综述和追问")

        api_key = st.text_input(
            "API Key", type="password",
            value=os.environ.get("DEEPSEEK_API_KEY", ""),
            placeholder="sk-...",
            key="sidebar_api_key"
        )
        api_base = st.text_input(
            "API 地址",
            value=DEFAULT_AI_BASE_URL,
            placeholder=DEFAULT_DEEPSEEK_BASE_URL,
            key="sidebar_api_base"
        )
        ai_model = st.text_input(
            "模型名称",
            value=DEFAULT_AI_MODEL,
            placeholder=DEFAULT_DEEPSEEK_MODEL,
            key="sidebar_ai_model"
        )
        ai_answer_length = st.select_slider(
            "答案详细程度",
            options=["简短", "标准", "详细"],
            value=st.session_state.ai_answer_length,
            key="sidebar_ai_length"
        )
        st.session_state.ai_answer_length = ai_answer_length

        st.caption("辅助功能")
        ac1, ac2, ac3 = st.columns(3)
        with ac1:
            st.session_state.ai_use_enhance = st.checkbox("查询改写", value=st.session_state.ai_use_enhance)
        with ac2:
            st.session_state.ai_use_answer = st.checkbox("结果综述", value=st.session_state.ai_use_answer)
        with ac3:
            st.session_state.ai_use_chat = st.checkbox("结果追问", value=st.session_state.ai_use_chat)

        # Store API config in session
        st.session_state._ai_api_key = api_key
        st.session_state._ai_base_url = api_base
        st.session_state._ai_model = ai_model

        api_configured = bool(api_key or os.environ.get("DEEPSEEK_API_KEY"))
        if api_configured:
            st.success("接口已配置")
        else:
            st.warning("请填写 API Key")
        st.caption("切换到增强检索后启用这些辅助功能")

    with st.expander("结果设置", expanded=True):
        top_k = st.slider("返回数量", 5, 50, 15, label_visibility="collapsed")

    st.divider()

    st.markdown("### 状态")
    if st.session_state.engine is not None:
        stats = st.session_state.engine.get_stats()
        st.markdown(f'<div class="sidebar-stat"><span class="dot dot-green"></span> {stats["num_docs"]} papers</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sidebar-stat"><span class="dot {"dot-green" if stats["tfidf_built"] else "dot-gray"}"></span> TF-IDF {"ready" if stats["tfidf_built"] else "pending"}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sidebar-stat"><span class="dot {"dot-green" if stats["bm25_built"] else "dot-gray"}"></span> BM25 {"ready" if stats["bm25_built"] else "pending"}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sidebar-stat"><span class="dot {"dot-green" if stats["sbert_built"] else "dot-gray"}"></span> SBERT {"ready" if stats["sbert_built"] else "pending"}</div>', unsafe_allow_html=True)
    else:
        st.caption("加载数据后显示状态")

    st.divider()
    st.caption("arXiv Research Search v1.0")

    st.button("← 返回检索", use_container_width=True, on_click=go_to_search_results)


# ===== LOAD DATA =====
if not use_mysql and not os.path.exists(data_source):
    st.error(f"数据文件不存在: {data_source}")
    st.stop()

if st.session_state.docs is None:
    source_label = "MySQL 数据库" if use_mysql else data_source
    with st.spinner(f"加载论文数据 ({source_label})..."):
        st.session_state.docs, st.session_state.engine = load_engine(data_source, max_docs)
    st.toast(f"✅ 已加载 {len(st.session_state.docs)} 篇论文", icon="📄")
    if not st.session_state.data_loaded_once:
        st.session_state.data_loaded_once = True
        st.rerun()

engine = st.session_state.engine


# ===== HEADER BAR =====
stats = engine.get_stats() if engine else {}
st.markdown(f"""
    <div class="app-header">
    <div>
        <div class="app-logo">arXiv <span>Research Search</span></div>
        <div class="app-header-subtitle">论文检索 · 推荐 · 增强问答</div>
    </div>
    <div class="app-stats">{stats.get('num_docs', '—')} papers · TF-IDF · BM25</div>
</div>
""", unsafe_allow_html=True)


# ======================== DETAIL PAGE ========================
if st.session_state.page == "detail" and st.session_state.selected_paper:
    paper = st.session_state.selected_paper

    # Breadcrumb
    st.markdown(f"""
    <div class="breadcrumb">
        <span>arXiv 检索</span><span>›</span><span>论文详情</span><span>›</span>
        <span style="color:#7c8792;">{safe_html(paper.get('id',''))}</span>
    </div>
    """, unsafe_allow_html=True)

    st.button("← 返回检索结果", key="back_from_detail", on_click=go_to_search_results)

    # Hero header
    cats = (paper.get("categories") or "").split()
    cats_html = "".join(
        f'<span class="cat-pill">{safe_html(c)}</span>' for c in cats
    )
    detail_title = safe_html(paper.get("title", "无标题"))
    detail_authors = safe_html(paper.get("authors", "—"), 100)
    detail_date = safe_html(paper.get("update_date", "—"))
    detail_id = safe_html(paper.get("id", "—"))

    st.markdown(f"""
    <div class="detail-hero">
        <div style="margin-bottom:0.6rem;">{cats_html}</div>
        <div class="detail-hero-title">{detail_title}</div>
        <div class="detail-hero-meta">
            <span>Authors: {detail_authors}</span>
            <span>Date: {detail_date}</span>
            <span>📄 {detail_id}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    raw_arxiv_id = (paper.get("id") or "").strip()
    if raw_arxiv_id:
        arxiv_abs_url = f"https://arxiv.org/abs/{raw_arxiv_id}"
        arxiv_pdf_url = f"https://arxiv.org/pdf/{raw_arxiv_id}"
        st.markdown('<div class="source-link-row">', unsafe_allow_html=True)
        link_col1, link_col2, link_col3 = st.columns([1, 1, 4])
        with link_col1:
            st.link_button("arXiv 摘要页", arxiv_abs_url, use_container_width=True)
        with link_col2:
            st.link_button("PDF 原文", arxiv_pdf_url, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Content grid: abstract + metadata
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown(f"""
        <div class="abstract-panel">
            <h3>📝 Abstract</h3>
            <div class="paper-abstract">{safe_html(paper.get("abstract", "无摘要"))}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="metadata-panel"><h3>Metadata</h3>', unsafe_allow_html=True)
        meta_fields = [
            ("DOI", paper.get("doi", "")),
            ("Journal Ref", paper.get("journal_ref", "")),
            ("Comments", paper.get("comments", "")),
            ("Report No", paper.get("report_no", "")),
            ("License", paper.get("license", "")),
            ("Submitter", paper.get("submitter", "")),
        ]
        rendered_meta = False
        for label, val in meta_fields:
            if val:
                rendered_meta = True
                st.markdown(f"""
                <div class="meta-item">
                    <div class="meta-label">{safe_html(label)}</div>
                    <div class="meta-value">{safe_html(val)}</div>
                </div>
                """, unsafe_allow_html=True)
        if not rendered_meta:
            st.markdown('<div class="metadata-empty">该论文在当前样例数据中没有 DOI、期刊引用或提交者等扩展元数据。</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Similar papers
    st.divider()
    st.markdown("### 📚 相似论文推荐")

    idx = st.session_state.docs.index(paper)
    algo_name = st.session_state.get("last_algo", "TF-IDF + Cosine")

    try:
        if algo_name == "TF-IDF + Cosine":
            sims = engine.get_similar_tfidf(idx, top_k=4)
        elif algo_name == "BM25":
            sims = engine.get_similar_bm25(idx, top_k=4)
        else:
            sims = engine.get_similar_sbert(idx, top_k=4, model_name=sbert_model)
    except Exception:
        sims = []

    if sims:
        sim_cols = st.columns(2)
        for i, (si, sscore, sdoc) in enumerate(sims):
            with sim_cols[i % 2]:
                st.markdown(f"""
                <div class="similar-card">
                    <div class="similar-card-title">{safe_html(sdoc.get('title', '无标题'), 100)}</div>
                    <div style="font-size:0.78rem;color:#52606d;margin:0.3rem 0;">
                        {safe_html(sdoc.get('authors',''), 60)}
                    </div>
                    <span class="similar-card-score">相似度 {sscore:.4f}</span>
                </div>
                """, unsafe_allow_html=True)
                st.button("查看详情", key=f"sim_{si}", on_click=open_paper_detail, args=(sdoc,))
    else:
        st.caption("暂无相似推荐")

    st.stop()


# ======================== CHAT PAGE ========================
if st.session_state.page == "chat":
    chat_results = st.session_state.last_results
    chat_query = st.session_state.last_query

    # Back button + header
    bc1, bc2 = st.columns([1, 5])
    with bc1:
        st.button("← 返回检索结果", key="back_from_chat", use_container_width=True, on_click=go_to_search_results)
    with bc2:
        st.markdown(f"""
        <div style="background:#f6f9fb;border:1px solid rgba(119,137,153,0.22);border-radius:8px;padding:0.7rem 1rem;">
            <span style="color:#7c8792;font-size:0.75rem;">当前检索</span>&nbsp;
            <span style="color:#202936;font-weight:600;">"{chat_query}"</span>
            <span style="color:#7c8792;font-size:0.78rem;">&nbsp;·&nbsp;基于 {len(chat_results)} 篇论文</span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Full chat history
    for msg in st.session_state.ai_chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Empty state
    if not st.session_state.ai_chat_history:
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;">
            <div style="font-size:3rem;margin-bottom:0.8rem;">💬</div>
            <div style="font-size:1.1rem;font-weight:600;color:#202936;margin-bottom:0.4rem;">
                对检索结果进行深度追问
            </div>
            <div style="color:#52606d;font-size:0.85rem;">
                AI 会基于当前检索到的论文回答你的问题
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Bottom toolbar
    st.divider()
    tb1, tb2 = st.columns([1, 5])
    with tb1:
        if st.button("🗑️ 清空对话", key="clear_chat_page", use_container_width=True):
            st.session_state.ai_chat_history = []
            st.rerun()

    # Chat input
    chat_input = st.chat_input("追问关于这些论文的问题...", key="chat_page_input")
    if chat_input:
        st.session_state.ai_chat_history.append({"role": "user", "content": chat_input})
        with st.chat_message("user"):
            st.markdown(chat_input)
        with st.chat_message("assistant"):
            try:
                ai_cfg = AISearchConfig(
                    api_key=st.session_state.get("_ai_api_key", ""),
                    base_url=st.session_state.get("_ai_base_url", DEFAULT_AI_BASE_URL),
                    model=st.session_state.get("_ai_model", DEFAULT_AI_MODEL),
                )
                ai_chat_engine = AISearchEngine(ai_cfg)
                top_papers = [doc for _, _, doc in chat_results[:5]]
                if ai_chat_engine.is_configured:
                    response_placeholder = st.empty()
                    full_reply = ""
                    for chunk in ai_chat_engine.chat_stream(
                        chat_input, st.session_state.ai_chat_history[:-1], top_papers
                    ):
                        full_reply += chunk
                        response_placeholder.markdown(full_reply + "▌")
                    response_placeholder.markdown(full_reply)
                    st.session_state.ai_chat_history.append(
                        {"role": "assistant", "content": full_reply}
                    )
                else:
                    st.warning("请先在侧边栏配置 API Key")
            except Exception as e:
                st.warning(f"对话暂时不可用: {e}")
        st.rerun()

    st.stop()


# ======================== SEARCH PAGE ========================

# Helper: render a result card
def render_card(rank, doc, score):
    cats = (doc.get("categories") or "").split()
    cats_html = "".join(
        f'<span class="cat-pill cat-{c}">{c}</span>' for c in cats[:6]
    )
    title = safe_html(doc.get("title", "无标题"))
    abstract = doc.get("abstract", "")
    abbr_abstract = abstract[:280]
    authors = safe_html(doc.get("authors", ""), 80)
    update_date = safe_html(doc.get("update_date", ""))
    paper_id = safe_html(doc.get("id", ""))

    st.markdown(f"""
    <div class="paper-card" style="animation-delay:{rank * 0.04}s">
        <span class="score-badge">{score:.4f}</span>
        <div class="card-cats">{cats_html}</div>
        <div class="card-title"><span class="rank-label">#{rank}</span> {title}</div>
        <div class="card-meta">Authors: {authors}{'...' if len(doc.get('authors','')) > 80 else ''}</div>
        <div class="card-meta">Date: {update_date} &nbsp;|&nbsp; ID: {paper_id}</div>
        <div class="card-divider"></div>
        <div class="card-abstract">{safe_html(abbr_abstract)}{'...' if len(abstract) > 280 else ''}</div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="card-action-btn">', unsafe_allow_html=True)
        st.button("查看详情 →", key=f"det_{rank}_{doc.get('id')}", on_click=open_paper_detail, args=(doc,))
        st.markdown('</div>', unsafe_allow_html=True)


# ---- Hero state (no search yet) ----
has_results = bool(
    st.session_state.last_results
    or st.session_state.last_query
    or st.session_state.last_advanced_signature
)

# ---- Resolve chip click before text_input renders ----
preset_query = st.session_state.pop("chip_query", None)
if preset_query:
    st.session_state.search_input = preset_query
trigger_search = st.session_state.pop("trigger_search", False)

# ---- Search mode switch ----
api_configured = bool(
    st.session_state.get("_ai_api_key", "")
    or os.environ.get("DEEPSEEK_API_KEY", "")
)

if not has_results:
    st.markdown("""
    <div class="hero-container">
        <div class="hero-kicker">Academic Retrieval Workspace</div>
        <div class="hero-title">论文检索工作台</div>
        <div class="hero-desc">
            面向计算机科学论文的检索、筛选与结果追问。当前样本库包含 <strong>20,000</strong> 篇 arXiv 论文，支持
            <span class="hero-accent">TF-IDF</span>,
            <span class="hero-accent">BM25</span>, and
            <span class="hero-accent">Sentence-BERT</span>.
        </div>
    </div>
    """, unsafe_allow_html=True)

# Mode switch UI
sc_mode, sc_ai_status = st.columns([5, 1])
with sc_mode:
    st.markdown('<div class="mode-switch-container">', unsafe_allow_html=True)
    mc1, mc2 = st.columns(2)
    with mc1:
        is_traditional = not st.session_state.ai_enabled
        if st.button("传统检索",
                     key="mode_traditional",
                     type="primary" if is_traditional else "secondary",
                     use_container_width=True):
            if st.session_state.ai_enabled:
                st.session_state.ai_enabled = False
                st.session_state.ai_answer = ""
                st.rerun()
    with mc2:
        is_ai = st.session_state.ai_enabled
        btn_label = "增强检索"
        if not api_configured:
            btn_label = "增强检索（未配置）"
        if st.button(btn_label,
                     key="mode_ai",
                     type="primary" if is_ai else "secondary",
                     use_container_width=True,
                     help="使用模型接口改写查询并生成结果综述" if api_configured else "请先在侧边栏增强检索设置中配置 API Key"):
            if not api_configured:
                st.toast("请先在侧边栏「增强检索设置」中填写 API Key")
            elif not st.session_state.ai_enabled:
                st.session_state.ai_enabled = True
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with sc_ai_status:
    if api_configured and st.session_state.ai_enabled:
        st.markdown(f"""
        <div style="text-align:right;padding-top:0.3rem;">
            <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#4f7259;margin-right:4px;"></span>
            <span style="font-size:0.78rem;color:#4f7259;">增强检索已启用</span>
        </div>
        """, unsafe_allow_html=True)
    elif not api_configured:
        st.markdown(f"""
        <div style="text-align:right;padding-top:0.3rem;">
            <span style="font-size:0.78rem;color:#7c8792;">未配置 API</span>
        </div>
        """, unsafe_allow_html=True)

# AI onboarding card (shown when AI mode is on, no results yet, and not already searched)
if (st.session_state.ai_enabled and not has_results
        and not st.session_state.last_query):
    st.markdown(f"""
    <div class="ai-onboarding">
        <div class="ai-onboarding-title">增强检索已开启</div>
        <div class="ai-onboarding-desc">
            用自然语言描述你的研究兴趣，系统会辅助完成三件事：
        </div>
        <div class="ai-onboarding-features">
            <div class="ai-feature-badge">
                查询改写
                <span style="color:#7c8792;">— 自动优化检索关键词</span>
            </div>
            <div class="ai-feature-badge">
                结果综述
                <span style="color:#7c8792;">— 综合多篇论文生成答案</span>
            </div>
            <div class="ai-feature-badge">
                结果追问
                <span style="color:#7c8792;">— 对结果进行深入提问</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---- Search bar ----
if st.session_state.ai_enabled:
    placeholder_text = "用自然语言描述你想了解的研究方向，例如：最近transformer在图像生成领域有哪些突破？"
else:
    placeholder_text = "输入英文关键词，如: large language model, image segmentation..."

if st.session_state.pending_clear_search:
    st.session_state.search_input = ""
    st.session_state.pending_clear_search = False

if has_results:
    sc1, sc2, sc3 = st.columns([5, 1, 1])
    with sc1:
        query = st.text_input(
            "search", placeholder=placeholder_text,
            label_visibility="collapsed",
            key="search_input"
        )
    with sc2:
        search_btn = st.button("搜 索", type="primary", use_container_width=True, key="search_btn")
    with sc3:
        if st.button("清空", use_container_width=True, key="clear_search"):
            st.session_state.last_results = []
            st.session_state.last_query = ""
            st.session_state.last_advanced_signature = ""
            st.session_state.ai_answer = ""
            st.session_state.ai_enhanced_keywords = ""
            st.session_state.ai_enhance_explanation = ""
            st.session_state.selected_paper = None
            st.session_state.page = "search"
            st.session_state.pending_clear_search = True
            st.rerun()
    st.markdown('<div class="search-row-note">修改关键词后按 Enter 或点击搜索即可刷新结果。</div>', unsafe_allow_html=True)
else:
    query = st.text_input(
        "search", placeholder=placeholder_text,
        label_visibility="collapsed",
        key="search_input"
    )
    search_btn = st.button("搜 索", type="primary", use_container_width=True, key="search_btn")

with st.expander("高级检索", expanded=False):
    st.caption("高级条件会和主搜索框取交集；主搜索框为空时，只按字段过滤。多个关键词可用英文逗号或分号分隔。")
    adv_col1, adv_col2, adv_col3 = st.columns([1.2, 1.2, 1])
    with adv_col1:
        adv_title = st.text_input("标题包含", key="adv_title", placeholder="graph neural network")
        adv_author = st.text_input("作者包含", key="adv_author", placeholder="Yoshua Bengio")
    with adv_col2:
        adv_abstract = st.text_input("摘要包含", key="adv_abstract", placeholder="retrieval, ranking")
        adv_category = st.text_input("分类包含", key="adv_category", placeholder="cs.LG")
    with adv_col3:
        adv_date_from = st.text_input("起始日期", key="adv_date_from", placeholder="YYYY-MM-DD")
        adv_date_to = st.text_input("结束日期", key="adv_date_to", placeholder="YYYY-MM-DD")
    adv_exclude = st.text_input("排除词", key="adv_exclude", placeholder="survey; finance")

advanced_filters = {
    "title": adv_title,
    "author": adv_author,
    "abstract": adv_abstract,
    "category": adv_category,
    "date_from": adv_date_from,
    "date_to": adv_date_to,
    "exclude": adv_exclude,
}
advanced_active = advanced_filters_active(advanced_filters)
advanced_signature = json.dumps(advanced_filters, sort_keys=True, ensure_ascii=False)

if not has_results:
    # Suggestion chips
    if st.session_state.ai_enabled:
        st.markdown("""
        <div style="text-align:center;margin-top:0.5rem;">
        <span style="color:#7c8792;font-size:0.78rem;">自然语言示例：</span>
        </div>
        """, unsafe_allow_html=True)
        suggestions = [
            "Transformer在图像分割中的应用",
            "扩散模型和GAN的对比",
            "图神经网络最新进展",
            "大模型高效微调方法",
            "联邦学习隐私保护方案",
            "目标检测中的注意力机制",
        ]
    else:
        st.markdown("""
        <div style="text-align:center;margin-top:0.5rem;">
        <span style="color:#7c8792;font-size:0.78rem;">关键词示例：</span>
        </div>
        """, unsafe_allow_html=True)
        suggestions = [
            "large language model", "image segmentation", "graph neural network",
            "reinforcement learning", "diffusion model", "attention mechanism",
        ]
    chip_cols = st.columns(len(suggestions))
    for i, sug in enumerate(suggestions):
        with chip_cols[i]:
            if st.button(sug, key=f"chip_{i}", use_container_width=True):
                st.session_state.chip_query = sug
                st.session_state.trigger_search = True
                st.rerun()


# ---- Execute search ----
should_search = (trigger_search or search_btn or query or advanced_active) and (query.strip() or advanced_active)
if should_search:
    if (trigger_search or search_btn or st.session_state.last_query != query
            or st.session_state.last_advanced_signature != advanced_signature):
        ai_mode = st.session_state.ai_enabled
        # --- loading indicator ---
        load_placeholder = st.empty()
        with load_placeholder.container():
            safe_query = safe_html(query)
            if ai_mode:
                loading_title = f'正在改写查询并检索 <strong style="color:#4f6f8f;">"{safe_query}"</strong> ...'
                loading_sub = f"算法: {algo} &nbsp;|&nbsp; 查询改写 · 检索 · 综述"
            else:
                loading_title = f'正在检索 <strong style="color:#4f6f8f;">"{safe_query}"</strong> ...'
                loading_sub = f"算法: {algo} &nbsp;|&nbsp; 检索中，请稍候"
            st.markdown(f"""
            <div style="text-align:center; padding:4rem 1rem;">
                <div class="search-spinner"></div>
                <p style="color:#52606d;font-size:0.95rem;margin-top:1.2rem;">
                    {loading_title}
                </p>
                <p style="color:#7c8792;font-size:0.78rem;margin-top:0.4rem;">
                    {loading_sub}
                </p>
            </div>
            """, unsafe_allow_html=True)

        # --- AI query enhancement ---
        search_query = query
        st.session_state.ai_enhanced_keywords = ""
        st.session_state.ai_enhance_explanation = ""
        st.session_state.ai_answer = ""

        if query.strip() and st.session_state.ai_enabled and st.session_state.ai_use_enhance:
            try:
                ai_cfg = AISearchConfig(
                    api_key=st.session_state.get("_ai_api_key", ""),
                    base_url=st.session_state.get("_ai_base_url", DEFAULT_AI_BASE_URL),
                    model=st.session_state.get("_ai_model", DEFAULT_AI_MODEL),
                )
                ai_engine = AISearchEngine(ai_cfg)
                if ai_engine.is_configured:
                    enhanced, explanation = ai_engine.enhance_query(query)
                    if enhanced and enhanced != query:
                        search_query = enhanced
                        st.session_state.ai_enhanced_keywords = enhanced
                        st.session_state.ai_enhance_explanation = explanation
            except Exception:
                pass  # fallback to original query

        results = []
        try:
            results = run_search_with_filters(
                st.session_state.docs, engine, search_query, algo, top_k, sbert_model, advanced_filters
            )
        except Exception as e:
            st.error(str(e))
            results = []

        load_placeholder.empty()

        st.session_state.last_results = results
        st.session_state.last_query = query
        st.session_state.last_algo = algo
        st.session_state.last_advanced_signature = advanced_signature

        if results:
            st.toast(f"🔍 找到 {len(results)} 条结果", icon="🔍")
        st.rerun()


# ---- Render results ----
if has_results:
    results = st.session_state.last_results

    # --- show "no results" empty state ---
    if not results:
        st.markdown("""
        <div style="text-align:center; padding:4rem 1rem;">
            <div style="font-size:4rem; margin-bottom:1rem;">📭</div>
            <div style="font-size:1.2rem; font-weight:600; color:#202936; margin-bottom:0.5rem;">
                没有找到匹配的论文
            </div>
            <div style="color:#52606d; font-size:0.9rem; margin-bottom:1.5rem;">
                试试更换关键词、减少过滤条件，或切换到其他检索算法
            </div>
        </div>
        """, unsafe_allow_html=True)

        # clear results to go back to hero on next action
        if st.button("🔄 返回首页", key="clear_empty", type="primary"):
            st.session_state.last_results = []
            st.session_state.last_query = ""
            st.session_state.last_advanced_signature = ""
            st.rerun()
        st.stop()

    # --- AI Query Enhancement display ---
    if st.session_state.ai_enabled and st.session_state.ai_enhanced_keywords:
        with st.expander("🔍 AI 查询增强", expanded=True):
            ec1, ec2 = st.columns([1, 3])
            with ec1:
                    st.markdown(f"""
                <div style="background:rgba(255,255,255,0.62);border:1px solid rgba(119,137,153,0.22);border-radius:8px;padding:0.8rem 1rem;">
                    <div style="font-size:0.7rem;color:#7c8792;text-transform:uppercase;letter-spacing:0.05em;">原始查询</div>
                    <div style="font-size:0.9rem;color:#202936;margin-top:0.3rem;">{safe_html(st.session_state.last_query)}</div>
                </div>
                """, unsafe_allow_html=True)
            with ec2:
                st.markdown(f"""
                <div style="background:#ffffff;border:1px solid #4f6f8f;border-radius:8px;padding:0.8rem 1rem;">
                    <div style="font-size:0.7rem;color:#4f6f8f;text-transform:uppercase;letter-spacing:0.05em;">增强检索词</div>
                    <div style="font-size:0.95rem;color:#202936;margin-top:0.3rem;font-weight:600;">{safe_html(st.session_state.ai_enhanced_keywords)}</div>
                </div>
                """, unsafe_allow_html=True)
            if st.session_state.ai_enhance_explanation:
                st.caption(st.session_state.ai_enhance_explanation)

    # --- AI Answer Generation ---
    if (st.session_state.ai_enabled and st.session_state.ai_use_answer
            and results and not st.session_state.ai_answer):
        # Loading indicator
        loading_placeholder = st.empty()
        with loading_placeholder.container():
            st.markdown("""
            <div class="ai-answer-loading">
                <div class="search-spinner" style="width:24px;height:24px;border-width:2px;margin:0;"></div>
                <span class="loading-text">AI 正在分析检索结果...</span>
            </div>
            """, unsafe_allow_html=True)

        try:
            ai_cfg = AISearchConfig(
                api_key=st.session_state.get("_ai_api_key", ""),
                base_url=st.session_state.get("_ai_base_url", DEFAULT_AI_BASE_URL),
                model=st.session_state.get("_ai_model", DEFAULT_AI_MODEL),
            )
            length_map = {"简短": 1024, "标准": 2048, "详细": 4096}
            ai_cfg.max_tokens = length_map.get(st.session_state.ai_answer_length, 2048)
            ai_engine = AISearchEngine(ai_cfg)

            top_papers = [doc for _, _, doc in results[:10]]
            full_answer = ""

            # Stream answer with proper markdown rendering
            answer_container = st.empty()
            for chunk in ai_engine.generate_answer_stream(query, top_papers):
                full_answer += chunk
                with answer_container.container():
                    st.markdown("""
                    <div class="ai-answer-card">
                        <div class="ai-answer-card-header">结果综述</div>
                    """, unsafe_allow_html=True)
                    st.markdown(full_answer)
                    st.markdown("</div>", unsafe_allow_html=True)

            loading_placeholder.empty()
            st.session_state.ai_answer = full_answer
        except Exception as e:
            loading_placeholder.empty()
            st.warning(f"AI 分析暂时不可用: {e}")
    elif st.session_state.ai_answer:
        with st.container():
            st.markdown("""
            <div class="ai-answer-card">
                <div class="ai-answer-card-header">结果综述</div>
            """, unsafe_allow_html=True)
            st.markdown(st.session_state.ai_answer)
            st.markdown("</div>", unsafe_allow_html=True)

    # --- AI chat entry + section divider ---
    if st.session_state.ai_enabled and st.session_state.ai_use_chat:
        cc1, cc2, cc3 = st.columns([1, 2, 1])
        with cc2:
            if st.button("💬 对检索结果提问", key="enter_chat", type="primary", use_container_width=True):
                st.session_state.page = "chat"
                st.rerun()

    st.markdown(f"""
    <div class="results-toolbar">
        <div>
            <div class="section-label">Search Results</div>
            <div class="section-title">论文检索结果</div>
        </div>
        <div class="results-summary">查询：<strong>{safe_html(st.session_state.last_query or query or '高级检索')}</strong></div>
    </div>
    """, unsafe_allow_html=True)

    # Sort
    sort_labels = {"相关性": "Relevance", "日期": "Date", "标题": "Title"}
    current_sort_label = next(
        label for label, value in sort_labels.items()
        if value == st.session_state.sort_by
    )
    if hasattr(st, "segmented_control"):
        selected_sort_label = st.segmented_control(
            "排序",
            options=list(sort_labels.keys()),
            default=current_sort_label,
            label_visibility="collapsed",
            key="sort_segmented",
        )
        st.session_state.sort_by = sort_labels[selected_sort_label]
    else:
        sort_col1, sort_col2, sort_col3, sort_spacer = st.columns([1, 1, 1, 6])
        with sort_col1:
            if st.button("相关性", key="sort_rel",
                         type="primary" if st.session_state.sort_by == "Relevance" else "secondary",
                         use_container_width=True):
                st.session_state.sort_by = "Relevance"
                st.rerun()
        with sort_col2:
            if st.button("日期", key="sort_date",
                         type="primary" if st.session_state.sort_by == "Date" else "secondary",
                         use_container_width=True):
                st.session_state.sort_by = "Date"
                st.rerun()
        with sort_col3:
            if st.button("标题", key="sort_title",
                         type="primary" if st.session_state.sort_by == "Title" else "secondary",
                         use_container_width=True):
                st.session_state.sort_by = "Title"
                st.rerun()

    # Apply sort
    sorted_results = list(results)
    if st.session_state.sort_by == "Date":
        sorted_results.sort(key=lambda x: x[2].get("update_date", ""), reverse=True)
    elif st.session_state.sort_by == "Title":
        sorted_results.sort(key=lambda x: x[2].get("title", ""))

    # Results bar
    st.markdown(f"""
    <div class="results-bar">
        <span>📊 共 <strong style="color:#4f6f8f;">{len(results)}</strong> 条结果 &nbsp;|&nbsp;
        算法: <strong>{algo}</strong> &nbsp;|&nbsp;
        排序: <strong>{st.session_state.sort_by}</strong></span>
    </div>
    """, unsafe_allow_html=True)

    # Category filter pills (extract from results)
    all_cats = set()
    for _, _, doc in sorted_results:
        for c in (doc.get("categories") or "").split():
            all_cats.add(c)
    top_cats = sorted(all_cats)[:12]

    if "cat_filter" not in st.session_state:
        st.session_state.cat_filter = None

    if len(top_cats) > 1 and hasattr(st, "pills"):
        cat_options = ["全部"] + top_cats[:10]
        default_cat = st.session_state.cat_filter if st.session_state.cat_filter in top_cats else "全部"
        selected_cat = st.pills(
            "分类",
            options=cat_options,
            default=default_cat,
            label_visibility="collapsed",
            key="cat_pills",
        )
        st.session_state.cat_filter = None if selected_cat == "全部" else selected_cat
    elif len(top_cats) > 1:
        filter_cols = st.columns(min(len(top_cats), 10))
        for i, cat in enumerate(top_cats[:10]):
            with filter_cols[i]:
                is_active = st.session_state.cat_filter == cat
                if st.button(cat, key=f"filt_{cat}",
                            type="primary" if is_active else "secondary",
                            use_container_width=True):
                    st.session_state.cat_filter = None if is_active else cat
                    st.rerun()

    # Apply category filter
    if st.session_state.cat_filter:
        sorted_results = [
            (idx, score, doc) for idx, score, doc in sorted_results
            if st.session_state.cat_filter in (doc.get("categories") or "")
        ]

    if not sorted_results:
        st.markdown(f"""
        <div style="text-align:center; padding:3rem 1rem;">
            <div style="font-size:3rem; margin-bottom:0.8rem;">🏷️</div>
            <div style="font-size:1.1rem; font-weight:600; color:#202936; margin-bottom:0.4rem;">
                分类 <strong style="color:#4f6f8f;">{st.session_state.cat_filter}</strong> 下没有结果
            </div>
            <div style="color:#52606d; font-size:0.85rem;">
                点击上方分类标签可清除筛选
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # Render cards in a single-column literature database layout.
    for i, (_, score, doc) in enumerate(sorted_results, start=1):
        render_card(i, doc, score)
