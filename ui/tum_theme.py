"""TUM corporate design tokens and Streamlit styling helpers."""

from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

# Official TUM corporate design colors (CD manual)
TUM_BLUE = '#0065BD'
TUM_BLUE_DARK = '#005293'
TUM_BLUE_LIGHT = '#E8F1F8'
TUM_BLACK = '#000000'
TUM_GRAY_DARK = '#333333'
TUM_GRAY = '#666666'
TUM_GRAY_LIGHT = '#CCCCCC'
TUM_WHITE = '#FFFFFF'
TUM_BG = '#F5F8FA'

FONT_FAMILY = 'Arial, Helvetica, sans-serif'
FONT_FAMILY_PLOTLY = 'Arial'

# Chart palette — TUM blue for inventory/holding; semantic colors retained for readability
BG = TUM_BG
PANEL = TUM_WHITE
GRID = TUM_GRAY_LIGHT
TEXT = TUM_GRAY_DARK
MUTED = TUM_GRAY
C_INV = TUM_BLUE
C_DEM = '#E57373'
C_UNM = '#C62828'
C_ORD = '#F57C00'
C_HLD = '#4A90D9'
C_ORC = '#FBC02D'
C_LST = '#D32F2F'
C_HOR = TUM_BLUE_DARK
FUT_SHADE = '#FFF9E6'

ASSETS_DIR = Path(__file__).resolve().parent / 'assets'
LOGO_PATH = ASSETS_DIR / 'tum_logo_white.svg'


def _logo_data_uri() -> str:
    if LOGO_PATH.exists():
        encoded = base64.b64encode(LOGO_PATH.read_bytes()).decode('ascii')
        return f'data:image/svg+xml;base64,{encoded}'
    return ''


def inject_tum_styles() -> None:
    """Inject scoped CSS — avoid broad selectors that break Streamlit emotion-cache layout."""
    st.markdown(
        f"""
        <style>
          html, body,
          [data-testid="stAppViewContainer"],
          [data-testid="stSidebar"] {{
            font-family: {FONT_FAMILY};
          }}
          .main p, .main li, .main label,
          .main h1, .main h2, .main h3, .main h4,
          [data-testid="stSidebar"] p,
          [data-testid="stSidebar"] label,
          [data-testid="stSidebar"] h1,
          [data-testid="stSidebar"] h2,
          [data-testid="stSidebar"] h3 {{
            font-family: {FONT_FAMILY};
          }}
          [data-testid="stSidebar"] {{
            background-color: {TUM_BLUE_LIGHT};
            border-right: 1px solid {TUM_GRAY_LIGHT};
          }}
          [data-testid="stSidebar"] h1,
          [data-testid="stSidebar"] h2,
          [data-testid="stSidebar"] h3 {{
            color: {TUM_BLUE_DARK};
          }}
          [data-testid="stMetric"] {{
            background: {TUM_WHITE};
            border-top: 3px solid {TUM_BLUE};
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
          }}
          [data-testid="stMetricLabel"] p {{
            color: {TUM_GRAY};
            font-size: 0.8rem;
          }}
          [data-testid="stMetricValue"] {{
            color: {TUM_BLUE_DARK};
          }}
          div.stButton > button[kind="primary"] {{
            background-color: {TUM_BLUE};
            border-color: {TUM_BLUE};
            color: {TUM_WHITE};
            font-family: {FONT_FAMILY};
            font-weight: 600;
          }}
          div.stButton > button[kind="primary"]:hover {{
            background-color: {TUM_BLUE_DARK};
            border-color: {TUM_BLUE_DARK};
          }}
          hr {{
            border-color: {TUM_GRAY_LIGHT};
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_tum_header(title: str = 'PPO Inventory Optimizer',
                      subtitle: str = 'PwC x TUM · Reinforcement Learning Agent') -> None:
    """Render header in an isolated iframe to avoid Streamlit markdown text overlay."""
    logo_uri = _logo_data_uri()
    logo_html = (
        f'<img src="{logo_uri}" alt="TUM Logo" style="height:42px;width:auto;" />'
        if logo_uri else
        '<span style="font-size:28px;font-weight:700;letter-spacing:0.05em;">TUM</span>'
    )
    components.html(
        f"""
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="utf-8" />
          <style>
            * {{ box-sizing: border-box; margin: 0; padding: 0; }}
            body {{
              font-family: {FONT_FAMILY};
              background: transparent;
            }}
            .tum-header {{
              background: linear-gradient(90deg, {TUM_BLUE_DARK} 0%, {TUM_BLUE} 100%);
              color: {TUM_WHITE};
              padding: 16px 24px;
              border-radius: 6px;
              display: flex;
              align-items: center;
              gap: 20px;
              box-shadow: 0 2px 8px rgba(0,0,0,0.12);
            }}
            .tum-header-title {{
              font-size: 22px;
              font-weight: 700;
              line-height: 1.25;
            }}
            .tum-header-subtitle {{
              font-size: 14px;
              opacity: 0.92;
              margin-top: 4px;
            }}
          </style>
        </head>
        <body>
          <div class="tum-header">
            <div>{logo_html}</div>
            <div>
              <div class="tum-header-title">{title}</div>
              <div class="tum-header-subtitle">{subtitle}</div>
            </div>
          </div>
        </body>
        </html>
        """,
        height=88,
        scrolling=False,
    )
