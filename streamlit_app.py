import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="Mudarib v3 - Mobile Pro")

# --- STYLING (Mobile Optimized) ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    /* تحسين الهوامش للموبايل */
    .block-container { padding-top: 1rem; padding-bottom: 5rem; }
    h1 { font-size: 1.5rem !important; color: #00ffcc !important; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.header("⚙️ إعدادات السهم")
symbol_input = st.sidebar.text_input("رمز السهم", value="2222").upper()
market_suffix = st.sidebar.selectbox("السوق", [".SR", "", ".L", ".HK"], index=0)
timeframe = st.sidebar.selectbox("الفاصي", ["1d", "1wk"], index=0)
lookback = st.sidebar.slider("المدة (سنوات)", 1, 5, 2)

full_symbol = f"{symbol_input}{market_suffix}" if market_suffix else symbol_input

# --- 1) DATA LOADING ---
@st.cache_data
def get_data(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df if not df.empty else None
    except:
        return None

data = get_data(full_symbol, f"{lookback}y", timeframe)

if data is not None:
    # --- 2) ANALYSIS ---
    # Smart Imbalance (Top 5% only)
    data['Body'] = abs(data['Close'] - data['Open'])
    threshold = data['Body'].quantile(0.95)
    data['Institutional_Move'] = data['Body'] > threshold

    # Pivots
    data['Pivot_High'] = data['High'].rolling(20, center=True).max()
    data['Pivot_Low'] = data['Low'].rolling(20, center=True).min()

    # --- 3) INTERACTIVE CHART SETUP ---
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=data.index, open=data['Open'], high=data['High'],
        low=data['Low'], close=data['Close'], name='Price'
    ))

    # Institutional Markers
    sig_moves = data[data['Institutional_Move']]
    if not sig_moves.empty:
        fig.add_trace(go.Scatter(
            x=sig_moves.index, y=sig_moves['High'],
            mode='markers', 
            marker=dict(color='yellow', size=8, symbol='diamond-open', line=dict(width=2)),
            name='Institutional Imbalance'
        ))

    # Support/Resistance Lines
    last_h = data['Pivot_High'].dropna().iloc[-1]
    last_l = data['Pivot_Low'].dropna().iloc[-1]
    fig.add_hline(y=last_h, line_dash="dash", line_color="red", annotation_text="Res", annotation_position="top right")
    fig.add_hline(y=last_l, line_dash="dash", line_color="green", annotation_text="Supp", annotation_position="bottom right")

    # --- KEY FIX: MOBILE LAYOUT & INTERACTIVITY ---
    fig.update_layout(
        title=f"{full_symbol}",
        template="plotly_dark",
        height=700,  # جعل الشارت أطول للموبايل
        xaxis_rangeslider_visible=False,
        dragmode='pan',  # السحب بالإصبع يحرك الشارت بدلاً من الزوم
        margin=dict(l=10, r=10, t=40, b=40), # استغلال كامل الشاشة
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # --- 4) DISPLAY WITH CONFIG ---
    st.title("⚡ Mudarib v3")
    
    # Metrics row
    curr = data['Close'].iloc[-1]
    c1, c2 = st.columns(2)
    c1.metric("السعر الحالي", f"{curr:.2f}")
    c2.metric("الحالة", "تجميع" if curr < last_h and curr > last_l else "اتجاه")

    # THE MAGIC FIX: config settings
    st.plotly_chart(fig, use_container_width=True, config={
        'scrollZoom': True,       # السماح بالزوم عبر اللمس
        'displayModeBar': True,   # إظهار شريط الأدوات
        'displaylogo': False,
        'modeBarButtonsIfNeeded': ['pan2d', 'zoomIn2d', 'zoomOut2d', 'resetScale2d']
    })

    st.caption("💡 نصيحة: استخدم إصبعين للتكبير/التصغير، وإصبع واحد لتحريك الشارت.")

else:
    st.error("لم يتم العثور على البيانات. تأكد من الرمز.")
