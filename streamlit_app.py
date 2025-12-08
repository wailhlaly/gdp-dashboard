import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from streamlit_lightweight_charts import renderLightweightCharts

# --- PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="Mudarib Pro (TV Style)")

# --- STYLING ---
st.markdown("""
<style>
    .stApp { background-color: #131722; }
    h1 { color: #d1d4dc !important; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Mudarib Pro | TradingView Engine")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Settings")
    symbol = st.text_input("Symbol", "2222.SR")
    period = st.selectbox("Lookback", ["1y", "2y", "5y"])
    timeframe = st.selectbox("Timeframe", ["1d", "1wk"])

# --- DATA ENGINE ---
@st.cache_data
def get_tv_data(ticker, p, i):
    try:
        df = yf.download(ticker, period=p, interval=i, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        if df.empty: return None

        # Reset index to make Date a column
        df = df.reset_index()
        
        # Calculate Logic (ICT Imbalance)
        df['Body'] = abs(df['Close'] - df['Open'])
        threshold = df['Body'].quantile(0.95)
        df['Imbalance'] = df['Body'] > threshold
        
        return df
    except:
        return None

df = get_tv_data(symbol, period, timeframe)

if df is not None:
    # --- PREPARE DATA FOR LIGHTWEIGHT CHARTS ---
    # المكتبة تحتاج تنسيق خاص جداً (List of Dictionaries)
    
    # 1. Candlestick Data
    candles = []
    for index, row in df.iterrows():
        # تحويل التاريخ إلى سترينج
        time_str = row['Date'].strftime('%Y-%m-%d')
        candles.append({
            "time": time_str,
            "open": row['Open'],
            "high": row['High'],
            "low": row['Low'],
            "close": row['Close']
        })

    # 2. Markers (Imbalance/ICT)
    markers = []
    imbalance_rows = df[df['Imbalance']]
    for index, row in imbalance_rows.iterrows():
        time_str = row['Date'].strftime('%Y-%m-%d')
        markers.append({
            "time": time_str,
            "position": "aboveBar",
            "color": "#e91e63", # لون مميز (وردي/أحمر)
            "shape": "arrowDown",
            "text": "FVG"
        })

    # --- CHART CONFIGURATION (TradingView Style) ---
    chartOptions = {
        "layout": {
            "backgroundColor": "#131722", # لون خلفية تريدنج فيو الداكن
            "textColor": "#d1d4dc"
        },
        "grid": {
            "vertLines": {"color": "rgba(42, 46, 57, 0.5)", "style": 1}, # خطوط شبكة خفيفة
            "horzLines": {"color": "rgba(42, 46, 57, 0.5)", "style": 1}
        },
        "crosshair": {
            "mode": 1 # Magnet mode
        },
        "priceScale": {
            "borderColor": "rgba(197, 203, 206, 0.8)"
        },
        "timeScale": {
            "borderColor": "rgba(197, 203, 206, 0.8)",
            "timeVisible": True
        }
    }

    # تعريف السلسلة (Candlestick Series)
    seriesCandlestickChart = [
        {
            "type": 'Candlestick',
            "data": candles,
            "options": {
                "upColor": '#26a69a',      # أخضر تريدنج فيو
                "downColor": '#ef5350',    # أحمر تريدنج فيو
                "borderVisible": False,
                "wickUpColor": '#26a69a',
                "wickDownColor": '#ef5350'
            },
            "markers": markers  # إضافة علامات الـ ICT
        }
    ]

    # --- RENDER THE CHART ---
    st.subheader(f"Chart: {symbol}")
    
    # رسم الشارت
    renderLightweightCharts(
        options=chartOptions,
        series=seriesCandlestickChart,
        height=600 # ارتفاع ممتاز للجوال
    )

    st.success("تم استخدام محرك TradingView (Lightweight Charts) لعرض سلس واحترافي.")

else:
    st.error("Data not found.")

