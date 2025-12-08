import streamlit as st
import yfinance as yf
import pandas as pd
import streamlit_lightweight_charts as slc # استيراد المكتبة

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Mudarib V5 Stable")

st.markdown("""
<style>
    .stApp { background-color: #131722; }
    h1, h2, h3, p, div, span { color: #d1d4dc; }
    /* إصلاح ألوان القوائم */
    .stTextInput input, .stSelectbox div[data-baseweb="select"] {
        color: #d1d4dc !important;
        background-color: #2a2e39 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Mudarib V5 | TradingView Engine")

# --- 2. SIDEBAR ---
with st.sidebar:
    st.header("Settings")
    # إدخال الرمز والسوق
    c1, c2 = st.columns([2, 1])
    with c1:
        symbol = st.text_input("Symbol", "2222")
    with c2:
        suffix = st.selectbox("Market", [".SR", "", ".L", ".HK"])
    
    full_ticker = f"{symbol}{suffix}" if suffix else symbol
    
    period = st.selectbox("Period", ["1y", "2y", "5y"])
    timeframe = st.selectbox("Interval", ["1d", "1wk"])

# --- 3. DATA ENGINE ---
@st.cache_data(ttl=300)
def get_data(ticker, p, i):
    try:
        df = yf.download(ticker, period=p, interval=i, progress=False)
        
        # FIX 1: MultiIndex Issue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        if df.empty: return None
        
        df = df.reset_index()
        # توحيد أسماء الأعمدة لتسهيل التعامل
        df.columns = [c.lower() for c in df.columns]
        
        # ICT Logic: Calculate Imbalance
        df['body'] = abs(df['close'] - df['open'])
        threshold = df['body'].quantile(0.95) # Top 5% candles
        df['imbalance'] = df['body'] > threshold
        
        return df
    except Exception:
        return None

df = get_data(full_ticker, period, timeframe)

if df is not None:
    # --- 4. DATA SERIALIZATION (THE FIX) ---
    # هنا نقوم بتحويل أرقام Numpy إلى Python Floats لمنع خطأ TypeError
    
    candles = []
    markers = []
    
    for _, row in df.iterrows():
        # تحويل التاريخ إلى نص
        t_str = row['date'].strftime('%Y-%m-%d')
        
        # إضافة الشموع (مع تحويل float إجباري)
        candles.append({
            "time": t_str,
            "open": float(row['open']),
            "high": float(row['high']),
            "low": float(row['low']),
            "close": float(row['close']),
        })
        
        # إضافة العلامات (فقط للشموع القوية)
        if row['imbalance']:
            markers.append({
                "time": t_str,
                "position": "aboveBar",
                "color": "#e91e63", # لون فوشي واضح
                "shape": "arrowDown",
                "text": "FVG",
                "size": 1
            })

    # --- 5. CHART OPTIONS ---
    chartOptions = {
        "layout": {
            "backgroundColor": "#131722",
            "textColor": "#d1d4dc"
        },
        "grid": {
            "vertLines": {"color": "rgba(42, 46, 57, 0.5)", "style": 1},
            "horzLines": {"color": "rgba(42, 46, 57, 0.5)", "style": 1}
        },
        "timeScale": {
            "borderColor": "#485c7b",
        }
    }

    series = [
        {
            "type": 'Candlestick',
            "data": candles,
            "options": {
                "upColor": '#26a69a',
                "downColor": '#ef5350',
                "borderVisible": False,
                "wickUpColor": '#26a69a',
                "wickDownColor": '#ef5350'
            },
            "markers": markers
        }
    ]

    # --- 6. RENDER ---
    curr_price = float(df['close'].iloc[-1])
    prev_price = float(df['close'].iloc[-2])
    chg = ((curr_price - prev_price)/prev_price)*100
    color = "green" if chg > 0 else "red"
    
    st.markdown(f"### {full_ticker} : {curr_price:.2f} <span style='color:{color}'>({chg:+.2f}%)</span>", unsafe_allow_html=True)
    
    # استدعاء الدالة
    slc.renderLightweightCharts(
        options=chartOptions,
        series=series,
        height=500
    )

else:
    st.error(f"Data Not Found for {full_ticker}. Check symbol/suffix.")