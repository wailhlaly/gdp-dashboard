import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# إعدادات الصفحة
st.set_page_config(layout="wide", page_title="Mudarib v3 - Pro Terminal")

# --- CSS Styling for Professional Look ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    h1, h2, h3 { color: #00ffcc !important; font-family: 'Courier New', monospace; }
    .metric-card { background-color: #1c1f26; padding: 10px; border-radius: 5px; border-left: 5px solid #00ffcc; }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title("⚡ Mudarib v3 | Institutional Analysis Terminal")
st.markdown("---")

# --- SIDEBAR INPUTS ---
st.sidebar.header("🔍 Stock Settings")
symbol_input = st.sidebar.text_input("Stock Symbol", value="AAPL").upper()
market_suffix = st.sidebar.selectbox("Market/Exchange Suffix", ["", ".SR", ".L", ".HK", ".NS"], index=0)
timeframe = st.sidebar.selectbox("Timeframe", ["1d", "1wk", "1mo"], index=0)
lookback_years = st.sidebar.slider("Lookback Period (Years)", 1, 10, 3)

full_symbol = f"{symbol_input}{market_suffix}" if market_suffix else symbol_input

# --- 1) DATA FETCHING & VALIDITY (Fixed for yfinance Update) ---
@st.cache_data
def load_data(ticker, period, interval):
    try:
        # تحميل البيانات
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        
        # --- FIX: معالجة مشكلة MultiIndex ---
        # إذا كانت الأعمدة معقدة (تحتوي على الرمز والسعر)، نقوم بتبسيطها
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        return data
    except Exception as e:
        return None

# جلب البيانات
data = load_data(full_symbol, f"{lookback_years}y", timeframe)

if data is not None and not data.empty:
    current_price = data['Close'].iloc[-1]
    
    # --- 2) MULTI-SCHOOL ALGORITHMS ---
    
    # >> A. ICT/SMC Logic (Fair Value Gaps - Imbalance detection)
    def identify_fvg(df):
        # سنستخدم طريقة الكشف عن الـ Imbalance بناءً على حجم الجسم مقارنة بالمتوسط
        # لتجنب أخطاء المصفوفات المعقدة
        df = df.copy() # العمل على نسخة لتفادي التحذيرات
        df['Body'] = abs(df['Close'] - df['Open'])
        df['AvgBody'] = df['Body'].rolling(20).mean()
        # نعتبر الشمعة Imbalance إذا كان جسمها أكبر من 1.5 ضعف المتوسط
        df['Imbalance'] = np.where(df['Body'] > 1.5 * df['AvgBody'], True, False)
        return df

    data = identify_fvg(data)

    # >> B. Support & Resistance (Pivot Points)
    data['Pivot_High'] = data['High'].rolling(window=20, center=True).max()
    data['Pivot_Low'] = data['Low'].rolling(window=20, center=True).min()

    # >> C. Volume Anomaly (Institutional Activity)
    vol_avg = data['Volume'].rolling(20).mean()
    data['Vol_Spike'] = data['Volume'] > 2 * vol_avg

    # --- 3) VISUALIZATION (THE CHART) ---
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'], high=data['High'],
        low=data['Low'], close=data['Close'],
        name='Price'
    ))

    # Plot FVG / Imbalance Candles (Markers)
    # التأكد من وجود بيانات imbalance قبل الرسم
    if 'Imbalance' in data.columns:
        imbalance_data = data[data['Imbalance']]
        if not imbalance_data.empty:
            fig.add_trace(go.Scatter(
                x=imbalance_data.index, y=imbalance_data['High'],
                mode='markers', marker=dict(color='yellow', size=5, symbol='diamond'),
                name='Imbalance/Momentum (ICT)'
            ))

    # Plot Key Structure Levels (Support/Resistance)
    last_pivots_h = data['Pivot_High'].dropna().unique()[-3:]
    last_pivots_l = data['Pivot_Low'].dropna().unique()[-3:]

    for level in last_pivots_h:
        fig.add_hline(y=level, line_dash="dash", line_color="red", annotation_text="Key Res (SMC)", annotation_position="top right")
    
    for level in last_pivots_l:
        fig.add_hline(y=level, line_dash="dash", line_color="green", annotation_text="Key Supp (SMC)", annotation_position="bottom right")

    # Layout Updates
    fig.update_layout(
        title=f"{full_symbol} - Institutional Analysis Chart",
        yaxis_title="Price",
        xaxis_title="Date",
        template="plotly_dark",
        height=700,
        xaxis_rangeslider_visible=False
    )

    # --- 4) GLOBAL CORRELATION MODEL ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("🌍 Global Macro Data")
    
    macro_tickers = {"Gold": "GC=F", "Oil": "CL=F", "S&P500": "^GSPC"} # DXY removed to prevent errors if data missing
    correlations = {}
    
    # Calculate Correlation
    for name, ticker in macro_tickers.items():
        try:
            macro_data = yf.download(ticker, period="1y", interval="1d", progress=False)
            if isinstance(macro_data.columns, pd.MultiIndex):
                macro_data.columns = macro_data.columns.get_level_values(0)
            
            macro_close = macro_data['Close']
            
            # Align data indices
            aligned_data = pd.concat([data['Close'], macro_close], axis=1).dropna()
            aligned_data.columns = ['Stock', 'Macro']
            
            if len(aligned_data) > 10: # Ensure enough data points
                corr = aligned_data.corr().iloc[0, 1]
                correlations[name] = corr
            else:
                correlations[name] = 0.0
        except:
            correlations[name] = 0.0

    # --- DISPLAY SECTIONS ---
    
    # Top Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"{current_price:.2f}")
    
    # Trend Bias Logic Safe Check
    trend_val = "Neutral"
    if len(data) > 20:
        trend_val = "Bullish" if data['Close'].iloc[-1] > data['Close'].iloc[-20] else "Bearish"
    col2.metric("Trend Bias", trend_val)
    
    # Vol Spike Count
    vol_spike_count = int(data['Vol_Spike'].iloc[-20:].sum()) if 'Vol_Spike' in data.columns else 0
    col3.metric("Vol Spikes (Last 20d)", vol_spike_count)
    
    # RSI Calc
    delta = data['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    rsi = 100 - (100 / (1 + rs))
    col4.metric("RSI (14)", f"{rsi.iloc[-1]:.1f}")

    # Main Chart
    st.plotly_chart(fig, use_container_width=True)

    # Analysis Layout
    c1, c2 = st.columns([2, 1])

    with c1:
        st.subheader("📋 Mudarib v3 Technical Report")
        pivot_ref = data['Pivot_High'].iloc[-25] if len(data) > 25 and not pd.isna(data['Pivot_High'].iloc[-25]) else current_price
        
        st.markdown(f"""
        **1. Market Structure (SMC/Price Action):**
        * السعر الحالي: **{current_price:.2f}**
        * مناطق السيولة (Liquidity) تتركز عند القيعان السابقة الموضحة بالخطوط الخضراء المتقطعة.
        
        **2. Volume & Imbalance (ICT):**
        * تم الكشف عن زخم مؤسساتي في **{vol_spike_count}** جلسات خلال الشهر الماضي.
        * الماس الأصفر على الشارت يوضح شموع الـ Imbalance القوية.
        """)

    with c2:
        st.subheader("🔗 Global Correlations")
        st.write("ارتباط السهم بالمؤشرات العالمية (1 سنة):")
        for name, corr in correlations.items():
            color = "green" if corr > 0.5 else "red" if corr < -0.5 else "white"
            st.markdown(f"**{name}:** <span style='color:{color}'>{corr:.2f}</span>", unsafe_allow_html=True)

else:
    st.error(f"Could not load data for symbol: {full_symbol}. Please check spelling or internet connection.")
