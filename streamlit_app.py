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

# --- 1) DATA FETCHING & VALIDITY ---
@st.cache_data
def load_data(ticker, period, interval):
    try:
        data = yf.download(ticker, period=period, interval=interval)
        return data
    except Exception as e:
        return None

# جلب البيانات
data = load_data(full_symbol, f"{lookback_years}y", timeframe)

if data is not None and not data.empty:
    current_price = data['Close'].iloc[-1]
    
    # --- 2) MULTI-SCHOOL ALGORITHMS ---
    
    # >> A. ICT/SMC Logic (Fair Value Gaps - FVG)
    def identify_fvg(df):
        fvg_zones = []
        for i in range(2, len(df)):
            # Bullish FVG: Low of candle i-2 > High of candle i
            if df['Low'].iloc[i-2] > df['High'].iloc[i]: 
                 # This is a simplifed logic for visual gap
                 pass
            
            # Simple Gap Logic for FVG (Bullish)
            # Candle 0 High < Candle 2 Low
            curr_high = df['High'].iloc[i]
            prev2_low = df['Low'].iloc[i-2]
            prev1_close = df['Close'].iloc[i-1]
            prev1_open = df['Open'].iloc[i-1]
            
            # Big Bullish Candle check
            if prev1_close > prev1_open and prev2_low > curr_high:
                 # Potentially huge gap, but let's stick to standard FVG
                 pass

        # Let's use a simpler heuristic for visualization:
        # Detect Imbalance (Large candles with little overlap)
        df['Body'] = abs(df['Close'] - df['Open'])
        df['AvgBody'] = df['Body'].rolling(20).mean()
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
    imbalance_dates = data[data['Imbalance']].index
    imbalance_prices = data[data['Imbalance']]['High']
    fig.add_trace(go.Scatter(
        x=imbalance_dates, y=imbalance_prices,
        mode='markers', marker=dict(color='yellow', size=5, symbol='diamond'),
        name='Imbalance/Momentum (ICT)'
    ))

    # Plot Key Structure Levels (Support/Resistance)
    # We take the last 3 distinct pivot levels to avoid clutter
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
    
    macro_tickers = {"Gold": "GC=F", "Oil": "CL=F", "S&P500": "^GSPC", "DXY": "DX-Y.NYB"}
    correlations = {}
    
    # Calculate Correlation
    for name, ticker in macro_tickers.items():
        macro_data = yf.download(ticker, period="1y", interval="1d", progress=False)['Close']
        # Align data indices
        aligned_data = pd.concat([data['Close'], macro_data], axis=1).dropna()
        aligned_data.columns = ['Stock', 'Macro']
        corr = aligned_data.corr().iloc[0, 1]
        correlations[name] = corr

    # --- DISPLAY SECTIONS ---
    
    # Top Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"{current_price:.2f}")
    col2.metric("Trend Bias", "Bullish" if data['Close'].iloc[-1] > data['Close'].iloc[-20] else "Bearish")
    col3.metric("Vol Spikes (Last 20d)", int(data['Vol_Spike'].iloc[-20:].sum()))
    col4.metric("RSI (Approx)", round(100 - (100 / (1 + (data['Close'].diff().clip(lower=0).rolling(14).mean() / data['Close'].diff().clip(upper=0).abs().rolling(14).mean()).iloc[-1])), 2))

    # Main Chart
    st.plotly_chart(fig, use_container_width=True)

    # Analysis Layout
    c1, c2 = st.columns([2, 1])

    with c1:
        st.subheader("📋 Mudarib v3 Technical Report")
        st.markdown(f"""
        **1. Market Structure (SMC/Price Action):**
        * السعر الحالي يتداول {'فوق' if current_price > data['Pivot_High'].iloc[-25] else 'تحت'} آخر قمة هيكلية (Swing High) المسجلة مؤخراً.
        * مناطق السيولة (Liquidity) تتركز عند القيعان السابقة الموضحة بالخطوط الخضراء المتقطعة.
        
        **2. Volume & Imbalance (ICT):**
        * تم اكتشاف عدد **{int(data['Imbalance'].iloc[-20:].sum())}** شموع زخم (Imbalance) في الفترة الأخيرة، مما يدل على تدخل مؤسساتي.
        * الماس الأصفر على الشارت يوضح أماكن الزخم العالي.

        **3. Wyckoff Perspective:**
        * بناءً على تحليل الحجم والسعر، إذا كان السعر في نطاق عرضي مع حجم منخفض، فقد نكون في مرحلة (Phase B - Building Cause).
        """)

    with c2:
        st.subheader("🔗 Global Correlations")
        st.write("ارتباط السهم بالمؤشرات العالمية (1 سنة):")
        for name, corr in correlations.items():
            color = "green" if corr > 0.5 else "red" if corr < -0.5 else "white"
            st.markdown(f"**{name}:** <span style='color:{color}'>{corr:.2f}</span>", unsafe_allow_html=True)
            
        st.info("الارتباط الإيجابي القوي (> 0.7) يعني أن السهم يتحرك مع المؤشر.")

else:
    st.error(f"Could not load data for symbol: {full_symbol}. Please check the symbol or market suffix.")

