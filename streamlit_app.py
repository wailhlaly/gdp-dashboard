import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="Mudarib v3 - Pro Terminal")

# --- STYLING ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    h1, h2, h3 { color: #00ffcc !important; font-family: 'Segoe UI', sans-serif; }
    .stMetric { background-color: #1c1f26; padding: 10px; border-radius: 5px; border: 1px solid #333; }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title("⚡ Mudarib v3 | Institutional Analysis")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Market Data")
symbol_input = st.sidebar.text_input("Symbol", value="2222").upper()
market_suffix = st.sidebar.selectbox("Market", [".SR", "", ".L", ".HK"], index=0)
timeframe = st.sidebar.selectbox("Timeframe", ["1d", "1wk"], index=0)
lookback = st.sidebar.slider("History (Years)", 1, 5, 2)

full_symbol = f"{symbol_input}{market_suffix}" if market_suffix else symbol_input

# --- 1) ROBUST DATA LOADING ---
@st.cache_data
def get_clean_data(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        
        # FIX: Flatten MultiIndex columns if they exist (The cause of ValueError)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if df.empty: return None
        
        return df
    except:
        return None

data = get_clean_data(full_symbol, f"{lookback}y", timeframe)

if data is not None:
    # --- 2) ADVANCED ANALYSIS ALGORITHMS ---
    
    # A. SMART IMBALANCE (CLEAN VERSION)
    # بدلاً من تحديد كل شمعة، نحدد فقط الشموع التي حجم جسمها أكبر من 95% من الشموع الأخرى
    data['Body'] = abs(data['Close'] - data['Open'])
    threshold = data['Body'].quantile(0.95) # Top 5% only
    data['Institutional_Move'] = data['Body'] > threshold

    # B. PIVOT POINTS (Structural Levels)
    data['Pivot_High'] = data['High'].rolling(20, center=True).max()
    data['Pivot_Low'] = data['Low'].rolling(20, center=True).min()

    # --- 3) PROFESSIONAL VISUALIZATION ---
    fig = go.Figure()

    # 1. Main Price
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'], high=data['High'],
        low=data['Low'], close=data['Close'],
        name='Price Action'
    ))

    # 2. Institutional Zones (Filtered - Not Spammy)
    # نرسم فقط العلامات المهمة جداً
    significant_moves = data[data['Institutional_Move']]
    if not significant_moves.empty:
        fig.add_trace(go.Scatter(
            x=significant_moves.index, 
            y=significant_moves['High'],
            mode='markers', 
            marker=dict(color='yellow', size=6, symbol='diamond-open', line=dict(width=2)),
            name='Institutional Imbalance (Top 5%)'
        ))

    # 3. Key Structure Levels (Only the latest active ones)
    last_h = data['Pivot_High'].dropna().iloc[-1]
    last_l = data['Pivot_Low'].dropna().iloc[-1]
    
    fig.add_hline(y=last_h, line_dash="dash", line_color="rgba(255, 0, 0, 0.5)", annotation_text="Major Res", annotation_position="top right")
    fig.add_hline(y=last_l, line_dash="dash", line_color="rgba(0, 255, 0, 0.5)", annotation_text="Major Supp", annotation_position="bottom right")

    # Chart Layout
    fig.update_layout(
        title=f"Institutional Chart: {full_symbol}",
        template="plotly_dark",
        height=650,
        xaxis_rangeslider_visible=False,
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117"
    )

    # --- DISPLAY DASHBOARD ---
    
    # Metrics
    curr_price = data['Close'].iloc[-1]
    prev_price = data['Close'].iloc[-2]
    change = ((curr_price - prev_price)/prev_price) * 100
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Current Price", f"{curr_price:.2f}", f"{change:.2f}%")
    c2.metric("Market Phase", "Accumulation?" if curr_price < last_h and curr_price > last_l else "Trending")
    c3.metric("Detected Inst. Moves", len(significant_moves))

    # Plot
    st.plotly_chart(fig, use_container_width=True)

    # Text Analysis
    st.info(f"""
    **تحليل السيولة (Liquidity Logic):**
    تم تنظيف الشارت لعرض مناطق الزخم الحقيقي فقط. العلامات الصفراء (الماسات) تشير الآن إلى **أقوى 5% من التحركات** خلال الفترة المحددة، وهي المناطق التي غالباً ما يترك فيها "صناع السوق" فجوات سعرية (FVG) يعود السعر لاختبارها لاحقاً.
    """)

else:
    st.error("Error: Symbol not found or API issue. Try a different ticker.")
