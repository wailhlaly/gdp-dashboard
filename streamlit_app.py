import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from streamlit_lightweight_charts import renderLightweightCharts

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Mudarib Pro V5")

# تنسيق الخلفية والألوان (Dark Mode الاحترافي)
st.markdown("""
<style>
    .stApp { background-color: #131722; }
    h1, h2, h3, p, div { color: #d1d4dc; }
    .stTextInput > div > div > input { color: #d1d4dc; background-color: #2a2e39; }
    .stSelectbox > div > div > div { color: #d1d4dc; background-color: #2a2e39; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Mudarib Pro | TradingView Engine")

# --- 2. SIDEBAR INPUTS (FIXED) ---
# أعدنا هذه القائمة لضمان عدم حدوث خطأ "Data not found"
with st.sidebar:
    st.header("إعدادات السهم")
    
    col_sym, col_mkt = st.columns([2, 1])
    with col_sym:
        symbol_val = st.text_input("رمز السهم", "2222") # أرامكو كمثال
    with col_mkt:
        market_suffix = st.selectbox("السوق", [".SR", "", ".L", ".HK"], index=0)
    
    # دمج الرمز مع الامتداد
    full_ticker = f"{symbol_val}{market_suffix}" if market_suffix else symbol_val
    
    period = st.selectbox("المدة الزمنية", ["1y", "2y", "5y"], index=0)
    timeframe = st.selectbox("الفاصي", ["1d", "1wk"], index=0)

# --- 3. DATA ENGINE (ROBUST) ---
@st.cache_data(ttl=3600) # تخزين مؤقت لساعة لتسريع الأداء
def fetch_data(ticker, p, i):
    try:
        # تحميل البيانات
        df = yf.download(ticker, period=p, interval=i, progress=False)
        
        # معالجة مشكلة yfinance الجديدة (MultiIndex)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if df.empty:
            return None

        # تجهيز البيانات
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns] # توحيد أسماء الأعمدة للصغيرة
        
        # خوارزمية ICT (Smart Imbalance)
        # نحدد فقط الشموع التي جسمها أكبر من 95% من الشموع الأخرى
        df['body'] = abs(df['close'] - df['open'])
        threshold = df['body'].quantile(0.95)
        df['imbalance'] = df['body'] > threshold
        
        return df
    except Exception as e:
        return None

# جلب البيانات
df = fetch_data(full_ticker, period, timeframe)

if df is not None:
    # --- 4. DATA PREPARATION FOR CHARTS ---
    # تحويل البيانات لصيغة تقبلها مكتبة Lightweight Charts
    
    # أ) بيانات الشموع
    candles_data = []
    for _, row in df.iterrows():
        candles_data.append({
            "time": row['date'].strftime('%Y-%m-%d'),
            "open": row['open'],
            "high": row['high'],
            "low": row['low'],
            "close": row['close']
        })

    # ب) بيانات العلامات (Imbalance Markers)
    markers_data = []
    imbalance_rows = df[df['imbalance']]
    
    for _, row in imbalance_rows.iterrows():
        markers_data.append({
            "time": row['date'].strftime('%Y-%m-%d'),
            "position": "aboveBar", # موقع العلامة
            "color": "#FFD700",     # لون ذهبي
            "shape": "arrowDown",
            "text": "ICT Gap",
            "size": 1               # حجم العلامة (صغير وأنيق)
        })

    # --- 5. CHART RENDER ---
    
    # إعدادات الشارت (نفس ستايل TradingView)
    chart_options = {
        "layout": {
            "backgroundColor": "#131722",
            "textColor": "#d1d4dc"
        },
        "grid": {
            "vertLines": {"color": "rgba(42, 46, 57, 0.2)", "style": 1},
            "horzLines": {"color": "rgba(42, 46, 57, 0.2)", "style": 1}
        },
        "rightPriceScale": {
            "borderColor": "rgba(197, 203, 206, 0.8)",
            "visible": True
        },
        "timeScale": {
            "borderColor": "rgba(197, 203, 206, 0.8)",
            "visible": True
        }
    }

    series_candlestick = [
        {
            "type": 'Candlestick',
            "data": candles_data,
            "options": {
                "upColor": '#089981',     # أخضر حديث
                "downColor": '#f23645',   # أحمر حديث
                "borderVisible": False,
                "wickUpColor": '#089981',
                "wickDownColor": '#f23645'
            },
            "markers": markers_data 
        }
    ]

    # عرض معلومات السعر الحالي
    curr_price = df['close'].iloc[-1]
    prev_price = df['close'].iloc[-2]
    change = ((curr_price - prev_price) / prev_price) * 100
    color_change = "green" if change > 0 else "red"
    
    st.markdown(f"### {full_ticker} : {curr_price:.2f} <span style='color:{color_change}'>({change:+.2f}%)</span>", unsafe_allow_html=True)

    # رسم الشارت النهائي
    renderLightweightCharts(
        options=chart_options,
        series=series_candlestick,
        height=550 # ارتفاع مناسب للجوال
    )
    
    st.caption("✅ تم التحليل باستخدام محرك: Lightweight Charts (TradingView)")

else:
    st.error(f"عذراً، لم يتم العثور على بيانات للسهم: {full_ticker}")
    st.info("تأكد من كتابة الرمز الصحيح واختيار السوق المناسب من القائمة الجانبية.")
