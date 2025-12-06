import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time

# --- استيراد القائمة ---
try:
    from data.saudi_tickers import STOCKS_DB
except ImportError:
    try:
        from saudi_tickers import STOCKS_DB
    except ImportError:
        st.error("🚨 ملف saudi_tickers.py مفقود.")
        st.stop()

TICKERS = {item['symbol']: item['name'] for item in STOCKS_DB}
SECTORS = {item['name']: item['sector'] for item in STOCKS_DB}

# --- 1. إعداد الصفحة والستايل (CSS المصحح) ---
st.set_page_config(page_title="Saudi Pro Ultimate", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    /* خلفية داكنة للتطبيق بالكامل */
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    
    /* إصلاح ألوان الجداول */
    .stDataFrame { border: 1px solid #30333d; }
    div[data-testid="stDataFrame"] div[class*="css"] { background-color: #161b24; color: white; }
    
    /* إصلاح ألوان البطاقات (Metrics) لتظهر بخلفية غامقة وكتابة بيضاء */
    div[data-testid="stMetric"] {
        background-color: #262730 !important;
        border: 1px solid #464b5f !important;
        padding: 15px !important;
        border-radius: 8px !important;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    [data-testid="stMetricLabel"] {
        color: #a3a8b8 !important;
    }
    
    /* تحسين الأزرار والتبويبات */
    div.stButton > button { background: linear-gradient(45deg, #2962ff, #0d47a1) !important; color: white !important; border: none; width: 100%; padding: 12px; border-radius: 8px; font-weight: bold; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: #1d212b; color: #e0e0e0; border-radius: 4px; }
    .stTabs [aria-selected="true"] { background-color: #2962ff !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# --- 2. الإعدادات ---
with st.sidebar:
    st.header("⚙️ الإعدادات")
    RSI_PERIOD = st.number_input("فترة RSI", value=24)
    EMA_PERIOD = st.number_input("فترة EMA", value=20)
    st.divider()
    ATR_MULT = st.number_input("مضاعف ATR", value=1.5)
    BOX_LOOKBACK = st.slider("عمر الصندوق (شمعة)", 5, 60, 25)

# --- 3. الدوال الفنية ---
def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    return ranges.max(axis=1).ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def process_data(df):
    # Basic Indicators
    df['Change'] = df['Close'].pct_change() * 100
    df['ATR'] = calculate_atr(df)
    
    # RVOL
    df['Vol_Avg'] = df['Volume'].rolling(window=20).mean()
    df['RVOL'] = df['Volume'] / df['Vol_Avg']
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/24, min_periods=24, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/24, min_periods=24, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Trend
    df['EMA'] = df['Close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA40'] = df['Close'].ewm(span=40, adjust=False).mean()
    df['EMA86'] = df['Close'].ewm(span=86, adjust=False).mean()
    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Score
    df['Trend_Score'] = (
        (df['Close'] > df['EMA']).astype(int) + 
        (df['Close'] > df['EMA20']).astype(int) + 
        (df['Close'] > df['EMA40']).astype(int) + 
        (df['Close'] > df['EMA86']).astype(int)
    )
    return df

def check_bullish_box(df):
    in_series = False; is_bullish = False; start_open = 0.0; end_close = 0.0; start_idx = 0; found_boxes = []
    prices = df.iloc[-100:].reset_index() if len(df) > 100 else df.reset_index()
    atrs = df['ATR'].iloc[-100:].values if len(df) > 100 else df['ATR'].values
    rvols = df['RVOL'].iloc[-100:].values if len(df) > 100 else df['RVOL'].values
    
    for i in range(len(prices)):
        row = prices.iloc[i]; close = row['Close']; open_p = row['Open']
        is_green = close > open_p; is_red = close < open_p
        current_atr = atrs[i]
        if np.isnan(current_atr): continue
        
        if not in_series:
            if is_green: in_series = True; is_bullish = True; start_open = open_p; start_idx = i
            elif is_red: in_series = True; is_bullish = False; start_open = open_p; start_idx = i
        elif in_series:
            if is_bullish and is_green: end_close = close
            elif not is_bullish and is_red: end_close = close
            elif (is_bullish and is_red) or (not is_bullish and is_green):
                final_close = end_close if end_close != 0 else start_open
                price_move = abs(final_close - start_open)
                if price_move >= current_atr * ATR_MULT and is_bullish:
                    days_ago = len(prices) - i
                    if days_ago <= BOX_LOOKBACK:
                        box_rvols = rvols[start_idx:i]
                        avg_rvol = np.mean(box_rvols) if len(box_rvols) > 0 else 1.0
                        found_boxes.append({
                            "Box_Top": max(start_open, final_close),
                            "Box_Bottom": min(start_open, final_close),
                            "Days_Ago": days_ago,
                            "Avg_RVOL": avg_rvol
                        })
                in_series = True; is_bullish = is_green; start_open = open_p; end_close = close; start_idx = i
    return found_boxes

def calculate_ai_score(last, box):
    score = 0; reasons = []
    if box['Avg_RVOL'] >= 1.5: score += 30; reasons.append("سيولة عالية")
    elif box['Avg_RVOL'] >= 1.0: score += 15
    mid = (box['Box_Top'] + box['Box_Bottom']) / 2
    if last['Close'] > mid: score += 20; reasons.append("فوق المنتصف")
    if last['Close'] > last['EMA']: score += 20; reasons.append("فوق EMA")
    if last['RSI'] > 50: score += 15
    if last['MACD'] > last['Signal']: score += 15; reasons.append("MACD إيجابي")
    return min(score, 100), reasons

# --- 4. المنطق والتشغيل ---
st.title("📊 محلل السوق السعودي (النسخة الثابتة)")

# تهيئة المتغيرات
for k in ['data', 'signals', 'boxes', 'history']:
    if k not in st.session_state: st.session_state[k] = [] if k != 'history' else {}

if st.button("🚀 تحديث البيانات"):
    st.session_state['data'] = []
    st.session_state['signals'] = []
    st.session_state['boxes'] = []
    st.session_state['history'] = {}
    
    prog = st.progress(0)
    status = st.empty()
    tickers_list = list(TICKERS.keys())
    
    chunk_size = 25
    for i in range(0, len(tickers_list), chunk_size):
        chunk = tickers_list[i:i + chunk_size]
        status.text(f"جاري التحليل... الدفعة {i//chunk_size + 1}")
        
        try:
            # هنا قمت بتأمين الكود بـ try/except بشكل صحيح لمنع SyntaxError
            raw = yf.download(chunk, period="1y", interval="1d", group_by='ticker', auto_adjust=False, threads=True, progress=False)
            
            if not raw.empty:
                for sym in chunk:
                    try:
                        name = TICKERS[sym]
                        try: df = raw[sym].copy()
                        except: continue
                        
                        col = 'Close' if 'Close' in df.columns else 'Adj Close'
                        if col in df.columns:
                            df = df.rename(columns={col: 'Close'})
                            df = df.dropna()
                            if len(df) > 90:
                                df = process_data(df)
                                last = df.iloc[-1]
                                link = f"https://www.tradingview.com/chart/?symbol=TADAWUL:{sym.replace('.SR','')}"
                                
                                st.session_state['history'][name] = df
                                # استخدام مفاتيح إنجليزية ثابتة لتجنب KeyError
                                st.session_state['data'].append({
                                    "Name": name, "Symbol": sym, "Sector": SECTORS.get(name, "عام"),
                                    "Price": last['Close'], "Change": last['Change'], 
                                    "RSI": last['RSI'], "MACD": last['MACD'], "Signal": last['Signal'],
                                    "RVOL": last['RVOL'], "Trend_Score": last['Trend_Score'], "TV": link
                                })
                                
                                # Boxes Logic
                                boxes = check_bullish_box(df)
                                if boxes:
                                    latest = boxes[-1]
                                    if last['Close'] >= latest['Box_Bottom']:
                                        score, reasons = calculate_ai_score(last, latest)
                                        st.session_state['boxes'].append({
                                            "Name": name, "Price": last['Close'], "AI_Score": score,
                                            "Reasons": ", ".join(reasons), "Box_Liq": f"x{latest['Avg_RVOL']:.1f}",
                                            "Days": latest['Days_Ago'], "TV": link,
                                            "Box_Top": latest['Box_Top'], "Box_Bottom": latest['Box_Bottom']
                                        })
                                        
                                # Sniper Logic
                                t = df.tail(4)
                                if len(t) == 4:
                                    rsi_x = False; ema_x = False
                                    for x in range(1, 4):
                                        if t['RSI'].iloc[x-1] <= 30 and t['RSI'].iloc[x] > 30: rsi_x = True
                                        if t['Close'].iloc[x-1] <= t['EMA'].iloc[x-1] and t['Close'].iloc[x] > t['EMA'].iloc[x]: ema_x = True
                                    if rsi_x and ema_x:
                                        st.session_state['signals'].append({
                                            "Name": name, "Price": last['Close'], "RSI": last['RSI'], "TV": link
                                        })
                    except: continue
        except Exception as e:
            print(f"Error: {e}") # طباعة الخطأ في الكونسول بدلاً من توقف البرنامج
            
        prog.progress(min((i + chunk_size) / len(tickers_list), 1.0))
        
    prog.empty()
    status.success("تم التحديث بنجاح!")

# --- 5. العرض ---
if st.session_state['data']:
    df = pd.DataFrame(st.session_state['data'])
    
    # البطاقات الإحصائية
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("عدد الشركات", len(df))
    # حل مشكلة KeyError: نستخدم 'Change' كما خزنّاها
    bullish_count = len(df[df['Change'] > 0])
    c2.metric("السوق أخضر", bullish_count)
    c3.metric("صناديق ذكية", len(st.session_state['boxes']))
    c4.metric("فرص قناص", len(st.session_state['signals']))
    
    st.divider()
    
    tabs = st.tabs(["📦 الصناديق الذكية", "🎯 القناص", "🗺️ الخريطة", "📋 السوق", "📈 الشارت"])
    link_col = st.column_config.LinkColumn("شارت", display_text="Open TV")
    
    with tabs[0]:
        if st.session_state['boxes']:
            box_df = pd.DataFrame(st.session_state['boxes']).sort_values('AI_Score', ascending=False)
            st.dataframe(
                box_df.style.format({"Price": "{:.2f}", "AI_Score": "{:.0f}"})
                .background_gradient(cmap='Greens', subset=['AI_Score']),
                column_config={"TV": link_col, "Name": "الاسم", "Price": "السعر", "AI_Score": "التقييم", "Reasons": "الأسباب", "Box_Liq": "السيولة", "Days": "منذ (يوم)"},
                use_container_width=True
            )
        else: st.info("لا توجد صناديق.")

    with tabs[1]:
        if st.session_state['signals']:
            sig_df = pd.DataFrame(st.session_state['signals'])
            st.dataframe(sig_df, column_config={"TV": link_col, "Name": "الاسم", "Price": "السعر"}, use_container_width=True)
        else: st.info("لا توجد إشارات.")

    with tabs[2]:
        fig = px.treemap(df, path=[px.Constant("السوق"), 'Sector', 'Name'], values='Price', color='Trend_Score', color_continuous_scale='RdYlGn', range_color=[0, 4])
        fig.update_layout(margin=dict(t=0, l=0, r=0, b=0), height=500)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        # عرض الجدول مع تعريب العناوين فقط هنا
        disp = df.copy().rename(columns={"Name": "الاسم", "Price": "السعر", "Change": "التغير %", "RSI": f"RSI ({RSI_PERIOD})"})
        cols = ["الاسم", "السعر", "التغير %", f"RSI ({RSI_PERIOD})", "MACD", "TV"]
        st.dataframe(
            disp[cols].style.format({"السعر": "{:.2f}", "التغير %": "{:.2f}%", f"RSI ({RSI_PERIOD})": "{:.2f}"})
            .background_gradient(cmap='RdYlGn', subset=['التغير %']),
            column_config={"TV": link_col}, use_container_width=True, height=600
        )

    with tabs[4]:
        sel = st.selectbox("سهم:", df['Name'].unique())
        if sel:
            hist = st.session_state['history'][sel]
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name='Price'), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['EMA'], line=dict(color='orange'), name='EMA'), row=1, col=1)
            
            box_res = check_bullish_box(hist)
            if box_res:
                latest = box_res[-1]
                fig.add_shape(type="rect", x0=hist.index[-latest['Days_Ago']-2], x1=hist.index[-1], y0=latest['Box_Bottom'], y1=latest['Box_Top'], line=dict(color="green", width=2), fillcolor="rgba(0,255,0,0.1)", row=1, col=1)
            
            fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False, paper_bgcolor='#161b24', plot_bgcolor='#161b24')
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("👋 جاهز! اضغط الزر الأزرق للتحديث.")
