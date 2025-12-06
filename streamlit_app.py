import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time

# --- 1. إعداد الصفحة والوضع الليلي ---
st.set_page_config(page_title="Saudi Pro Dashboard V3", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    /* تحسين الجداول */
    .stDataFrame { border: 1px solid #30333d; }
    div[data-testid="stDataFrame"] div[class*="css"] { background-color: #161b24; color: white; }
    /* تحسين البطاقات */
    div[data-testid="stMetric"] { background-color: #1d212b !important; border: 1px solid #30333d; padding: 10px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    div[data-testid="stMetricLabel"] { color: #90caf9 !important; font-size: 0.9rem; }
    div[data-testid="stMetricValue"] { color: #ffffff !important; font-size: 1.2rem; }
    /* زر التشغيل */
    div.stButton > button { background: linear-gradient(90deg, #2962ff, #2979ff); color: white; border: none; width: 100%; font-weight: bold; padding: 10px; border-radius: 8px; }
    div.stButton > button:hover { background: linear-gradient(90deg, #1565c0, #1e88e5); }
    /* التبويبات */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: #1d212b; color: #e0e0e0; border-radius: 4px; border: 1px solid #333; }
    .stTabs [aria-selected="true"] { background-color: #2962ff !important; color: white !important; border-color: #2962ff; }
</style>
""", unsafe_allow_html=True)

# --- 2. الإعدادات ---
with st.sidebar:
    st.header("⚙️ مركز التحكم")
    RSI_PERIOD = st.number_input("RSI Period", value=24)
    EMA_PERIOD = st.number_input("EMA Period", value=8)
    st.divider()
    st.subheader("📦 الصناديق")
    ATR_MULT = st.number_input("ATR Multiplier", value=1.5)
    BOX_LOOKBACK = st.slider("Box History", 10, 50, 20)
    st.info("اضغط الزر الأزرق الكبير في الصفحة للبدء.")

# --- 3. قاعدة البيانات (مع تصنيف القطاعات للهيت ماب) ---
# قمت بإضافة حقل "Sector" لكل سهم لإنشاء الخريطة الحرارية
STOCKS_DB = [
    {"symbol": "2222.SR", "name": "أرامكو", "sector": "الطاقة"},
    {"symbol": "2010.SR", "name": "سابك", "sector": "المواد الأساسية"},
    {"symbol": "1120.SR", "name": "الراجحي", "sector": "البنوك"},
    {"symbol": "1180.SR", "name": "الأهلي", "sector": "البنوك"},
    {"symbol": "7010.SR", "name": "STC", "sector": "الأتصالات"},
    {"symbol": "1211.SR", "name": "معادن", "sector": "المواد الأساسية"},
    {"symbol": "2020.SR", "name": "سابك للمغذيات", "sector": "المواد الأساسية"},
    {"symbol": "1150.SR", "name": "الإنماء", "sector": "البنوك"},
    {"symbol": "1010.SR", "name": "الرياض", "sector": "البنوك"},
    {"symbol": "4190.SR", "name": "جرير", "sector": "تجزئة"},
    {"symbol": "4002.SR", "name": "المواساة", "sector": "رعاية صحية"},
    {"symbol": "4013.SR", "name": "سليمان الحبيب", "sector": "رعاية صحية"},
    {"symbol": "2280.SR", "name": "المراعي", "sector": "أغذية"},
    {"symbol": "7202.SR", "name": "سلوشنز", "sector": "تقنية"},
    {"symbol": "7203.SR", "name": "علم", "sector": "تقنية"},
    {"symbol": "4200.SR", "name": "الدريس", "sector": "الطاقة"},
    {"symbol": "4030.SR", "name": "البحري", "sector": "الطاقة"},
    {"symbol": "5110.SR", "name": "الكهرباء", "sector": "مرافق"},
    {"symbol": "7020.SR", "name": "موبايلي", "sector": "الأتصالات"},
    {"symbol": "7030.SR", "name": "زين", "sector": "الأتصالات"},
    {"symbol": "4300.SR", "name": "دار الأركان", "sector": "عقارات"},
    {"symbol": "4250.SR", "name": "جبل عمر", "sector": "عقارات"},
    {"symbol": "8010.SR", "name": "التعاونية", "sector": "تأمين"},
    {"symbol": "8210.SR", "name": "بوبا", "sector": "تأمين"},
    {"symbol": "1810.SR", "name": "سيرا", "sector": "خدمات"},
    {"symbol": "1830.SR", "name": "وقت اللياقة", "sector": "خدمات"},
    {"symbol": "4261.SR", "name": "ذيب", "sector": "نقل"},
    {"symbol": "^TASI.SR", "name": "المؤشر العام", "sector": "مؤشر"}
]
# تحويل القائمة لقاموس لسهولة الوصول
TICKERS = {item['symbol']: item['name'] for item in STOCKS_DB}
SECTORS = {item['name']: item['sector'] for item in STOCKS_DB}

# --- 4. الدوال الفنية ---
def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    return ranges.max(axis=1).ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def check_bullish_box(df, atr_series):
    # (نفس دالة الصندوق السابقة)
    in_series = False; is_bullish = False; start_open = 0.0; end_close = 0.0; found_boxes = []
    prices = df.iloc[-100:].reset_index() if len(df) > 100 else df.reset_index()
    atrs = atr_series.iloc[-100:].values if len(df) > 100 else atr_series.values
    
    for i in range(len(prices)):
        row = prices.iloc[i]; close = row['Close']; open_p = row['Open']
        is_green = close > open_p; is_red = close < open_p
        current_atr = atrs[i]
        if np.isnan(current_atr): continue
        
        if not in_series:
            if is_green: in_series = True; is_bullish = True; start_open = open_p
            elif is_red: in_series = True; is_bullish = False; start_open = open_p
        elif in_series:
            if is_bullish and is_green: end_close = close
            elif not is_bullish and is_red: end_close = close
            elif (is_bullish and is_red) or (not is_bullish and is_green):
                final_close = end_close if end_close != 0 else start_open
                price_move = abs(final_close - start_open)
                if price_move >= current_atr * ATR_MULT and is_bullish:
                    days_ago = len(prices) - i
                    if days_ago <= BOX_LOOKBACK:
                        found_boxes.append({"Box_Top": max(start_open, final_close), "Box_Bottom": min(start_open, final_close), "Days_Ago": days_ago})
                in_series = True; is_bullish = is_green; start_open = open_p; end_close = close
    return found_boxes

def process_data(df):
    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/24, min_periods=24, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/24, min_periods=24, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # EMA & MACD
    df['EMA'] = df['Close'].ewm(span=8, adjust=False).mean()
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Change & Volatility
    df['Change'] = df['Close'].pct_change() * 100
    df['ATR'] = calculate_atr(df)
    
    # Volume Analysis (RVOL) - كاشف السيولة
    # RVOL = حجم اليوم / متوسط حجم 20 يوم
    df['Vol_Avg'] = df['Volume'].rolling(window=20).mean()
    df['RVOL'] = df['Volume'] / df['Vol_Avg']
    
    return df

# --- 5. الواجهة الرئيسية ---
st.title("💎 Saudi Market Pro (V3)")

if 'data' not in st.session_state: st.session_state['data'] = []
if 'signals' not in st.session_state: st.session_state['signals'] = []
if 'boxes' not in st.session_state: st.session_state['boxes'] = [] 
if 'history' not in st.session_state: st.session_state['history'] = {}

if st.button("🚀 تحديث وتحليل السوق (Live Scan)"):
    st.session_state['data'] = []
    st.session_state['signals'] = []
    st.session_state['boxes'] = []
    st.session_state['history'] = {}
    
    prog_bar = st.progress(0)
    status = st.empty()
    tickers_list = list(TICKERS.keys())
    
    # تقسيم التحميل
    chunk_size = 20 # تقليل الحجم لضمان السرعة مع البيانات الإضافية
    for i in range(0, len(tickers_list), chunk_size):
        chunk = tickers_list[i:i + chunk_size]
        status.text(f"جاري تحليل القطاع {i//chunk_size + 1}...")
        try:
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
                            if len(df) > 60:
                                df = process_data(df)
                                last = df.iloc[-1]
                                
                                # إعداد الرابط
                                link = f"https://www.tradingview.com/chart/?symbol=TADAWUL:{sym.replace('.SR','')}"
                                
                                st.session_state['history'][name] = df
                                st.session_state['data'].append({
                                    "Name": name, "Symbol": sym, "Sector": SECTORS.get(name, "عام"),
                                    "Price": last['Close'], "Change": last['Change'], 
                                    "RSI": last['RSI'], "MACD": last['MACD'], 
                                    "RVOL": last['RVOL'], # السيولة النسبية
                                    "Volume": last['Volume'],
                                    "TV": link
                                })
                                
                                # الصناديق
                                boxes = check_bullish_box(df, df['ATR'])
                                if boxes:
                                    latest = boxes[-1]
                                    mp = (latest['Box_Top'] + latest['Box_Bottom'])/2
                                    in_box = latest['Box_Bottom'] <= last['Close'] <= latest['Box_Top']
                                    if in_box:
                                        st.session_state['boxes'].append({
                                            "الاسم": name, "السعر": last['Close'], "المنتصف": mp,
                                            "الحالة": "🟢 فوق" if last['Close'] >= mp else "🔴 تحت",
                                            "TV": link
                                        })
                                
                                # القناص
                                t = df.tail(4)
                                if len(t) == 4:
                                    rsi_x = False; ema_x = False
                                    for x in range(1, 4):
                                        if t['RSI'].iloc[x-1] <= 30 and t['RSI'].iloc[x] > 30: rsi_x = True
                                        if t['Close'].iloc[x-1] <= t['EMA'].iloc[x-1] and t['Close'].iloc[x] > t['EMA'].iloc[x]: ema_x = True
                                    if rsi_x and ema_x:
                                        st.session_state['signals'].append({
                                            "الاسم": name, "السعر": last['Close'], "RSI": last['RSI'], 
                                            "السيولة": "🔥 عالية" if last['RVOL'] > 1.5 else "عادية",
                                            "TV": link
                                        })
                    except: continue
        except: pass
        prog_bar.progress(min((i + chunk_size) / len(tickers_list), 1.0))
    
    prog_bar.empty()
    status.success("تم التحديث!")

# --- 6. لوحة العرض (Dashboard) ---
if st.session_state['data']:
    df = pd.DataFrame(st.session_state['data'])
    
    # 1. بطاقات الأداء (Top Movers)
    st.markdown("##### 📊 نظرة سريعة")
    col1, col2, col3, col4 = st.columns(4)
    
    # الأكثر ارتفاعاً
    top_gainer = df.loc[df['Change'].idxmax()]
    col1.metric("🚀 الأكثر ارتفاعاً", f"{top_gainer['Name']}", f"{top_gainer['Change']:.2f}%")
    
    # الأكثر انخفاضاً
    top_loser = df.loc[df['Change'].idxmin()]
    col2.metric("🩸 الأكثر انخفاضاً", f"{top_loser['Name']}", f"{top_loser['Change']:.2f}%")
    
    # الأعلى سيولة (RVOL)
    top_vol = df.loc[df['RVOL'].idxmax()]
    col3.metric("🔥 انفجار سيولة", f"{top_vol['Name']}", f"x{top_vol['RVOL']:.1f}")
    
    # عدد الفرص
    col4.metric("🎯 فرص القناص", len(st.session_state['signals']))
    
    st.divider()
    
    # التبويبات
    tabs = st.tabs(["🗺️ خريطة السوق", "🎯 القناص", "📦 الصناديق", "📋 القائمة الكاملة", "📈 الشارت"])
    
    # رابط TradingView
    link_col = st.column_config.LinkColumn("شارت", display_text="Open TV")

    # Tab 1: Heatmap (الميزة الجديدة!)
    with tabs[0]:
        st.subheader("خريطة السيولة والقطاعات")
        # نستخدم Plotly Treemap
        fig = px.treemap(
            df, 
            path=[px.Constant("السوق السعودي"), 'Sector', 'Name'], 
            values='Price', # حجم المربع (يمكن تغييره لـ Market Cap لو توفرت)
            color='Change',
            color_continuous_scale=['red', '#1e222d', 'green'],
            color_continuous_midpoint=0,
            hover_data=['Price', 'RSI', 'RVOL']
        )
        fig.update_layout(margin=dict(t=0, l=0, r=0, b=0), height=500)
        st.plotly_chart(fig, use_container_width=True)

    # Tab 2: Sniper
    with tabs[1]:
        if st.session_state['signals']:
            st.dataframe(pd.DataFrame(st.session_state['signals']), column_config={"TV": link_col}, use_container_width=True)
        else: st.info("لا توجد إشارات.")

    # Tab 3: Boxes
    with tabs[2]:
        if st.session_state['boxes']:
            st.dataframe(pd.DataFrame(st.session_state['boxes']), column_config={"TV": link_col}, use_container_width=True)
        else: st.info("لا توجد صناديق.")

    # Tab 4: Full List (مع عمود السيولة الجديد)
    with tabs[3]:
        # تنسيق الجدول
        display_df = df.copy()
        # إضافة عمود السيولة النسبية كنص
        display_df['RVOL_Txt'] = display_df['RVOL'].apply(lambda x: f"x{x:.1f}" if x < 2 else f"🔥 x{x:.1f}")
        
        cols = ["Name", "Price", "Change", "RSI", "RVOL_Txt", "TV"]
        st.dataframe(
            display_df[cols].style.format({"Price": "{:.2f}", "Change": "{:.2f}%", "RSI": "{:.1f}"})
            .background_gradient(cmap='RdYlGn', subset=['Change']),
            column_config={"TV": link_col, "RVOL_Txt": "Relative Vol"},
            use_container_width=True, height=600
        )

    # Tab 5: Chart
    with tabs[4]:
        sel = st.selectbox("سهم:", df['Name'].unique())
        if sel:
            hist = st.session_state['history'][sel]
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            # Price
            fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name='Price'), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['EMA'], line=dict(color='orange'), name='EMA'), row=1, col=1)
            # Volume bar color based on close
            colors = ['green' if c >= o else 'red' for c, o in zip(hist['Close'], hist['Open'])]
            fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], marker_color=colors, opacity=0.5, name='Vol'), row=2, col=1)
            
            fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False, paper_bgcolor='#161b24', plot_bgcolor='#161b24')
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👋 مرحباً بك في الإصدار الثالث (V3)! اضغط زر التشغيل للبدء.")
