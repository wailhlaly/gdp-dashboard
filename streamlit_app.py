import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time

# --- 1. إعداد الصفحة والوضع الليلي ---
st.set_page_config(page_title="Saudi Pro V4", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    .stDataFrame { border: 1px solid #30333d; }
    div[data-testid="stDataFrame"] div[class*="css"] { background-color: #161b24; color: white; }
    div[data-testid="stMetric"] { background-color: #1d212b !important; border: 1px solid #30333d; padding: 10px; border-radius: 8px; }
    div[data-testid="stMetricValue"] { color: #ffffff !important; }
    div.stButton > button { background: linear-gradient(90deg, #2962ff, #2979ff); color: white; border: none; width: 100%; font-weight: bold; padding: 10px; border-radius: 8px; }
    div.stButton > button:hover { background: linear-gradient(90deg, #1565c0, #1e88e5); }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: #1d212b; color: #e0e0e0; border-radius: 4px; border: 1px solid #333; }
    .stTabs [aria-selected="true"] { background-color: #2962ff !important; color: white !important; }
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

# --- 3. قاعدة البيانات الشاملة (مع القطاعات) ---
# تم دمج القائمة الكاملة مع تصنيف القطاعات لتعمل الخرائط بشكل صحيح
STOCKS_DB = [
    # الطاقة
    {"symbol": "2222.SR", "name": "أرامكو", "sector": "الطاقة"}, {"symbol": "2030.SR", "name": "المصافي", "sector": "الطاقة"},
    {"symbol": "4200.SR", "name": "الدريس", "sector": "الطاقة"}, {"symbol": "4030.SR", "name": "البحري", "sector": "الطاقة"},
    {"symbol": "2381.SR", "name": "الحفر العربية", "sector": "الطاقة"}, {"symbol": "2382.SR", "name": "أديس", "sector": "الطاقة"},
    # المواد الأساسية
    {"symbol": "2010.SR", "name": "سابك", "sector": "المواد الأساسية"}, {"symbol": "1211.SR", "name": "معادن", "sector": "المواد الأساسية"},
    {"symbol": "2020.SR", "name": "سابك للمغذيات", "sector": "المواد الأساسية"}, {"symbol": "2310.SR", "name": "سبكيم", "sector": "المواد الأساسية"},
    {"symbol": "2290.SR", "name": "ينساب", "sector": "المواد الأساسية"}, {"symbol": "2060.SR", "name": "التصنيع", "sector": "المواد الأساسية"},
    {"symbol": "2350.SR", "name": "كيان", "sector": "المواد الأساسية"}, {"symbol": "2330.SR", "name": "المتقدمة", "sector": "المواد الأساسية"},
    # البنوك
    {"symbol": "1120.SR", "name": "الراجحي", "sector": "البنوك"}, {"symbol": "1180.SR", "name": "الأهلي", "sector": "البنوك"},
    {"symbol": "1010.SR", "name": "الرياض", "sector": "البنوك"}, {"symbol": "1150.SR", "name": "الإنماء", "sector": "البنوك"},
    {"symbol": "1060.SR", "name": "الأول", "sector": "البنوك"}, {"symbol": "1140.SR", "name": "البلاد", "sector": "البنوك"},
    {"symbol": "1020.SR", "name": "الجزيرة", "sector": "البنوك"}, {"symbol": "1050.SR", "name": "الفرنسي", "sector": "البنوك"},
    # الاتصالات
    {"symbol": "7010.SR", "name": "STC", "sector": "الأتصالات"}, {"symbol": "7020.SR", "name": "موبايلي", "sector": "الأتصالات"},
    {"symbol": "7030.SR", "name": "زين", "sector": "الأتصالات"}, {"symbol": "7040.SR", "name": "عذيب", "sector": "الأتصالات"},
    # التقنية
    {"symbol": "7202.SR", "name": "سلوشنز", "sector": "التقنية"}, {"symbol": "7203.SR", "name": "علم", "sector": "التقنية"},
    {"symbol": "7200.SR", "name": "المعمر", "sector": "التقنية"}, {"symbol": "7201.SR", "name": "بحر العرب", "sector": "التقنية"},
    # المرافق
    {"symbol": "5110.SR", "name": "الكهرباء", "sector": "المرافق"}, {"symbol": "2080.SR", "name": "الغاز", "sector": "المرافق"},
    {"symbol": "1834.SR", "name": "مرافق", "sector": "المرافق"},
    # التجزئة والأغذية
    {"symbol": "4190.SR", "name": "جرير", "sector": "تجزئة"}, {"symbol": "4001.SR", "name": "العثيم", "sector": "تجزئة"},
    {"symbol": "4003.SR", "name": "إكسترا", "sector": "تجزئة"}, {"symbol": "4164.SR", "name": "النهدي", "sector": "تجزئة"},
    {"symbol": "2280.SR", "name": "المراعي", "sector": "أغذية"}, {"symbol": "2270.SR", "name": "سدافكو", "sector": "أغذية"},
    {"symbol": "6010.SR", "name": "نادك", "sector": "أغذية"}, {"symbol": "1810.SR", "name": "سيرا", "sector": "خدمات"},
    {"symbol": "1830.SR", "name": "وقت اللياقة", "sector": "خدمات"}, {"symbol": "4261.SR", "name": "ذيب", "sector": "نقل"},
    # الصحة والتأمين
    {"symbol": "4002.SR", "name": "المواساة", "sector": "رعاية صحية"}, {"symbol": "4013.SR", "name": "سليمان الحبيب", "sector": "رعاية صحية"},
    {"symbol": "4004.SR", "name": "دلة", "sector": "رعاية صحية"}, {"symbol": "4007.SR", "name": "الحمادي", "sector": "رعاية صحية"},
    {"symbol": "8010.SR", "name": "التعاونية", "sector": "تأمين"}, {"symbol": "8210.SR", "name": "بوبا", "sector": "تأمين"},
    {"symbol": "8230.SR", "name": "الراجحي تكافل", "sector": "تأمين"},
    # العقار
    {"symbol": "4300.SR", "name": "دار الأركان", "sector": "عقارات"}, {"symbol": "4250.SR", "name": "جبل عمر", "sector": "عقارات"},
    {"symbol": "4220.SR", "name": "إعمار", "sector": "عقارات"}, {"symbol": "4090.SR", "name": "طيبة", "sector": "عقارات"},
    {"symbol": "4321.SR", "name": "سينومي سنترز", "sector": "عقارات"},
    # المؤشر
    {"symbol": "^TASI.SR", "name": "المؤشر العام", "sector": "مؤشر"}
]
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
    # المؤشرات الأساسية
    delta = df['Close'].diff()
    gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/24, min_periods=24, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/24, min_periods=24, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['EMA'] = df['Close'].ewm(span=8, adjust=False).mean()
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Change'] = df['Close'].pct_change() * 100
    df['ATR'] = calculate_atr(df)
    
    # RVOL (السيولة النسبية)
    df['Vol_Avg'] = df['Volume'].rolling(window=20).mean()
    df['RVOL'] = df['Volume'] / df['Vol_Avg']

    # --- الجديد: حساب متوسطات خريطة التوجه ---
    df['EMA8'] = df['Close'].ewm(span=8, adjust=False).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA40'] = df['Close'].ewm(span=40, adjust=False).mean()
    df['EMA86'] = df['Close'].ewm(span=86, adjust=False).mean()
    
    # حساب نقاط قوة الترند (Score 0-4)
    # كل متوسط يتجاوزه السعر يضيف نقطة
    score = (
        (df['Close'] > df['EMA8']).astype(int) + 
        (df['Close'] > df['EMA20']).astype(int) + 
        (df['Close'] > df['EMA40']).astype(int) + 
        (df['Close'] > df['EMA86']).astype(int)
    )
    df['Trend_Score'] = score
    
    return df

# --- 5. الواجهة الرئيسية ---
st.title("💎 Saudi Market Pro (V4)")

if 'data' not in st.session_state: st.session_state['data'] = []
if 'signals' not in st.session_state: st.session_state['signals'] = []
if 'boxes' not in st.session_state: st.session_state['boxes'] = [] 
if 'history' not in st.session_state: st.session_state['history'] = {}

if st.button("🚀 تحديث وتحليل السوق (Trend Map)"):
    st.session_state['data'] = []
    st.session_state['signals'] = []
    st.session_state['boxes'] = []
    st.session_state['history'] = {}
    
    prog_bar = st.progress(0)
    status = st.empty()
    tickers_list = list(TICKERS.keys())
    
    chunk_size = 20
    for i in range(0, len(tickers_list), chunk_size):
        chunk = tickers_list[i:i + chunk_size]
        status.text(f"جاري تحليل القطاع {i//chunk_size + 1}...")
        try:
            # نحتاج بيانات 100 يوم على الأقل لضمان حساب EMA86
            raw = yf.download(chunk, period="6mo", interval="1d", group_by='ticker', auto_adjust=False, threads=True, progress=False)
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
                            if len(df) > 90: # زدنا الشرط لضمان وجود EMA86
                                df = process_data(df)
                                last = df.iloc[-1]
                                link = f"https://www.tradingview.com/chart/?symbol=TADAWUL:{sym.replace('.SR','')}"
                                
                                st.session_state['history'][name] = df
                                st.session_state['data'].append({
                                    "Name": name, "Symbol": sym, "Sector": SECTORS.get(name, "عام"),
                                    "Price": last['Close'], "Change": last['Change'], 
                                    "RSI": last['RSI'], "MACD": last['MACD'], 
                                    "RVOL": last['RVOL'], "Volume": last['Volume'],
                                    "Trend_Score": last['Trend_Score'], # نقاط الترند الجديدة
                                    "TV": link
                                })
                                
                                # Boxes
                                boxes = check_bullish_box(df, df['ATR'])
                                if boxes:
                                    latest = boxes[-1]
                                    mp = (latest['Box_Top'] + latest['Box_Bottom'])/2
                                    if latest['Box_Bottom'] <= last['Close'] <= latest['Box_Top']:
                                        st.session_state['boxes'].append({
                                            "الاسم": name, "السعر": last['Close'], "المنتصف": mp,
                                            "الحالة": "🟢 فوق" if last['Close'] >= mp else "🔴 تحت",
                                            "TV": link
                                        })
                                
                                # Sniper
                                t = df.tail(4)
                                if len(t) == 4:
                                    rsi_x = False; ema_x = False
                                    for x in range(1, 4):
                                        if t['RSI'].iloc[x-1] <= 30 and t['RSI'].iloc[x] > 30: rsi_x = True
                                        if t['Close'].iloc[x-1] <= t['EMA'].iloc[x-1] and t['Close'].iloc[x] > t['EMA'].iloc[x]: ema_x = True
                                    if rsi_x and ema_x:
                                        st.session_state['signals'].append({
                                            "الاسم": name, "السعر": last['Close'], "RSI": last['RSI'], 
                                            "السيولة": "🔥 عالية" if last['RVOL'] > 1.5 else "عادية", "TV": link
                                        })
                    except: continue
        except: pass
        prog_bar.progress(min((i + chunk_size) / len(tickers_list), 1.0))
    
    prog_bar.empty()
    status.success("تم التحديث!")

# --- 6. لوحة العرض (Dashboard) ---
if st.session_state['data']:
    df = pd.DataFrame(st.session_state['data'])
    
    st.markdown("##### 📊 نظرة سريعة")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🚀 الأكثر ارتفاعاً", f"{df.loc[df['Change'].idxmax()]['Name']}", f"{df['Change'].max():.2f}%")
    c2.metric("🩸 الأكثر انخفاضاً", f"{df.loc[df['Change'].idxmin()]['Name']}", f"{df['Change'].min():.2f}%")
    c3.metric("🔥 انفجار سيولة", f"{df.loc[df['RVOL'].idxmax()]['Name']}", f"x{df['RVOL'].max():.1f}")
    c4.metric("📈 ترند صاعد قوي", len(df[df['Trend_Score'] == 4]))
    
    st.divider()
    
    # التبويبات (تمت إضافة الخريطة الجديدة)
    tabs = st.tabs(["🗺️ خريطة المتوسطات", "📦 الصناديق", "🎯 القناص", "📋 السوق الكامل", "📈 الشارت"])
    link_col = st.column_config.LinkColumn("شارت", display_text="Open TV")

    # --- TAB 1: EMA Heatmap (الميزة الجديدة) ---
    with tabs[0]:
        st.subheader("خريطة قوة الترند (EMA 8-20-40-86)")
        st.caption("اللون الأخضر الفاقع = السعر فوق جميع المتوسطات (4/4). اللون الأحمر = السعر تحت جميع المتوسطات (0/4).")
        
        # خريطة حرارية تعتمد على Trend_Score
        fig_ema = px.treemap(
            df, 
            path=[px.Constant("السوق السعودي"), 'Sector', 'Name'], 
            values='Price', # حجم المربع
            color='Trend_Score', # لون المربع حسب عدد المتوسطات التي اخترقها
            color_continuous_scale='RdYlGn', # من الأحمر (0) إلى الأخضر (4)
            range_color=[0, 4], # تثبيت النطاق
            hover_data=['Price', 'Change', 'Trend_Score']
        )
        fig_ema.update_layout(margin=dict(t=0, l=0, r=0, b=0), height=500)
        st.plotly_chart(fig_ema, use_container_width=True)

    # Tab 2: Boxes
    with tabs[1]:
        if st.session_state['boxes']:
            st.dataframe(pd.DataFrame(st.session_state['boxes']), column_config={"TV": link_col}, use_container_width=True)
        else: st.info("لا توجد صناديق.")

    # Tab 3: Sniper
    with tabs[2]:
        if st.session_state['signals']:
            st.dataframe(pd.DataFrame(st.session_state['signals']), column_config={"TV": link_col}, use_container_width=True)
        else: st.info("لا توجد إشارات.")

    # Tab 4: Full List
    with tabs[3]:
        display_df = df.copy()
        display_df['RVOL_Txt'] = display_df['RVOL'].apply(lambda x: f"x{x:.1f}" if x < 2 else f"🔥 x{x:.1f}")
        # إضافة عمود نقاط الترند للجدول أيضاً
        display_df['Trend'] = display_df['Trend_Score'].apply(lambda x: "🟢 قوي" if x==4 else ("🟡 متوسط" if x>=2 else "🔴 هابط"))
        
        cols = ["Name", "Price", "Change", "RSI", "Trend", "RVOL_Txt", "TV"]
        st.dataframe(
            display_df[cols].style.format({"Price": "{:.2f}", "Change": "{:.2f}%", "RSI": "{:.1f}"})
            .background_gradient(cmap='RdYlGn', subset=['Change']),
            column_config={"TV": link_col},
            use_container_width=True, height=600
        )

    # Tab 5: Chart
    with tabs[4]:
        sel = st.selectbox("سهم:", df['Name'].unique())
        if sel:
            hist = st.session_state['history'][sel]
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name='Price'), row=1, col=1)
            # رسم المتوسطات الجديدة على الشارت
            fig.add_trace(go.Scatter(x=hist.index, y=hist['EMA8'], line=dict(color='yellow', width=1), name='EMA 8'), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['EMA20'], line=dict(color='orange', width=1), name='EMA 20'), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['EMA40'], line=dict(color='red', width=1), name='EMA 40'), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['EMA86'], line=dict(color='blue', width=2), name='EMA 86'), row=1, col=1)
            
            fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False, paper_bgcolor='#161b24', plot_bgcolor='#161b24')
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👋 V4 جاهز! اضغط زر التحديث.")
