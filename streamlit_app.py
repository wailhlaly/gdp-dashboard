import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from streamlit_option_menu import option_menu
from scipy.signal import argrelextrema
import time

# --- استيراد البيانات ---
try:
    from data.saudi_tickers import STOCKS_DB
except ImportError:
    st.error("🚨 ملف البيانات مفقود.")
    st.stop()

TICKERS = {item['symbol']: item['name'] for item in STOCKS_DB}
SECTORS = {item['name']: item['sector'] for item in STOCKS_DB}

# --- 1. إعداد الصفحة والستايل ---
st.set_page_config(page_title="TASI Pro V9", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif; }
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    .stDataFrame { border: 1px solid #30333d; }
    div[data-testid="stDataFrame"] div[class*="css"] { background-color: #161b24; color: white; }
    div[data-testid="stMetric"] { background-color: #1d212b !important; border: 1px solid #464b5f !important; padding: 15px !important; border-radius: 10px !important; }
    [data-testid="stMetricValue"] { color: #ffffff !important; }
    div.stButton > button { background: linear-gradient(90deg, #ff6d00, #ff3d00); color: white; border: none; padding: 12px 20px; border-radius: 8px; font-weight: bold; width: 100%; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: #1d212b; color: #e0e0e0; border-radius: 4px; }
    .stTabs [aria-selected="true"] { background-color: #ff6d00 !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# --- 2. القائمة العلوية ---
selected_tab = option_menu(
    menu_title=None,
    options=["القائمة الشاملة (Master List)", "التحليل الفني", "الخريطة", "الشارت"],
    icons=["list-task", "cpu", "grid", "graph-up"],
    default_index=0,
    orientation="horizontal",
    styles={"container": {"background-color": "transparent"}, "nav-link-selected": {"background-color": "#ff6d00"}}
)

# --- 3. الإعدادات ---
with st.sidebar:
    st.header("⚙️ إعدادات التحليل")
    RSI_PERIOD = st.number_input("RSI Period", 14, 30, 24)
    EMA_PERIOD = st.number_input("EMA Trend", 10, 200, 20)
    st.divider()
    ATR_MULT = st.number_input("ATR Multiplier", 1.0, 3.0, 1.5)
    BOX_LOOKBACK = st.slider("Box History", 10, 100, 25)

# --- 4. الدوال الفنية ---
def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    return ranges.max(axis=1).ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def process_data(df):
    df['Change'] = df['Close'].pct_change() * 100
    df['ATR'] = calculate_atr(df)
    df['Vol_Avg'] = df['Volume'].rolling(window=20).mean()
    df['RVOL'] = df['Volume'] / df['Vol_Avg']
    
    delta = df['Close'].diff()
    gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/24, min_periods=24, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/24, min_periods=24, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['EMA'] = df['Close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['Trend_Score'] = ((df['Close'] > df['EMA']).astype(int) + (df['Close'] > df['EMA50']).astype(int))
    return df

# 🔥 كاشف الصناديق الصعودية (Bullish)
def check_bullish_box(df, atr_series):
    in_series = False; is_bullish = False; start_open = 0.0; end_close = 0.0; start_idx = 0; found_boxes = []
    prices = df.iloc[-150:].reset_index(); atrs = atr_series.iloc[-150:].values
    
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
                    periods_ago = len(prices) - i
                    if periods_ago <= BOX_LOOKBACK:
                        found_boxes.append({
                            "Type": "Bullish",
                            "Box_Top": max(start_open, final_close), "Box_Bottom": min(start_open, final_close),
                            "Days_Ago": periods_ago, "Start_Index": len(df) - periods_ago - (i - start_idx), "End_Index": len(df) - periods_ago
                        })
                in_series = True; is_bullish = is_green; start_open = open_p; end_close = close; start_idx = i
    return found_boxes

# ❄️ كاشف الصناديق الهبوطية (Bearish) - الجديد
def check_bearish_box(df, atr_series):
    in_series = False; is_bearish_series = False; start_open = 0.0; end_close = 0.0; start_idx = 0; found_boxes = []
    prices = df.iloc[-150:].reset_index(); atrs = atr_series.iloc[-150:].values
    
    for i in range(len(prices)):
        row = prices.iloc[i]; close = row['Close']; open_p = row['Open']
        is_red = close < open_p; is_green = close > open_p
        current_atr = atrs[i]
        if np.isnan(current_atr): continue
        
        if not in_series:
            if is_red: in_series = True; is_bearish_series = True; start_open = open_p; start_idx = i
            elif is_green: in_series = True; is_bearish_series = False; start_open = open_p; start_idx = i
        elif in_series:
            if is_bearish_series and is_red: end_close = close
            elif not is_bearish_series and is_green: end_close = close
            elif (is_bearish_series and is_green) or (not is_bearish_series and is_red):
                final_close = end_close if end_close != 0 else start_open
                price_move = abs(final_close - start_open)
                # شرط الهبوط: سلسلة حمراء قوية
                if price_move >= current_atr * ATR_MULT and is_bearish_series:
                    periods_ago = len(prices) - i
                    if periods_ago <= BOX_LOOKBACK:
                        found_boxes.append({
                            "Type": "Bearish",
                            "Box_Top": max(start_open, final_close), "Box_Bottom": min(start_open, final_close),
                            "Days_Ago": periods_ago, "Start_Index": len(df) - periods_ago - (i - start_idx), "End_Index": len(df) - periods_ago
                        })
                in_series = True; is_bearish_series = is_red; start_open = open_p; end_close = close; start_idx = i
    return found_boxes

# --- 5. المنطق (Engine) ---
if 'boxes_list' not in st.session_state: st.session_state['boxes_list'] = []
if 'history' not in st.session_state: st.session_state['history'] = {}

c1, c2 = st.columns([1, 4])
with c2:
    if st.button("🚀 تحديث القائمة الشاملة (يومي/أسبوعي/شهري)"):
        st.session_state['boxes_list'] = []
        st.session_state['history'] = {}
        progress = st.progress(0); status = st.empty()
        tickers = list(TICKERS.keys())
        
        # إعداد الفريمات
        TIMEFRAMES = {
            'Daily': {'p': '1y', 'i': '1d', 'lbl': 'يومي 📅'},
            'Weekly': {'p': '2y', 'i': '1wk', 'lbl': 'أسبوعي 🗓️'},
            'Monthly': {'p': '5y', 'i': '1mo', 'lbl': 'شهري 📆'}
        }
        
        total_steps = len(tickers) * 3
        curr_step = 0
        
        chunk_size = 20
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i:i + chunk_size]
            
            for tf_name, tf_cfg in TIMEFRAMES.items():
                status.markdown(f"**جاري فحص {tf_cfg['lbl']}... ({i}/{len(tickers)})**")
                try:
                    raw = yf.download(chunk, period=tf_cfg['p'], interval=tf_cfg['i'], group_by='ticker', auto_adjust=False, threads=True, progress=False)
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
                                    if len(df) > 20:
                                        df = process_data(df)
                                        last = df.iloc[-1]
                                        
                                        if tf_name == 'Daily': # نحفظ اليومي فقط للشارت
                                            link = f"https://www.tradingview.com/chart/?symbol=TADAWUL:{sym.replace('.SR','')}"
                                            st.session_state['history'][name] = df
                                        
                                        # فحص الصناديق (بنوعيها)
                                        bull_boxes = check_bullish_box(df, df['ATR'])
                                        bear_boxes = check_bearish_box(df, df['ATR'])
                                        
                                        # دمج النتائج
                                        all_boxes = bull_boxes + bear_boxes
                                        
                                        if all_boxes:
                                            # نأخذ أحدث صندوق من كل نوع إذا وجد
                                            latest_box = all_boxes[-1] 
                                            
                                            st.session_state['boxes_list'].append({
                                                "Name": name,
                                                "Timeframe": tf_cfg['lbl'],
                                                "Type": latest_box['Type'],
                                                "Price": last['Close'],
                                                "Box_Top": latest_box['Box_Top'],
                                                "Box_Bottom": latest_box['Box_Bottom'],
                                                "Age": latest_box['Days_Ago'],
                                                "TV": f"https://www.tradingview.com/chart/?symbol=TADAWUL:{sym.replace('.SR','')}",
                                                "Raw_TF": tf_name, # للفلترة
                                                "Start_Idx": latest_box['Start_Index'], "End_Idx": latest_box['End_Index'] # للرسم
                                            })
                            except: continue
                except: pass
                curr_step += len(chunk)
                progress.progress(min(curr_step / total_steps, 1.0))
        
        progress.empty(); status.success("تم بناء القائمة الشاملة!")

# --- 6. العرض ---
if st.session_state['boxes_list']:
    df = pd.DataFrame(st.session_state['boxes_list'])
    link_col = st.column_config.LinkColumn("شارت", display_text="TV")

    if selected_tab == "القائمة الشاملة (Master List)":
        # الفلاتر العلوية
        c1, c2, c3 = st.columns(3)
        with c1: filter_type = st.multiselect("نوع الصندوق", ["Bullish", "Bearish"], default=["Bullish", "Bearish"])
        with c2: filter_tf = st.multiselect("الفريم الزمني", ["يومي 📅", "أسبوعي 🗓️", "شهري 📆"], default=["يومي 📅", "أسبوعي 🗓️"])
        with c3: sort_by = st.selectbox("ترتيب حسب", ["الأحدث (Age)", "السعر"])
        
        # تطبيق الفلتر
        view_df = df.copy()
        if filter_type: view_df = view_df[view_df['Type'].isin(filter_type)]
        if filter_tf: view_df = view_df[view_df['Timeframe'].isin(filter_tf)]
        
        if sort_by == "الأحدث (Age)": view_df = view_df.sort_values('Age', ascending=True)
        else: view_df = view_df.sort_values('Price', ascending=False)
        
        # التلوين
        def color_row(row):
            box_col = '#00c853' if row['Type'] == 'Bullish' else '#ff5252' # أخضر للصاعد، أحمر للهابط
            return [f'color: {box_col}; font-weight: bold' if col == 'Type' else '' for col in row.index]

        st.dataframe(
            view_df[['Name', 'Timeframe', 'Type', 'Price', 'Box_Top', 'Box_Bottom', 'Age', 'TV']].style
            .format({"Price": "{:.2f}", "Box_Top": "{:.2f}", "Box_Bottom": "{:.2f}"})
            .apply(color_row, axis=1),
            column_config={"TV": link_col}, use_container_width=True, height=600
        )

    elif selected_tab == "الشارت":
        c_sel, _ = st.columns([1, 3])
        with c_sel: sel_stock = st.selectbox("اختر السهم:", df['Name'].unique())
        
        if sel_stock:
            if sel_stock in st.session_state['history']:
                hist = st.session_state['history'][sel_stock]
                
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.02)
                fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name='Price'), row=1, col=1)
                
                # رسم الصناديق الخاصة بالسهم (كل الأنواع والفريمات)
                stock_boxes = df[df['Name'] == sel_stock]
                for _, box in stock_boxes.iterrows():
                    # نلون حسب النوع والفريم
                    if box['Type'] == 'Bullish':
                        color = "rgba(0, 200, 83, 0.4)" if box['Raw_TF'] == 'Daily' else "rgba(0, 200, 83, 0.2)"
                        border = "#00c853"
                    else:
                        color = "rgba(255, 82, 82, 0.4)" if box['Raw_TF'] == 'Daily' else "rgba(255, 82, 82, 0.2)"
                        border = "#ff5252"
                    
                    # ملاحظة: الرسم هنا تقريبي للفريمات الأكبر لأنه يعتمد على إحداثيات اليومي
                    # لكنه يعطي دلالة بصرية جيدة
                    if box['Raw_TF'] == 'Daily': # نرسم اليومي بدقة
                        fig.add_shape(type="rect", x0=hist.index[-box['Age']], x1=hist.index[-1], y0=box['Box_Bottom'], y1=box['Box_Top'], line=dict(color=border, width=1), fillcolor=color, row=1, col=1)

                fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False, paper_bgcolor='#131722', plot_bgcolor='#131722')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("بيانات الشارت غير متوفرة لهذا السهم (قد يكون صندوق أسبوعي/شهري فقط).")

else:
    st.info("👋 اضغط الزر البرتقالي لبناء القائمة الشاملة.")
