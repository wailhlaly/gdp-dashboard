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
st.set_page_config(page_title="TASI Pro V9.1", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif; }
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    .stDataFrame { border: 1px solid #30333d; }
    div[data-testid="stDataFrame"] div[class*="css"] { background-color: #161b24; color: white; }
    div[data-testid="stMetric"] { background-color: #1d212b !important; border: 1px solid #464b5f !important; padding: 15px; border-radius: 10px; }
    [data-testid="stMetricValue"] { color: #ffffff !important; }
    
    /* زر التحديث */
    div.stButton > button {
        background: linear-gradient(90deg, #2962ff, #0039cb);
        color: white; border: none; padding: 12px; border-radius: 8px; font-weight: bold; width: 100%;
    }
    
    /* القوائم */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: #1d212b; color: #e0e0e0; border-radius: 4px; }
    .stTabs [aria-selected="true"] { background-color: #2962ff !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# --- 2. القائمة العلوية ---
selected_tab = option_menu(
    menu_title=None,
    options=["القائمة الشاملة", "الخريطة الحرارية (Pro)", "الشارت الفني"],
    icons=["list-task", "grid", "graph-up"],
    default_index=1, # جعلنا الخريطة هي الافتراضية
    orientation="horizontal",
    styles={"container": {"background-color": "transparent"}, "nav-link-selected": {"background-color": "#2962ff"}}
)

# --- 3. الإعدادات ---
with st.sidebar:
    st.header("⚙️ إعدادات")
    RSI_PERIOD = st.number_input("RSI Period", 14, 30, 24)
    EMA_PERIOD = st.number_input("EMA Trend", 10, 200, 20)
    ATR_MULT = st.number_input("ATR Multiplier", 1.0, 3.0, 1.5)
    BOX_LOOKBACK = st.slider("نطاق الصندوق", 10, 100, 25)

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
    
    delta = df['Close'].diff()
    gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/24, min_periods=24, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/24, min_periods=24, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['EMA'] = df['Close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    return df

def check_boxes(df, atr_series, box_type='bull'):
    in_series = False; mode_active = False; start_open = 0.0; end_close = 0.0; start_idx = 0; found_boxes = []
    prices = df.iloc[-150:].reset_index(); atrs = atr_series.iloc[-150:].values
    
    for i in range(len(prices)):
        row = prices.iloc[i]; close = row['Close']; open_p = row['Open']
        is_green = close > open_p; is_red = close < open_p
        current_atr = atrs[i]
        if np.isnan(current_atr): continue
        
        condition_start = is_green if box_type == 'bull' else is_red
        condition_break = is_red if box_type == 'bull' else is_green
        
        if not in_series:
            if condition_start: in_series = True; mode_active = True; start_open = open_p; start_idx = i
            elif condition_break: in_series = True; mode_active = False; start_open = open_p; start_idx = i
        elif in_series:
            if mode_active and condition_start: end_close = close
            elif not mode_active and condition_break: end_close = close
            elif (mode_active and condition_break) or (not mode_active and condition_start):
                final_close = end_close if end_close != 0 else start_open
                price_move = abs(final_close - start_open)
                if price_move >= current_atr * ATR_MULT and mode_active:
                    periods_ago = len(prices) - i
                    if periods_ago <= BOX_LOOKBACK:
                        found_boxes.append({
                            "Type": "Bullish" if box_type == 'bull' else "Bearish",
                            "Box_Top": max(start_open, final_close), "Box_Bottom": min(start_open, final_close),
                            "Days_Ago": periods_ago, "Start_Index": len(df) - periods_ago - (i - start_idx), "End_Index": len(df) - periods_ago
                        })
                in_series = True; mode_active = True if condition_start else False; start_open = open_p; end_close = close; start_idx = i
    return found_boxes

# --- 5. المنطق (Engine) ---
if 'boxes_list' not in st.session_state: st.session_state['boxes_list'] = []
if 'history' not in st.session_state: st.session_state['history'] = {}
if 'market_summary' not in st.session_state: st.session_state['market_summary'] = []

c1, c2 = st.columns([1, 4])
with c2:
    if st.button("🚀 تحديث البيانات (Live Update)"):
        st.session_state['boxes_list'] = []
        st.session_state['history'] = {}
        st.session_state['market_summary'] = []
        
        progress = st.progress(0); status = st.empty()
        tickers = list(TICKERS.keys())
        
        TIMEFRAMES = {
            'Daily': {'p': '1y', 'i': '1d', 'lbl': 'يومي 📅'},
            'Weekly': {'p': '2y', 'i': '1wk', 'lbl': 'أسبوعي 🗓️'}
        }
        
        total_steps = len(tickers) * 2
        curr_step = 0
        chunk_size = 30
        
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i:i + chunk_size]
            
            for tf_name, tf_cfg in TIMEFRAMES.items():
                status.text(f"معالجة {tf_cfg['lbl']}... ({i}/{len(tickers)})")
                try:
                    raw = yf.download(chunk, period=tf_cfg['p'], interval=tf_cfg['i'], group_by='ticker', auto_adjust=False, threads=True, progress=False)
                    if not raw.empty:
                        for sym in chunk:
                            try:
                                name = TICKERS.get(sym, sym)
                                try: df = raw[sym].copy()
                                except: continue
                                
                                col = 'Close' if 'Close' in df.columns else 'Adj Close'
                                if col in df.columns:
                                    df = df.rename(columns={col: 'Close'})
                                    df = df.dropna()
                                    if len(df) > 20:
                                        df = process_data(df)
                                        last = df.iloc[-1]
                                        
                                        if tf_name == 'Daily':
                                            st.session_state['history'][name] = df
                                            st.session_state['market_summary'].append({
                                                "Name": name, "Symbol": sym, "Sector": SECTORS.get(name, "عام"),
                                                "Price": last['Close'], "Change": last['Change'],
                                                "Volume": last['Volume']
                                            })
                                        
                                        bulls = check_boxes(df, df['ATR'], 'bull')
                                        bears = check_boxes(df, df['ATR'], 'bear')
                                        all_boxes = bulls + bears
                                        
                                        if all_boxes:
                                            latest_box = all_boxes[-1]
                                            st.session_state['boxes_list'].append({
                                                "Name": name, "Timeframe": tf_cfg['lbl'],
                                                "Type": latest_box['Type'], "Price": last['Close'],
                                                "Box_Top": latest_box['Box_Top'], "Box_Bottom": latest_box['Box_Bottom'],
                                                "Age": latest_box['Days_Ago'], "Raw_TF": tf_name,
                                                "TV": f"https://www.tradingview.com/chart/?symbol=TADAWUL:{sym.replace('.SR','')}"
                                            })
                            except: continue
                except: pass
                curr_step += len(chunk)
                progress.progress(min(curr_step / total_steps, 1.0))
        
        progress.empty(); status.success("تم التحديث!")

# --- 6. العرض ---
if st.session_state['market_summary']:
    df_market = pd.DataFrame(st.session_state['market_summary'])
    df_boxes = pd.DataFrame(st.session_state['boxes_list'])
    link_col = st.column_config.LinkColumn("شارت", display_text="TV")

    # --- تبويب الخريطة الحرارية (Heatmap) ---
    if selected_tab == "الخريطة الحرارية (Pro)":
        if not df_market.empty:
            
            # --- الألوان الاحترافية (TradingView Style) ---
            # أخضر هادئ (ربح) - رمادي غامق (حياد) - أحمر هادئ (خسارة)
            fig = px.treemap(
                df_market, 
                path=[px.Constant("السوق السعودي"), 'Sector', 'Name'], 
                values='Price', # حجم المربع حسب السعر (أو القيمة السوقية لو توفرت)
                color='Change',
                color_continuous_scale=[
                    (0, "#f23645"),    # أحمر داكن (خسارة كبيرة)
                    (0.5, "#2a2e39"),  # رمادي مزرق (حياد)
                    (1, "#089981")     # أخضر داكن (ربح كبير)
                ],
                range_color=[-3, 3], # تثبيت النطاق لتوحيد الألوان (من -3% إلى +3%)
                custom_data=['Symbol', 'Price', 'Change']
            )
            
            # تحسين النصوص داخل المربعات
            fig.update_traces(
                texttemplate="<b>%{label}</b><br>%{customdata[2]:.2f}%", 
                textfont=dict(size=14, color='white'),
                hovertemplate="<b>%{label}</b><br>السعر: %{customdata[1]:.2f}<br>التغير: %{customdata[2]:.2f}%<extra></extra>",
                marker=dict(line=dict(color='#131722', width=1)) # حدود فواصل أنيقة
            )
            
            fig.update_layout(
                margin=dict(t=0, l=0, r=0, b=0),
                height=650,
                paper_bgcolor='#131722', # لون الخلفية
                coloraxis_colorbar=dict(title="التغير %", tickfont=dict(color='white'))
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else: st.info("البيانات غير متوفرة.")

    # --- تبويب القائمة الشاملة ---
    elif selected_tab == "القائمة الشاملة":
        if not df_boxes.empty:
            c1, c2, c3 = st.columns(3)
            with c1: f_type = st.multiselect("النوع", ["Bullish", "Bearish"], default=["Bullish"])
            with c2: f_tf = st.multiselect("الفريم", ["يومي 📅", "أسبوعي 🗓️"], default=["يومي 📅"])
            with c3: f_sort = st.selectbox("ترتيب", ["الأحدث (Age)", "السعر"])
            
            view = df_boxes.copy()
            if f_type: view = view[view['Type'].isin(f_type)]
            if f_tf: view = view[view['Timeframe'].isin(f_tf)]
            if f_sort == "الأحدث (Age)": view = view.sort_values('Age')
            else: view = view.sort_values('Price', ascending=False)
            
            # تلوين الجدول بشكل احترافي
            def color_row(row):
                # أخضر للصاعد، أحمر للهابط
                color = '#00c853' if row['Type'] == 'Bullish' else '#ff5252'
                return [f'color: {color}; font-weight: bold' if col in ['Type', 'Name'] else '' for col in row.index]

            st.dataframe(
                view[['Name', 'Timeframe', 'Type', 'Price', 'Box_Top', 'Box_Bottom', 'Age', 'TV']].style
                .format({"Price": "{:.2f}", "Box_Top": "{:.2f}", "Box_Bottom": "{:.2f}"})
                .apply(color_row, axis=1),
                column_config={"TV": link_col}, use_container_width=True, height=600
            )
        else: st.info("لا توجد صناديق.")

    # --- تبويب الشارت ---
    elif selected_tab == "الشارت الفني":
        c_sel, _ = st.columns([1, 3])
        with c_sel: sel_stock = st.selectbox("اختر السهم:", df_market['Name'].unique())
        
        if sel_stock and sel_stock in st.session_state['history']:
            hist = st.session_state['history'][sel_stock]
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.02)
            
            fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name='Price', increasing_line_color='#089981', decreasing_line_color='#f23645'), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['EMA'], line=dict(color='#2962ff', width=1.5), name=f'EMA {EMA_PERIOD}'), row=1, col=1)
            
            # رسم الصناديق بدقة
            if not df_boxes.empty:
                stock_boxes = df_boxes[df_boxes['Name'] == sel_stock]
                for _, box in stock_boxes.iterrows():
                    if box['Raw_TF'] == 'Daily': # رسم اليومي فقط للشارت اليومي
                        color = "rgba(8, 153, 129, 0.2)" if box['Type']=='Bullish' else "rgba(242, 54, 69, 0.2)"
                        border = "#089981" if box['Type']=='Bullish' else "#f23645"
                        fig.add_shape(type="rect", x0=hist.index[-box['Age']], x1=hist.index[-1], y0=box['Box_Bottom'], y1=box['Box_Top'], line=dict(color=border, width=1), fillcolor=color, row=1, col=1)

            fig.add_trace(go.Scatter(x=hist.index, y=hist['RSI'], line=dict(color='#b2b5be', width=1.5), name='RSI'), row=2, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="#f23645", row=2, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="#089981", row=2, col=1)
            
            fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False, paper_bgcolor='#131722', plot_bgcolor='#131722', margin=dict(l=0, r=50, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👋 جاهز! اضغط الزر الأزرق.")
