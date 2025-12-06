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

# --- 1. إعداد الصفحة ---
st.set_page_config(page_title="Saudi Pro Ultimate MTF", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    .stDataFrame { border: 1px solid #30333d; }
    div[data-testid="stDataFrame"] div[class*="css"] { background-color: #161b24; color: white; }
    div[data-testid="stMetric"] { background-color: #1d212b !important; border: 1px solid #30333d; padding: 10px; border-radius: 8px; }
    div[data-testid="stMetricValue"] { color: #ffffff !important; }
    /* زر تشغيل مميز */
    div.stButton > button { background: linear-gradient(45deg, #FFD700, #FF8C00) !important; color: black !important; border: none; width: 100%; font-weight: 900; padding: 15px; border-radius: 12px; font-size: 18px; text-transform: uppercase; box-shadow: 0 4px 15px rgba(255, 215, 0, 0.4); transition: all 0.3s ease; }
    div.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(255, 215, 0, 0.6); }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: #1d212b; color: #e0e0e0; border-radius: 4px; border: 1px solid #333; }
    .stTabs [aria-selected="true"] { background-color: #FFD700 !important; color: black !important; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- 2. الإعدادات ---
with st.sidebar:
    st.header("⚙️ إعدادات التحليل")
    st.info("ملاحظة: التحليل الأسبوعي والشهري يستغرق وقتاً أطول في جلب البيانات. يرجى التحلي بالصبر.")
    RSI_PERIOD = st.number_input("RSI Period", value=24)
    EMA_PERIOD = st.number_input("EMA Trend Period", value=20)
    st.divider()
    st.subheader("📦 إعدادات الصندوق")
    ATR_MULT = st.number_input("ATR Multiplier", value=1.5)
    BOX_LOOKBACK = st.slider("Box Lookback (Periods)", 5, 50, 25, help="عدد الشموع السابقة للبحث عن صندوق (يومي/أسبوعي/شهري)")

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
    
    # RVOL (Relative Volume)
    df['Vol_Avg'] = df['Volume'].rolling(window=20).mean()
    df['RVOL'] = df['Volume'] / df['Vol_Avg']
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/24, min_periods=24, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/24, min_periods=24, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Trend EMAs
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA86'] = df['Close'].ewm(span=86, adjust=False).mean()
    
    # Trend Score (لليومي فقط)
    df['Trend_Score'] = (
        (df['Close'] > df['Close'].ewm(span=8, adjust=False).mean()).astype(int) + 
        (df['Close'] > df['EMA20']).astype(int) + 
        (df['Close'] > df['Close'].ewm(span=40, adjust=False).mean()).astype(int) + 
        (df['Close'] > df['EMA86']).astype(int)
    )
    return df

# 🔥 دالة كشف الصناديق (Advanced)
def check_bullish_box_advanced(df):
    in_series = False; is_bullish = False; start_open = 0.0; end_close = 0.0
    start_idx = 0; found_boxes = []
    
    # نأخذ آخر N شمعة حسب الإعدادات
    lookback = min(len(df), 150) # ننظر للوراء 150 شمعة كحد أقصى
    prices = df.iloc[-lookback:].reset_index()
    atrs = df['ATR'].iloc[-lookback:].values
    rvols = df['RVOL'].iloc[-lookback:].values
    
    for i in range(len(prices)):
        row = prices.iloc[i]; close = row['Close']; open_p = row['Open']
        is_green = close > open_p; is_red = close < open_p
        current_atr = atrs[i]
        if np.isnan(current_atr) or current_atr == 0: continue
        
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
                        box_rvols = rvols[start_idx:i]
                        avg_box_rvol = np.mean(box_rvols) if len(box_rvols) > 0 else 1.0
                        found_boxes.append({
                            # نخزن الإندكس النسبي للشارت بدلاً من التاريخ
                            "Start_Index": len(df) - periods_ago - (i - start_idx), 
                            "End_Index": len(df) - periods_ago,
                            "Box_Top": max(start_open, final_close),
                            "Box_Bottom": min(start_open, final_close),
                            "Periods_Ago": periods_ago,
                            "Box_Avg_RVOL": avg_box_rvol
                        })
                in_series = True; is_bullish = is_green; start_open = open_p; end_close = close; start_idx = i
    return found_boxes

# 🧠 دالة حساب التقييم الذكي (AI Score)
def calculate_ai_score(last_row, box_info):
    score = 0; reasons = []
    if box_info['Box_Avg_RVOL'] >= 1.5: score += 30; reasons.append("سيولة عالية 🔥")
    elif box_info['Box_Avg_RVOL'] >= 1.0: score += 15
    mid = (box_info['Box_Top'] + box_info['Box_Bottom']) / 2
    if last_row['Close'] > mid: score += 20; reasons.append("فوق المنتصف")
    if last_row['Close'] > last_row['EMA20']: score += 20; reasons.append("ترند صاعد")
    if last_row['RSI'] > 50: score += 15
    if 'MACD' in last_row and last_row['MACD'] > last_row['Signal']: score += 15; reasons.append("MACD إيجابي")
    return min(score, 100), reasons

# --- 4. التشغيل الرئيسي (Main Loop) ---
st.title("🚀 Saudi Pro Ultimate: تحليل متعدد الفريمات")

# تهيئة الذاكرة
for k in ['data', 'signals', 'boxes', 'history_daily', 'history_weekly', 'history_monthly']:
    if k not in st.session_state: st.session_state[k] = [] if k not in ['history_daily', 'history_weekly', 'history_monthly'] else {}

if st.button("🌟 بدء التحليل الشامل (يومي - أسبوعي - شهري) 🌟"):
    # تصفير البيانات القديمة
    st.session_state['data'] = []
    st.session_state['signals'] = []
    st.session_state['boxes'] = []
    st.session_state['history_daily'] = {}
    st.session_state['history_weekly'] = {}
    st.session_state['history_monthly'] = {}
    
    prog = st.progress(0)
    status = st.empty()
    tickers_list = list(TICKERS.keys())
    chunk_size = 15 # تقليل الحجم بسبب زيادة الحمل
    
    # إعدادات الفريمات
    TIMEFRAMES = {
        'Daily': {'interval': '1d', 'period': '1y', 'hist_key': 'history_daily'},
        'Weekly': {'interval': '1wk', 'period': '2y', 'hist_key': 'history_weekly'},
        'Monthly': {'interval': '1mo', 'period': '5y', 'hist_key': 'history_monthly'}
    }
    
    total_steps = len(tickers_list) * len(TIMEFRAMES)
    current_step = 0
    
    for i in range(0, len(tickers_list), chunk_size):
        chunk = tickers_list[i:i + chunk_size]
        
        # تكرار التحليل لكل فريم زمني
        for tf_name, tf_config in TIMEFRAMES.items():
            status.text(f"جاري تحليل القطاع {i//chunk_size + 1} - الفريم: {tf_name} ⏳")
            
            try:
                # جلب البيانات حسب الفريم
                raw = yf.download(chunk, period=tf_config['period'], interval=tf_config['interval'], group_by='ticker', auto_adjust=False, threads=True, progress=False)
                
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
                                min_bars = 50 if tf_name == 'Daily' else (30 if tf_name == 'Weekly' else 12)
                                if len(df) > min_bars:
                                    df = process_data(df)
                                    last = df.iloc[-1]
                                    link = f"https://www.tradingview.com/chart/?symbol=TADAWUL:{sym.replace('.SR','')}"
                                    
                                    # تخزين التاريخ للشارت
                                    st.session_state[tf_config['hist_key']][name] = df
                                    
                                    # البيانات الأساسية (نأخذها من اليومي فقط لتجنب التكرار)
                                    if tf_name == 'Daily':
                                        st.session_state['data'].append({
                                            "Name": name, "Symbol": sym, "Sector": SECTORS.get(name, "عام"),
                                            "Price": last['Close'], "Change": last['Change'], 
                                            "RSI": last['RSI'], "Trend_Score": last['Trend_Score'],
                                            "TV": link
                                        })
                                    
                                    # --- تحليل الصناديق (لكل الفريمات) ---
                                    boxes = check_bullish_box_advanced(df)
                                    if boxes:
                                        latest = boxes[-1]
                                        # شرط: السعر الحالي لم يكسر قاع الصندوق
                                        if last['Close'] >= latest['Box_Bottom']:
                                            ai_score, ai_reasons = calculate_ai_score(last, latest)
                                            
                                            st.session_state['boxes'].append({
                                                "الاسم": name,
                                                "الفريم": tf_name, # 📅 يومي / 🗓️ أسبوعي / 📆 شهري
                                                "السعر": last['Close'],
                                                "AI Score": ai_score,
                                                "الأسباب": ", ".join(ai_reasons),
                                                "السيولة": f"x{latest['Box_Avg_RVOL']:.1f}",
                                                "منذ (شمعة)": latest['Periods_Ago'],
                                                "TV": link,
                                                # بيانات رسم الشارت
                                                "Box_Top": latest['Box_Top'],
                                                "Box_Bottom": latest['Box_Bottom'],
                                                "Start_Index": latest['Start_Index'],
                                                "End_Index": latest['End_Index']
                                            })
                                            
            except Exception as e: print(f"Error in {tf_name}: {e}")
            current_step += len(chunk)
            prog.progress(min(current_step / total_steps, 1.0))
            
    prog.empty()
    status.success("✅ اكتمل التحليل الشامل لجميع الفريمات!")

# --- 5. العرض ---
if st.session_state['data']:
    df_daily = pd.DataFrame(st.session_state['data'])
    df_boxes = pd.DataFrame(st.session_state['boxes']) if st.session_state['boxes'] else pd.DataFrame()
    
    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("شركات السوق", len(df_daily))
    
    # إحصائيات الصناديق
    daily_boxes = len(df_boxes[df_boxes['الفريم'] == 'Daily']) if not df_boxes.empty else 0
    weekly_boxes = len(df_boxes[df_boxes['الفريم'] == 'Weekly']) if not df_boxes.empty else 0
    monthly_boxes = len(df_boxes[df_boxes['الفريم'] == 'Monthly']) if not df_boxes.empty else 0
    
    c2.metric("صناديق يومية 📅", daily_boxes)
    c3.metric("صناديق أسبوعية 🗓️", weekly_boxes)
    c4.metric("صناديق شهرية 📆", monthly_boxes)
    
    st.divider()
    
    tabs = st.tabs(["💎 الصناديق (متعدد الفريمات)", "🗺️ الخريطة اليومية", "📈 الشارت الاحترافي"])
    link_col = st.column_config.LinkColumn("شارت", display_text="Open TV")
    
    # --- TAB 1: Multi-Timeframe Boxes ---
    with tabs[0]:
        if not df_boxes.empty:
            st.markdown("### 💎 تحليل الصناديق الاستراتيجية (يومي - أسبوعي - شهري)")
            st.caption("ابحث عن الأسهم التي لديها صناديق على فريمات كبيرة (أسبوعي/شهري) فهي الأقوى.")
            
            # ترتيب حسب الأهمية (شهري > أسبوعي > يومي) ثم الـ AI Score
            df_boxes['Timeframe_Rank'] = df_boxes['الفريم'].map({'Monthly': 3, 'Weekly': 2, 'Daily': 1})
            df_boxes = df_boxes.sort_values(by=['Timeframe_Rank', 'AI Score'], ascending=[False, False]).drop(columns=['Timeframe_Rank'])
            
            # تنسيق الأعمدة
            def color_timeframe(val):
                if val == 'Monthly': return 'color: #ff5252; font-weight: bold;' # أحمر للشهري
                elif val == 'Weekly': return 'color: #FFD700; font-weight: bold;' # ذهبي للأسبوعي
                else: return 'color: #69f0ae;' # أخضر لليومي
                
            st.dataframe(
                df_boxes[['الاسم', 'الفريم', 'السعر', 'AI Score', 'السيولة', 'الأسباب', 'TV']].style
                .format({"السعر": "{:.2f}", "AI Score": "{:.0f}"})
                .map(color_timeframe, subset=['الفريم'])
                .background_gradient(cmap='Greens', subset=['AI Score']),
                column_config={"TV": link_col}, use_container_width=True, height=600
            )
        else: st.info("لم يتم العثور على صناديق نشطة.")

    # --- TAB 2: Map ---
    with tabs[1]:
        st.subheader("خريطة السوق اليومية")
        fig_map = px.treemap(
            df_daily, path=[px.Constant("السوق"), 'Sector', 'Name'], values='Price',
            color='Trend_Score', color_continuous_scale='RdYlGn', range_color=[0, 4],
            custom_data=['Symbol', 'TV', 'Price']
        )
        fig_map.update_traces(hovertemplate="<b>%{label}</b><br>السعر: %{customdata[2]:.2f}<br>الترند: %{color:.0f}/4")
        fig_map.update_layout(margin=dict(t=0, l=0, r=0, b=0), height=500)
        st.plotly_chart(fig_map, use_container_width=True)

    # --- TAB 3: Chart (The Ultimate MTF Chart) ---
    with tabs[2]:
        st.subheader("📈 الشارت الاحترافي (يظهر جميع الصناديق)")
        sel = st.selectbox("اختر سهم لعرض تحليله الشامل:", df_daily['Name'].unique(), key="chart_select_mtf")
        
        if sel:
            # نستخدم البيانات اليومية كأساس للشارت
            hist = st.session_state['history_daily'][sel]
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.03)

            # 1. رسم الشموع اليومية
            fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name='Price (Daily)'), row=1, col=1)

            # 2. رسم المتوسطات
            fig.add_trace(go.Scatter(x=hist.index, y=hist['EMA20'], line=dict(color='#2962ff', width=1), name='EMA 20 (Daily)'), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['EMA86'], line=dict(color='#ff6d00', width=2), name='EMA 86 (Daily)'), row=1, col=1)

            # 3. رسم الحجم
            colors_vol = ['#00c853' if c >= o else '#ff5252' for c, o in zip(hist['Close'], hist['Open'])]
            fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], marker_color=colors_vol, opacity=0.5, name='Volume'), row=2, col=1)
            
            # --- 4. رسم مؤشرات الصناديق المتعددة الفريمات ---
            if not df_boxes.empty:
                # فلترة الصناديق الخاصة بهذا السهم فقط
                stock_boxes = df_boxes[df_boxes['الاسم'] == sel]
                
                for _, box in stock_boxes.iterrows():
                    tf = box['الفريم']
                    # تحديد الستايل حسب الفريم
                    if tf == 'Monthly':
                        box_color = "rgba(255, 82, 82, 0.9)"; line_width = 3; border_color = "#ff5252" # أحمر عريض للشهري
                    elif tf == 'Weekly':
                        box_color = "rgba(255, 215, 0, 0.7)"; line_width = 2; border_color = "#FFD700" # ذهبي للأسبوعي
                    else:
                        box_color = "rgba(0, 200, 83, 0.5)"; line_width = 1; border_color = "#00c853" # أخضر لليومي

                    # رسم المستطيل باستخدام الإندكس لضمان الدقة
                    try:
                        x0_idx = box['Start_Index']; x1_idx = box['End_Index']
                        # التأكد من أن الإندكس داخل حدود البيانات اليومية
                        if x0_idx < len(hist) and x1_idx < len(hist):
                            fig.add_shape(
                                type="rect",
                                x0=hist.index[x0_idx], x1=hist.index[x1_idx], # استخدام تواريخ البيانات اليومية
                                y0=box['Box_Bottom'], y1=box['Box_Top'],
                                line=dict(color=border_color, width=line_width),
                                fillcolor=box_color, layer="below", row=1, col=1
                            )
                    except: continue # تجاوز أي خطأ في الرسم

            # تنسيق الشارت
            fig.update_layout(
                template="plotly_dark", height=700, xaxis_rangeslider_visible=False,
                paper_bgcolor='#131722', plot_bgcolor='#131722',
                title=f"تحليل {sel} - مناطق التجميع (يومي/أسبوعي/شهري)",
                legend=dict(orientation="h", y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)

else:
    st.markdown("---")
    st.header("👋 مرحباً بك في الإصدار الأقوى (Ultimate MTF)")
    st.info("اضغط الزر الذهبي في الأعلى لبدء تحليل السوق على الفريمات: اليومي، الأسبوعي، والشهري.")
