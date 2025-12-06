import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from streamlit_option_menu import option_menu # المكتبة الجديدة للتصميم
import time

# --- استيراد البيانات ---
try:
    from data.saudi_tickers import STOCKS_DB
except ImportError:
    try:
        from saudi_tickers import STOCKS_DB
    except ImportError:
        st.error("🚨 ملف البيانات مفقود.")
        st.stop()

TICKERS = {item['symbol']: item['name'] for item in STOCKS_DB}
SECTORS = {item['name']: item['sector'] for item in STOCKS_DB}

# --- 1. إعداد الصفحة وتصميم الـ CSS الاحترافي ---
st.set_page_config(page_title="TASI.AI Pro", layout="wide", initial_sidebar_state="collapsed")

# حقن CSS متقدم (The Magic Sauce)
st.markdown("""
<style>
    /* استيراد خط تجاري حديث */
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Cairo', sans-serif;
    }

    /* الخلفية العامة (تدرج لوني داكن جداً) */
    .stApp {
        background: linear-gradient(to bottom right, #000000, #131722);
        color: #ffffff;
    }

    /* إخفاء القائمة العلوية الافتراضية */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* تصميم البطاقات (Glassmorphism Cards) */
    div[data-testid="stMetric"] {
        background: rgba(30, 34, 45, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        border-color: #2962ff;
    }
    div[data-testid="stMetricLabel"] { color: #8b9bb4 !important; font-size: 0.9rem; }
    div[data-testid="stMetricValue"] { color: #ffffff !important; font-size: 1.6rem; font-weight: 700; }

    /* تصميم الجداول */
    .stDataFrame { border: none !important; }
    div[data-testid="stDataFrame"] {
        background: rgba(30, 34, 45, 0.4);
        border-radius: 10px;
        padding: 10px;
    }

    /* زر التشغيل الكبير (Neon Glow) */
    .stButton > button {
        background: linear-gradient(90deg, #2962ff, #2979ff);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: bold;
        font-size: 1.1rem;
        box-shadow: 0 0 15px rgba(41, 98, 255, 0.5);
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #0039cb, #2962ff);
        box-shadow: 0 0 25px rgba(41, 98, 255, 0.8);
        transform: scale(1.02);
    }

    /* تحسين القوائم المنسدلة */
    .stSelectbox > div > div {
        background-color: #1e222d;
        color: white;
        border: 1px solid #434651;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. الهيدر والقائمة العلوية (Navigation) ---
c1, c2 = st.columns([1, 6])
with c1:
    # يمكنك وضع لوقو هنا
    st.markdown("## 🦅") 
with c2:
    st.title("TASI.AI Platform")
    st.caption("الجيل القادم من تحليل السوق السعودي")

# قائمة التنقل العلوية (بديلة عن الجانبية)
selected_tab = option_menu(
    menu_title=None,
    options=["الرئيسية", "الماسح الذكي", "الخريطة الحرارية", "الشارت المتقدم"],
    icons=["house", "cpu", "grid", "graph-up"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "transparent"},
        "icon": {"color": "#2962ff", "font-size": "18px"}, 
        "nav-link": {"font-size": "16px", "text-align": "center", "margin": "0px", "--hover-color": "#1e222d", "color": "white"},
        "nav-link-selected": {"background-color": "#2962ff", "font-weight": "bold"},
    }
)

# --- 3. الإعدادات (في Expander مخفي) ---
with st.expander("⚙️ إعدادات الخوارزمية (Algorithm Settings)"):
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1: RSI_PERIOD = st.number_input("RSI Length", 14, 30, 24)
    with col_s2: EMA_PERIOD = st.number_input("EMA Trend", 5, 200, 20)
    with col_s3: ATR_MULT = st.number_input("ATR Factor", 1.0, 3.0, 1.5)
    with col_s4: BOX_LOOKBACK = st.slider("Scan History", 10, 100, 25)

# --- 4. الدوال الفنية (نفس المنطق القوي) ---
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
    avg_gain = gain.ewm(alpha=1/RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['EMA'] = df['Close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Trend Score
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    score = (
        (df['Close'] > df['EMA']).astype(int) + 
        (df['Close'] > df['EMA20']).astype(int) + 
        (df['Close'] > df['EMA50']).astype(int) + 
        (df['Close'] > df['EMA200']).astype(int)
    )
    df['Trend_Score'] = score
    return df

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
                            "Box_Top": max(start_open, final_close), "Box_Bottom": min(start_open, final_close),
                            "Days_Ago": periods_ago, "Start_Index": len(df) - periods_ago - (i - start_idx), "End_Index": len(df) - periods_ago
                        })
                in_series = True; is_bullish = is_green; start_open = open_p; end_close = close; start_idx = i
    return found_boxes

def calculate_ai_score(last, box):
    score = 0
    mid = (box['Box_Top'] + box['Box_Bottom']) / 2
    if last['Close'] > mid: score += 25
    if last['Close'] > last['EMA']: score += 25
    if last['RSI'] > 50: score += 20
    if last['MACD'] > last['Signal']: score += 20
    if last['RVOL'] > 1.2: score += 10
    return min(score, 100)

# --- 5. المنطق (Cache) ---
if 'data' not in st.session_state: st.session_state['data'] = []
if 'history' not in st.session_state: st.session_state['history'] = {}

# زر التحديث
col_b1, col_b2, col_b3 = st.columns([1, 2, 1])
with col_b2:
    run_scan = st.button("⚡ تشغيل المحلل الذكي (Run AI Scanner)")

if run_scan:
    st.session_state['data'] = []
    st.session_state['history'] = {}
    
    progress = st.progress(0)
    status = st.empty()
    tickers_list = list(TICKERS.keys())
    chunk_size = 25
    
    for i in range(0, len(tickers_list), chunk_size):
        chunk = tickers_list[i:i + chunk_size]
        status.markdown(f"**📡 جاري سحب بيانات السوق... ({i}/{len(tickers_list)})**")
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
                            if len(df) > 90:
                                df = process_data(df)
                                last = df.iloc[-1]
                                link = f"https://www.tradingview.com/chart/?symbol=TADAWUL:{sym.replace('.SR','')}"
                                st.session_state['history'][name] = df
                                
                                # Boxes & AI Score
                                boxes = check_bullish_box(df, df['ATR'])
                                ai_score = 0
                                box_status = "لا يوجد"
                                if boxes:
                                    latest = boxes[-1]
                                    if last['Close'] >= latest['Box_Bottom']:
                                        ai_score = calculate_ai_score(last, latest)
                                        box_status = "داخل الصندوق" if last['Close'] <= latest['Box_Top'] else "اختراق"

                                st.session_state['data'].append({
                                    "Name": name, "Symbol": sym, "Sector": SECTORS.get(name, "عام"),
                                    "Price": last['Close'], "Change": last['Change'], 
                                    "RSI": last['RSI'], "Trend": last['Trend_Score'],
                                    "RVOL": last['RVOL'], "AI_Score": ai_score,
                                    "Box_Status": box_status, "TV": link
                                })
                    except: continue
        except: pass
        progress.progress(min((i + chunk_size) / len(tickers_list), 1.0))
    progress.empty()
    status.success("✅ تم تحديث البيانات!")

# --- 6. العرض حسب التبويب المختار ---
if st.session_state['data']:
    df = pd.DataFrame(st.session_state['data'])
    link_col = st.column_config.LinkColumn("شارت", display_text="Open TV")

    # --- الصفحة الرئيسية ---
    if selected_tab == "الرئيسية":
        # KPIs Cards
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("عدد الشركات", len(df), border=True)
        top_gainer = df.loc[df['Change'].idxmax()]
        k2.metric("🚀 النجم اليوم", top_gainer['Name'], f"{top_gainer['Change']:.2f}%", border=True)
        top_ai = df.loc[df['AI_Score'].idxmax()]
        k3.metric("🧠 اختيار الذكاء", top_ai['Name'], f"{top_ai['AI_Score']}/100", border=True)
        k4.metric("السوق أخضر", len(df[df['Change'] > 0]), border=True)
        
        st.markdown("### 📊 نظرة عامة على السوق")
        st.dataframe(
            df[['Name', 'Price', 'Change', 'RSI', 'Trend', 'RVOL', 'TV']].style
            .format({"Price": "{:.2f}", "Change": "{:.2f}%", "RSI": "{:.1f}", "RVOL": "{:.1f}x"})
            .background_gradient(cmap='RdYlGn', subset=['Change']),
            column_config={"TV": link_col}, use_container_width=True, height=500
        )

    # --- الماسح الذكي (AI & Boxes) ---
    elif selected_tab == "الماسح الذكي":
        st.markdown("### 💎 الفرص الذهبية (AI + Boxes)")
        col_f1, col_f2 = st.columns(2)
        with col_f1: min_score = st.slider("فلتر حسب تقييم AI", 0, 100, 60)
        
        filtered_df = df[(df['AI_Score'] >= min_score) & (df['Box_Status'] != "لا يوجد")]
        
        if not filtered_df.empty:
            st.dataframe(
                filtered_df[['Name', 'Price', 'AI_Score', 'Box_Status', 'Trend', 'TV']].sort_values('AI_Score', ascending=False)
                .style.format({"Price": "{:.2f}", "AI_Score": "{:.0f}"})
                .background_gradient(cmap='Greens', subset=['AI_Score']),
                column_config={"TV": link_col}, use_container_width=True
            )
        else:
            st.info("لا توجد شركات تحقق شروط الفلتر الحالي.")

    # --- الخريطة الحرارية ---
    elif selected_tab == "الخريطة الحرارية":
        st.markdown("### 🗺️ خريطة السيولة والقطاعات")
        fig = px.treemap(
            df, path=[px.Constant("السوق"), 'Sector', 'Name'], values='Price',
            color='Trend', color_continuous_scale='RdYlGn', range_color=[0, 4],
            custom_data=['Symbol', 'Price', 'Change']
        )
        fig.update_traces(hovertemplate="<b>%{label}</b><br>السعر: %{customdata[1]:.2f}<br>التغير: %{customdata[2]:.2f}%")
        fig.update_layout(margin=dict(t=0, l=0, r=0, b=0), height=600, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    # --- الشارت المتقدم ---
    elif selected_tab == "الشارت المتقدم":
        col_c1, col_c2 = st.columns([1, 3])
        with col_c1:
            sel_stock = st.selectbox("اختر السهم:", df['Name'].unique())
        
        if sel_stock:
            hist = st.session_state['history'][sel_stock]
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.02)
            
            # الشموع
            fig.add_trace(go.Candlestick(
                x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'],
                name='Price', increasing_line_color='#00E676', decreasing_line_color='#FF5252'
            ), row=1, col=1)
            
            # المتوسطات
            fig.add_trace(go.Scatter(x=hist.index, y=hist['EMA'], line=dict(color='#2979FF', width=2), name='EMA 20'), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['EMA50'], line=dict(color='#FFEA00', width=1.5), name='EMA 50'), row=1, col=1)
            
            # الصندوق
            box_res = check_bullish_box(hist, hist['ATR'])
            if box_res:
                latest = box_res[-1]
                mid = (latest['Box_Top'] + latest['Box_Bottom']) / 2
                fig.add_shape(type="rect", x0=hist.index[-latest['Days_Ago']], x1=hist.index[-1], y0=latest['Box_Bottom'], y1=latest['Box_Top'],
                              line=dict(color="rgba(0, 230, 118, 0.6)", width=1), fillcolor="rgba(0, 230, 118, 0.1)", row=1, col=1)
                fig.add_shape(type="line", x0=hist.index[-latest['Days_Ago']], x1=hist.index[-1], y0=mid, y1=mid,
                              line=dict(color="#2979FF", width=1, dash="dot"), row=1, col=1)

            # الحجم
            colors = ['#00E676' if c >= o else '#FF5252' for c, o in zip(hist['Close'], hist['Open'])]
            fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], marker_color=colors, name='Vol'), row=2, col=1)
            
            # تنسيق TradingView الداكن
            fig.update_layout(
                template="plotly_dark", height=600, xaxis_rangeslider_visible=False,
                paper_bgcolor='#131722', plot_bgcolor='#131722',
                xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#2a2e39')
            )
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👋 مرحباً بك في TASI.AI Pro. اضغط زر التشغيل للبدء.")
