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
import calendar

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
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif; }

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

def check_bullish_box(df, atr_series): # تم تصحيح المعاملات
    in_series = False; is_bullish = False; start_open = 0.0; end_close = 0.0; start_idx = 0; found_boxes = []
    prices = df.iloc[-100:].reset_index() if len(df) > 100 else df.reset_index()
    atrs = atr_series.iloc[-100:].values if len(df) > 100 else atr_series.values
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

def check_divergence(df, order=5):
    price_lows = argrelextrema(df['Low'].values, np.less_equal, order=order)[0]
    rsi_lows = argrelextrema(df['RSI'].values, np.less_equal, order=order)[0]
    if len(price_lows) >= 2 and len(rsi_lows) >= 2:
        p_last = price_lows[-1]; p_prev = price_lows[-2]
        if (len(df) - p_last) <= 15:
            if df['Low'].iloc[p_last] <= df['Low'].iloc[p_prev] and df['RSI'].iloc[p_last] > df['RSI'].iloc[p_prev]:
                return "إيجابي ✅"
    return "لا"

# --- 4. القائمة العلوية ---
selected_tab = option_menu(
    menu_title=None,
    options=["الرئيسية", "الماسح الذكي", "كاشف الانفراجات", "الخريطة", "تحليل السهم العميق"],
    icons=["house", "cpu", "eye", "grid", "graph-up-arrow"],
    default_index=0,
    orientation="horizontal",
    styles={"container": {"background-color": "transparent"}, "nav-link-selected": {"background-color": "#2962ff"}}
)

# --- 5. المنطق (Engine) ---
if 'data' not in st.session_state: st.session_state['data'] = []
if 'history' not in st.session_state: st.session_state['history'] = {}

# زر التحديث
c1, c2 = st.columns([1, 4])
with c2:
    run_scan = st.button("🚀 تحديث البيانات (Live Scan)")

if run_scan:
    st.session_state['data'] = []
    st.session_state['history'] = {}
    
    progress = st.progress(0)
    status = st.empty()
    tickers_list = list(TICKERS.keys())
    chunk_size = 25
    
    for i in range(0, len(tickers_list), chunk_size):
        chunk = tickers_list[i:i + chunk_size]
        status.caption(f"جاري معالجة الدفعة {i//chunk_size + 1}...")
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
                                
                                # Boxes Logic
                                boxes = check_bullish_box(df, df['ATR'])
                                ai_score = 0; box_status = "لا يوجد"; box_age = 0
                                if boxes:
                                    latest = boxes[-1]
                                    box_age = latest['Days_Ago']
                                    if last['Close'] >= latest['Box_Bottom']:
                                        ai_score, _ = calculate_ai_score(last, latest)
                                        box_status = "نشط"
                                
                                div_status = check_divergence(df)

                                st.session_state['data'].append({
                                    "Name": name, "Symbol": sym, "Sector": SECTORS.get(name, "عام"),
                                    "Price": last['Close'], "Change": last['Change'], 
                                    "RSI": last['RSI'], "Trend": last['Trend_Score'],
                                    "RVOL": last['RVOL'], "AI_Score": ai_score,
                                    "Box_Status": box_status, "Box_Age": box_age,
                                    "Divergence": div_status, "TV": link
                                })
                    except: continue
        except Exception as e:
            print(f"Error: {e}") # طباعة الخطأ في الكونسول بدلاً من توقف البرنامج
            
        progress.progress(min((i + chunk_size) / len(tickers_list), 1.0))
    progress.empty()
    status.success("✅ تم التحديث بنجاح!")

# --- 6. العرض (Dashboard UI) ---
if st.session_state['data']:
    df = pd.DataFrame(st.session_state['data'])
    link_col = st.column_config.LinkColumn("شارت", display_text="Open TV")

    # --- الرئيسية ---
    if selected_tab == "الرئيسية":
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("عدد الشركات", len(df))
        # حل مشكلة KeyError: نستخدم 'Change' كما خزنّاها
        bullish_count = len(df[df['Change'] > 0])
        k2.metric("السوق أخضر", bullish_count)
        k3.metric("صناديق ذهبية", len(df[df['AI_Score'] >= 70]))
        k4.metric("سيولة عالية", len(df[df['RVOL'] > 2.0]))
        
        st.markdown("### 📊 ملخص السوق")
        st.dataframe(
            df[['Name', 'Price', 'Change', 'RSI', 'Trend', 'RVOL', 'TV']].style
            .format({"Price": "{:.2f}", "Change": "{:.2f}%", "RSI": "{:.1f}", "RVOL": "{:.1f}x"})
            .background_gradient(cmap='RdYlGn', subset=['Change']),
            column_config={"TV": link_col}, use_container_width=True, height=600
        )

    # --- الماسح الذكي ---
    elif selected_tab == "الماسح الذكي":
        st.markdown("### 📦 الصناديق الذكية (AI Scored)")
        score_filter = st.slider("الحد الأدنى للتقييم", 0, 100, 60)
        filtered = df[(df['AI_Score'] >= score_filter) & (df['Box_Status'] != "لا يوجد")]
        
        if not filtered.empty:
            st.dataframe(
                filtered[['Name', 'Price', 'AI_Score', 'Box_Status', 'Box_Age', 'TV']].sort_values('AI_Score', ascending=False)
                .style.format({"Price": "{:.2f}", "AI_Score": "{:.0f}"})
                .background_gradient(cmap='Greens', subset=['AI_Score']),
                column_config={"TV": link_col}, use_container_width=True
            )
        else: st.info("لا توجد نتائج.")

    # --- كاشف الانفراجات ---
    elif selected_tab == "كاشف الانفراجات":
        st.markdown("### 🦅 فرص الانفراج (RSI Divergence)")
        div_df = df[df['Divergence'] != "لا يوجد"]
        if not div_df.empty:
            st.dataframe(
                div_df[['Name', 'Price', 'RSI', 'Divergence', 'TV']]
                .style.format({"Price": "{:.2f}", "RSI": "{:.1f}"})
                .applymap(lambda v: 'background-color: #1b5e20; color: white;', subset=['Divergence']),
                column_config={"TV": link_col}, use_container_width=True
            )
        else: st.info("لم يتم رصد انفراجات مؤكدة اليوم.")

    # --- الخريطة ---
    elif selected_tab == "الخريطة":
        fig = px.treemap(
            df, path=[px.Constant("السوق"), 'Sector', 'Name'], values='Price',
            color='Change', color_continuous_scale='RdYlGn', color_continuous_midpoint=0,
            custom_data=['Symbol', 'Price', 'Change']
        )
        fig.update_traces(hovertemplate="<b>%{label}</b><br>السعر: %{customdata[1]:.2f}<br>التغير: %{customdata[2]:.2f}%")
        fig.update_layout(margin=dict(t=0, l=0, r=0, b=0), height=600, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    # --- تحليل السهم العميق (الميزة الجديدة: Heatmap تاريخي) ---
    elif selected_tab == "تحليل السهم العميق":
        col_sel, _ = st.columns([1, 3])
        with col_sel:
            sel_stock = st.selectbox("اختر السهم للتحليل الشامل:", df['Name'].unique())
        
        if sel_stock:
            hist = st.session_state['history'][sel_stock]
            
            # 1. الشارت الفني (TradingView Style)
            st.subheader("📈 التحليل الفني")
            fig_main = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.0, row_heights=[0.8, 0.2], specs=[[{"secondary_y": True}], [{"secondary_y": False}]])
            
            # الشموع
            fig_main.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name='Price', increasing_line_color='#089981', decreasing_line_color='#f23645'), row=1, col=1)
            # المتوسطات
            fig_main.add_trace(go.Scatter(x=hist.index, y=hist['EMA'], line=dict(color='#2962ff', width=1.5), name='EMA 20'), row=1, col=1)
            # الصندوق
            box_res = check_bullish_box(hist, hist['ATR'])
            if box_res:
                latest = box_res[-1]
                fig_main.add_shape(type="rect", x0=hist.index[-latest['Days_Ago']], x1=hist.index[-1], y0=latest['Box_Bottom'], y1=latest['Box_Top'], line=dict(color="rgba(8, 153, 129, 0.4)", width=1), fillcolor="rgba(8, 153, 129, 0.1)", row=1, col=1)
            # الفوليوم
            colors_vol = ['rgba(8, 153, 129, 0.5)' if c >= o else 'rgba(242, 54, 69, 0.5)' for c, o in zip(hist['Close'], hist['Open'])]
            fig_main.add_trace(go.Bar(x=hist.index, y=hist['Volume'], marker_color=colors_vol, showlegend=False), row=1, col=1, secondary_y=True)
            # RSI
            fig_main.add_trace(go.Scatter(x=hist.index, y=hist['RSI'], line=dict(color='#b2b5be', width=1.5), name='RSI'), row=2, col=1)
            fig_main.add_hline(y=70, line_dash="dot", line_color="#f23645", row=2, col=1); fig_main.add_hline(y=30, line_dash="dot", line_color="#089981", row=2, col=1)
            
            fig_main.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False, paper_bgcolor='#131722', plot_bgcolor='#131722', margin=dict(l=0, r=50, t=10, b=0))
            st.plotly_chart(fig_main, use_container_width=True)

            # 2. الخريطة الحرارية التاريخية (Calendar Heatmap) - الميزة الجديدة 🗓️
            st.divider()
            st.subheader(f"📅 الأداء التاريخي الشهري لـ {sel_stock}")
            
            # تجهيز البيانات (نحتاج بيانات 5 سنوات)
            # بما أننا سحبنا سنة واحدة، سنعرض السنة الحالية، وللمستقبل يمكن زيادة المدة
            monthly_ret = hist['Close'].resample('ME').last().pct_change() * 100
            monthly_ret = monthly_ret.dropna()
            
            if not monthly_ret.empty:
                # تشكيل الجدول (السنة صفوف، الشهور أعمدة)
                years = monthly_ret.index.year.unique()
                months = list(calendar.month_abbr)[1:] # Jan, Feb...
                
                # مصفوفة البيانات
                heatmap_data = []
                for y in years:
                    year_data = []
                    for m in range(1, 13):
                        try:
                            # محاولة العثور على عائد هذا الشهر في هذه السنة
                            val = monthly_ret[(monthly_ret.index.year == y) & (monthly_ret.index.month == m)].values
                            year_data.append(val[0] if len(val) > 0 else 0)
                        except:
                            year_data.append(0)
                    heatmap_data.append(year_data)
                
                # رسم الهيت ماب
                fig_cal = go.Figure(data=go.Heatmap(
                    z=heatmap_data,
                    x=months,
                    y=years,
                    colorscale='RdYlGn', # أحمر للأصفر للأخضر
                    zmid=0, # الصفر هو الوسط
                    texttemplate="%{z:.1f}%", # إظهار النسبة
                    textfont={"size": 12},
                    xgap=2, ygap=2 # فواصل بين المربعات
                ))
                
                fig_cal.update_layout(
                    template="plotly_dark",
                    height=300 + (len(years)*30), # ارتفاع ديناميكي
                    paper_bgcolor='#131722', plot_bgcolor='#131722',
                    title="نسبة التغير الشهري (%)",
                    xaxis_side="top"
                )
                st.plotly_chart(fig_cal, use_container_width=True)
            else:
                st.warning("لا توجد بيانات تاريخية كافية لرسم الخريطة الشهرية.")

else:
    st.info("👋 جاهز! اضغط الزر الأزرق للتحديث.")
