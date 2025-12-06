import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from streamlit_option_menu import option_menu
import calendar

# --- استيراد البيانات ---
try:
    from data.saudi_tickers import STOCKS_DB
except ImportError:
    st.error("🚨 ملف البيانات مفقود.")
    st.stop()

TICKERS = {item['symbol']: item['name'] for item in STOCKS_DB}
SECTORS = {item['name']: item['sector'] for item in STOCKS_DB}

# --- 1. إعداد الصفحة والستايل ---
st.set_page_config(page_title="TASI Pro V8", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif; }
    
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    
    /* إصلاح الجداول والأخطاء */
    .stDataFrame { border: 1px solid #30333d; }
    div[data-testid="stDataFrame"] div[class*="css"] { background-color: #161b24; color: white; }
    
    /* إصلاح البطاقات البيضاء */
    div[data-testid="stMetric"] {
        background-color: #1d212b !important;
        border: 1px solid #464b5f !important;
        padding: 15px !important;
        border-radius: 10px !important;
    }
    [data-testid="stMetricValue"] { color: #ffffff !important; }
    [data-testid="stMetricLabel"] { color: #a3a8b8 !important; }
    
    /* الأزرار */
    div.stButton > button {
        background: linear-gradient(90deg, #2962ff, #0d47a1);
        color: white; border: none; padding: 10px 20px;
        border-radius: 8px; font-weight: bold; width: 100%;
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
    options=["الرئيسية", "الماسح الذكي", "الخريطة الحرارية", "تحليل السهم العميق"],
    icons=["house", "cpu", "grid", "graph-up-arrow"],
    default_index=0,
    orientation="horizontal",
    styles={"container": {"background-color": "transparent"}, "nav-link-selected": {"background-color": "#2962ff"}}
)

# --- 3. الإعدادات ---
with st.sidebar:
    st.header("⚙️ الإعدادات")
    RSI_PERIOD = st.number_input("RSI Period", 14, 30, 24)
    EMA_PERIOD = st.number_input("EMA Trend", 10, 200, 20)
    ATR_MULT = st.number_input("ATR Mult", 1.0, 3.0, 1.5)
    BOX_LOOKBACK = st.slider("Box History", 10, 100, 25)

# --- 4. الدوال الفنية ---
def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    return ranges.max(axis=1).ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def check_bullish_box(df, atr_series):
    in_series = False; is_bullish = False; start_open = 0.0; end_close = 0.0; start_idx = 0; found_boxes = []
    prices = df.iloc[-100:].reset_index(); atrs = atr_series.iloc[-100:].values
    
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
    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    df['Trend_Score'] = ((df['Close'] > df['EMA']).astype(int) + (df['Close'] > df['EMA50']).astype(int))
    return df

# --- 5. المنطق (Engine) ---
if 'data' not in st.session_state: st.session_state['data'] = []
if 'history' not in st.session_state: st.session_state['history'] = {}

# زر التحديث (تم تأمينه لمنع الأخطاء)
c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    if st.button("🚀 تحديث البيانات (Live Scan)"):
        st.session_state['data'] = []
        st.session_state['history'] = {}
        progress = st.progress(0)
        status = st.empty()
        tickers_list = list(TICKERS.keys())
        chunk_size = 25
        
        for i in range(0, len(tickers_list), chunk_size):
            chunk = tickers_list[i:i + chunk_size]
            status.markdown(f"**📡 جاري المعالجة... ({int((i/len(tickers_list))*100)}%)**")
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
                                    
                                    # تخزين التاريخ
                                    st.session_state['history'][name] = df
                                    
                                    # الصناديق
                                    boxes = check_bullish_box(df, df['ATR'])
                                    ai_score = 0; box_status = "لا يوجد"
                                    if boxes:
                                        latest = boxes[-1]
                                        if last['Close'] >= latest['Box_Bottom']:
                                            # حساب تقييم بسيط
                                            mid = (latest['Box_Top'] + latest['Box_Bottom']) / 2
                                            if last['Close'] > mid: ai_score += 50
                                            if last['RSI'] > 50: ai_score += 25
                                            if last['RVOL'] > 1: ai_score += 25
                                            box_status = "نشط"

                                    # إضافة للجدول (مفاتيح إنجليزية لتجنب KeyError)
                                    st.session_state['data'].append({
                                        "Name": name, "Symbol": sym, "Sector": SECTORS.get(name, "عام"),
                                        "Price": last['Close'], "Change": last['Change'], 
                                        "RSI": last['RSI'], "Trend": last['Trend_Score'],
                                        "RVOL": last['RVOL'], "AI_Score": ai_score,
                                        "Box_Status": box_status, "TV": link
                                    })
                        except: continue
            except Exception as e: print(f"Error chunk: {e}")
            progress.progress(min((i + chunk_size) / len(tickers_list), 1.0))
        progress.empty()
        status.success("✅ اكتمل التحديث!")

# --- 6. العرض (Dashboard) ---
if st.session_state['data']:
    df = pd.DataFrame(st.session_state['data'])
    link_col = st.column_config.LinkColumn("شارت", display_text="TV")

    # --- الرئيسية ---
    if selected_tab == "الرئيسية":
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("الشركات", len(df))
        # استخدام المفتاح الصحيح 'Change' لحل KeyError
        k2.metric("السوق أخضر", len(df[df['Change'] > 0]))
        k3.metric("صناديق ذهبية", len(df[df['AI_Score'] >= 75]))
        k4.metric("سيولة عالية", len(df[df['RVOL'] > 2.0]))
        
        st.markdown("### 📋 ملخص السوق")
        st.dataframe(
            df[['Name', 'Price', 'Change', 'RSI', 'Trend', 'RVOL', 'TV']].style
            .format({"Price": "{:.2f}", "Change": "{:.2f}%", "RSI": "{:.1f}", "RVOL": "{:.1f}x"})
            .background_gradient(cmap='RdYlGn', subset=['Change']),
            column_config={"TV": link_col}, use_container_width=True
        )

    # --- الماسح الذكي ---
    elif selected_tab == "الماسح الذكي":
        st.markdown("### 📦 الفرص الذكية")
        min_score = st.slider("فلتر التقييم", 0, 100, 50)
        filtered = df[(df['AI_Score'] >= min_score) & (df['Box_Status'] == "نشط")]
        
        if not filtered.empty:
            st.dataframe(
                filtered[['Name', 'Price', 'AI_Score', 'Trend', 'TV']].sort_values('AI_Score', ascending=False)
                .style.format({"Price": "{:.2f}", "AI_Score": "{:.0f}"})
                .background_gradient(cmap='Greens', subset=['AI_Score']),
                column_config={"TV": link_col}, use_container_width=True
            )
        else: st.info("لا توجد فرص حالياً.")

    # --- الخريطة الحرارية ---
    elif selected_tab == "الخريطة الحرارية":
        fig = px.treemap(
            df, path=[px.Constant("TASI"), 'Sector', 'Name'], values='Price',
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
    st.info("👋 جاهز. اضغط الزر الأزرق للتحديث.")
