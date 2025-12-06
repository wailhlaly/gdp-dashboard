import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from streamlit_option_menu import option_menu
from scipy.signal import argrelextrema

# --- استيراد البيانات ---
try:
    from data.saudi_tickers import STOCKS_DB
except ImportError:
    st.error("🚨 ملف البيانات مفقود.")
    st.stop()

TICKERS = {item['symbol']: item['name'] for item in STOCKS_DB}
SECTORS = {item['name']: item['sector'] for item in STOCKS_DB}

# --- 1. إعداد الصفحة والستايل (CSS الاحترافي) ---
st.set_page_config(page_title="TASI.AI Pro", layout="wide", initial_sidebar_state="collapsed")

# حقن CSS لتغيير شكل الموقع جذرياً
st.markdown("""
<style>
    /* استيراد خط عصري */
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Cairo', sans-serif;
    }
    
    /* الخلفية العامة: لون كحلي عميق جداً */
    .stApp {
        background-color: #0e1117;
        background-image: radial-gradient(circle at 50% 0%, #1c202b 0%, #0e1117 70%);
    }
    
    /* إخفاء الهيدر الافتراضي */
    header {visibility: hidden;}
    
    /* تصميم البطاقات (Metrics) - تأثير الزجاج */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        border-color: #2962ff;
        background: rgba(41, 98, 255, 0.1);
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 28px !important;
        font-weight: 800 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #a0aab9 !important;
        font-size: 14px !important;
    }

    /* الأزرار بتدرج لوني */
    div.stButton > button {
        background: linear-gradient(135deg, #2962ff 0%, #0039cb 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 16px;
        box-shadow: 0 4px 15px rgba(41, 98, 255, 0.4);
        transition: all 0.3s;
        width: 100%;
    }
    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(41, 98, 255, 0.6);
    }

    /* تحسين الجداول */
    .stDataFrame {
        border: 1px solid #2d3748;
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* القوائم المنسدلة */
    .stSelectbox > div > div {
        background-color: #1a1d26;
        color: white;
        border-radius: 10px;
        border: 1px solid #2d3748;
    }
    
    /* العناوين */
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. القائمة العلوية (Navigation Bar) ---
selected_tab = option_menu(
    menu_title=None,
    options=["الرئيسية", "الماسح الذكي", "الخريطة الحرارية", "الشارت المتقدم"],
    icons=["speedometer2", "cpu-fill", "grid-fill", "graph-up-arrow"],
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "transparent"},
        "icon": {"color": "#2962ff", "font-size": "18px"}, 
        "nav-link": {"font-size": "16px", "text-align": "center", "margin": "5px", "--hover-color": "#1a1d26", "color": "#e0e0e0"},
        "nav-link-selected": {"background-color": "#2962ff", "border-radius": "10px", "font-weight": "bold"},
    }
)

# --- 3. الإعدادات (في قائمة جانبية منبثقة) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/5373/5373324.png", width=80)
    st.markdown("### ⚙️ إعدادات الخوارزمية")
    RSI_PERIOD = st.number_input("RSI Period", 14, 30, 24)
    EMA_PERIOD = st.number_input("EMA Trend", 10, 200, 20)
    st.divider()
    ATR_MULT = st.number_input("ATR Multiplier", 1.0, 3.0, 1.5)
    BOX_LOOKBACK = st.slider("نطاق البحث (أيام)", 10, 100, 25)
    st.info("تم تطوير هذا النظام بواسطة Gemini AI 🤖")

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
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean() # للإستراتيجيات طويلة المدى
    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Trend Score (0-3)
    df['Trend_Score'] = (
        (df['Close'] > df['EMA']).astype(int) + 
        (df['Close'] > df['EMA50']).astype(int) +
        (df['Close'] > df['EMA200']).astype(int)
    )
    return df

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

def calculate_ai_score(last, box):
    score = 0
    mid = (box['Box_Top'] + box['Box_Bottom']) / 2
    if last['Close'] > mid: score += 25
    if last['Close'] > last['EMA']: score += 25
    if last['RSI'] > 50: score += 20
    if last['MACD'] > last['Signal']: score += 20
    if last['RVOL'] > 1.2: score += 10
    return min(score, 100)

def check_divergence(df, order=5):
    price_lows = argrelextrema(df['Low'].values, np.less_equal, order=order)[0]
    rsi_lows = argrelextrema(df['RSI'].values, np.less_equal, order=order)[0]
    if len(price_lows) >= 2 and len(rsi_lows) >= 2:
        p_last = price_lows[-1]; p_prev = price_lows[-2]
        if (len(df) - p_last) <= 15:
            if df['Low'].iloc[p_last] <= df['Low'].iloc[p_prev] and df['RSI'].iloc[p_last] > df['RSI'].iloc[p_prev]:
                return "إيجابي ✅"
    return "لا"

# --- 5. المنطق (Cache) ---
if 'data' not in st.session_state: st.session_state['data'] = []
if 'history' not in st.session_state: st.session_state['history'] = {}

# زر التحديث
c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    if st.button("🚀 بدء المسح الذكي للسوق (Start Scan)"):
        st.session_state['data'] = []
        st.session_state['history'] = {}
        progress = st.progress(0)
        status = st.empty()
        tickers_list = list(TICKERS.keys())
        chunk_size = 30
        
        for i in range(0, len(tickers_list), chunk_size):
            chunk = tickers_list[i:i + chunk_size]
            status.markdown(f"**📡 جاري تحليل البيانات... ({int((i/len(tickers_list))*100)}%)**")
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
                                    
                                    # Logic
                                    boxes = check_bullish_box(df, df['ATR'])
                                    ai_score = 0; box_status = "---"
                                    if boxes:
                                        latest = boxes[-1]
                                        if last['Close'] >= latest['Box_Bottom']:
                                            ai_score = calculate_ai_score(last, latest)
                                            box_status = "Active"
                                    
                                    div_status = check_divergence(df)

                                    st.session_state['data'].append({
                                        "Name": name, "Symbol": sym, "Sector": SECTORS.get(name, "عام"),
                                        "Price": last['Close'], "Change": last['Change'], 
                                        "RSI": last['RSI'], "Trend": last['Trend_Score'],
                                        "RVOL": last['RVOL'], "AI_Score": ai_score,
                                        "Box_Status": box_status, "Divergence": div_status, "TV": link
                                    })
                        except: continue
            except: pass
            progress.progress(min((i + chunk_size) / len(tickers_list), 1.0))
        progress.empty()
        status.success("✅ اكتمل التحليل!")

# --- 6. العرض (UI) ---
if st.session_state['data']:
    df = pd.DataFrame(st.session_state['data'])
    link_col = st.column_config.LinkColumn("Chart", display_text="Open TV")

    # --- الصفحة الرئيسية ---
    if selected_tab == "الرئيسية":
        # KPIs بتصميم جديد
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("عدد الشركات", len(df))
        col2.metric("السوق أخضر", len(df[df['Change'] > 0]))
        col3.metric("صناديق ذهبية", len(df[df['AI_Score'] >= 80]))
        col4.metric("انفراجات إيجابية", len(df[df['Divergence'] == "إيجابي ✅"]))
        
        st.markdown("### 📋 ملخص السوق المباشر")
        st.dataframe(
            df[['Name', 'Price', 'Change', 'RSI', 'Trend', 'RVOL', 'TV']].style
            .format({"Price": "{:.2f}", "Change": "{:.2f}%", "RSI": "{:.1f}", "RVOL": "{:.1f}x"})
            .background_gradient(cmap='RdYlGn', subset=['Change']),
            column_config={"TV": link_col, "Trend": st.column_config.ProgressColumn("قوة الترند", min_value=0, max_value=3, format="%d/3")},
            use_container_width=True, height=600
        )

    # --- الماسح الذكي ---
    elif selected_tab == "الماسح الذكي":
        st.markdown("### 🧠 الفرص الذكية (AI Opportunities)")
        c1, c2 = st.columns([1, 3])
        with c1:
            min_ai = st.slider("فلتر الذكاء (AI Score)", 0, 100, 70)
            show_div = st.checkbox("عرض الانفراجات فقط", False)
        
        filtered = df[df['AI_Score'] >= min_ai]
        if show_div: filtered = filtered[filtered['Divergence'] == "إيجابي ✅"]
        
        if not filtered.empty:
            st.dataframe(
                filtered[['Name', 'Price', 'AI_Score', 'Divergence', 'RVOL', 'TV']].sort_values('AI_Score', ascending=False)
                .style.format({"Price": "{:.2f}", "AI_Score": "{:.0f}", "RVOL": "{:.1f}x"})
                .background_gradient(cmap='Greens', subset=['AI_Score']),
                column_config={"TV": link_col}, use_container_width=True
            )
        else: st.warning("لا توجد شركات تطابق شروط الفلتر الحالية.")

    # --- الخريطة الحرارية ---
    elif selected_tab == "الخريطة الحرارية":
        st.markdown("### 🗺️ خريطة السيولة")
        fig = px.treemap(
            df, path=[px.Constant("TASI"), 'Sector', 'Name'], values='Price',
            color='Change', color_continuous_scale='RdYlGn', color_continuous_midpoint=0,
            custom_data=['Symbol', 'Price', 'Change', 'RSI']
        )
        fig.update_traces(
            hovertemplate="<b>%{label}</b><br>السعر: %{customdata[1]:.2f}<br>التغير: %{customdata[2]:.2f}%<br>RSI: %{customdata[3]:.1f}",
            textinfo="label+text+value", textfont_size=14
        )
        fig.update_layout(margin=dict(t=0, l=0, r=0, b=0), height=650, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    # --- الشارت المتقدم ---
    elif selected_tab == "الشارت المتقدم":
        c_sel, _ = st.columns([1, 3])
        with c_sel:
            sel_stock = st.selectbox("اختر السهم:", df['Name'].unique())
        
        if sel_stock:
            hist = st.session_state['history'][sel_stock]
            
            # تخطيط TradingView
            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.0, 
                row_heights=[0.8, 0.2], specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
            )

            # الشموع
            fig.add_trace(go.Candlestick(
                x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'],
                name='Price', increasing_line_color='#089981', decreasing_line_color='#f23645',
                increasing_fillcolor='#089981', decreasing_fillcolor='#f23645'
            ), row=1, col=1, secondary_y=False)

            # المتوسطات
            fig.add_trace(go.Scatter(x=hist.index, y=hist['EMA'], line=dict(color='#2962ff', width=1.5), name=f'EMA {EMA_PERIOD}'), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['EMA50'], line=dict(color='#ff9800', width=1.5), name='EMA 50'), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['EMA200'], line=dict(color='#9c27b0', width=2), name='EMA 200'), row=1, col=1)

            # الصناديق
            box_res = check_bullish_box(hist, hist['ATR'])
            if box_res:
                latest = box_res[-1]
                mid = (latest['Box_Top'] + latest['Box_Bottom']) / 2
                fig.add_shape(type="rect", x0=hist.index[-latest['Days_Ago']], x1=hist.index[-1], y0=latest['Box_Bottom'], y1=latest['Box_Top'],
                              line=dict(color="rgba(8, 153, 129, 0.4)", width=1), fillcolor="rgba(8, 153, 129, 0.1)", row=1, col=1)
                fig.add_shape(type="line", x0=hist.index[-latest['Days_Ago']], x1=hist.index[-1], y0=mid, y1=mid,
                              line=dict(color="#2962ff", width=1, dash="dot"), row=1, col=1)

            # الفوليوم
            colors_vol = ['rgba(8, 153, 129, 0.5)' if c >= o else 'rgba(242, 54, 69, 0.5)' for c, o in zip(hist['Close'], hist['Open'])]
            fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], marker_color=colors_vol, name='Volume', showlegend=False), row=1, col=1, secondary_y=True)

            # RSI
            fig.add_trace(go.Scatter(x=hist.index, y=hist['RSI'], line=dict(color='#b2b5be', width=1.5), name='RSI'), row=2, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="#f23645", row=2, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="#089981", row=2, col=1)

            # التنسيق النهائي
            fig.update_layout(
                template="plotly_dark", height=650, 
                paper_bgcolor='#131722', plot_bgcolor='#131722',
                margin=dict(l=0, r=60, t=20, b=0),
                xaxis_rangeslider_visible=False,
                legend=dict(orientation="h", y=1, x=0, bgcolor='rgba(0,0,0,0)'),
                hovermode='x unified', dragmode='pan',
                xaxis=dict(showgrid=False, color='#787b86'),
                yaxis=dict(showgrid=True, gridcolor='#2a2e39', side='right'),
                yaxis2=dict(showgrid=False, showticklabels=False),
                yaxis3=dict(showgrid=False, range=[0, 100], side='right')
            )
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

else:
    st.info("👋 اضغط الزر الأزرق لبدء التحليل.")
