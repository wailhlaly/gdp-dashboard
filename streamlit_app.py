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
    try:
        from saudi_tickers import STOCKS_DB
    except ImportError:
        st.error("🚨 ملف البيانات (saudi_tickers.py) مفقود.")
        st.stop()

TICKERS = {item['symbol']: item['name'] for item in STOCKS_DB}
SECTORS = {item['name']: item['sector'] for item in STOCKS_DB}

# --- 1. إعداد الصفحة والستايل ---
st.set_page_config(page_title="TASI Pro TV-Style", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif; }
    
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    .stDataFrame { border: 1px solid #30333d; }
    div[data-testid="stDataFrame"] div[class*="css"] { background-color: #161b24; color: white; }
    
    div[data-testid="stMetric"] {
        background: rgba(30, 34, 45, 0.6); backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px;
        padding: 15px; transition: 0.3s;
    }
    div[data-testid="stMetricValue"] { color: #ffffff !important; }
    
    .stButton > button {
        background: linear-gradient(90deg, #2962ff, #2979ff); color: white; border: none;
        border-radius: 8px; padding: 0.6rem 1.2rem; font-weight: bold; width: 100%;
        box-shadow: 0 4px 15px rgba(41, 98, 255, 0.3);
    }
    .stButton > button:hover { transform: scale(1.02); }
    
    .stSelectbox > div > div { background-color: #1e222d; color: white; border: 1px solid #434651; }
</style>
""", unsafe_allow_html=True)

# --- 2. القائمة العلوية ---
selected_tab = option_menu(
    menu_title=None,
    options=["الرئيسية", "الماسح الذكي", "كاشف الانفراجات", "الخريطة", "الشارت"],
    icons=["house", "cpu", "eye", "grid", "graph-up"],
    default_index=4, # جعلنا الشارت هو الافتراضي للتجربة السريعة
    orientation="horizontal",
    styles={
        "container": {"background-color": "transparent", "padding": "0!important"},
        "nav-link-selected": {"background-color": "#2962ff"}
    }
)

# --- 3. الإعدادات ---
with st.sidebar:
    st.header("⚙️ إعدادات المحلل")
    RSI_PERIOD = st.number_input("RSI Period", 14, 30, 24)
    EMA_PERIOD = st.number_input("EMA Trend", 10, 200, 20)
    st.divider()
    ATR_MULT = st.number_input("ATR Multiplier", 1.0, 3.0, 1.5)
    BOX_LOOKBACK = st.slider("Box History (Days)", 10, 100, 25)

# --- 4. الدوال الفنية ---
def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    return ranges.max(axis=1).ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def check_divergence(df, order=5):
    price_lows = argrelextrema(df['Low'].values, np.less_equal, order=order)[0]
    rsi_lows = argrelextrema(df['RSI'].values, np.less_equal, order=order)[0]
    divergence = "لا يوجد"
    if len(price_lows) >= 2 and len(rsi_lows) >= 2:
        p_last = price_lows[-1]; p_prev = price_lows[-2]
        if (len(df) - p_last) <= 10:
            price_low1 = df['Low'].iloc[p_prev]; price_low2 = df['Low'].iloc[p_last]
            rsi_low1 = df['RSI'].iloc[p_prev]; rsi_low2 = df['RSI'].iloc[p_last]
            if price_low2 <= price_low1 and rsi_low2 > rsi_low1:
                divergence = "إيجابي قوي 🔥"
    return divergence

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
                            "Days_Ago": periods_ago
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
    
    score = ((df['Close'] > df['EMA']).astype(int) + (df['Close'] > df['EMA50']).astype(int))
    df['Trend_Score'] = score
    return df

def calculate_ai_score(last, box):
    score = 0
    mid = (box['Box_Top'] + box['Box_Bottom']) / 2
    if last['Close'] > mid: score += 25
    if last['Close'] > last['EMA']: score += 25
    if last['RSI'] > 50: score += 20
    if last['MACD'] > last['Signal']: score += 20
    if last['RVOL'] > 1.2: score += 10
    return min(score, 100)

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
                                
                                boxes = check_bullish_box(df, df['ATR'])
                                ai_score = 0; box_status = "لا يوجد"; box_age = 0
                                if boxes:
                                    latest = boxes[-1]
                                    box_age = latest['Days_Ago']
                                    if last['Close'] >= latest['Box_Bottom']:
                                        ai_score = calculate_ai_score(last, latest)
                                        box_status = "داخل الصندوق" if last['Close'] <= latest['Box_Top'] else "اختراق"
                                
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
        except: pass
        progress.progress(min((i + chunk_size) / len(tickers_list), 1.0))
    progress.empty()
    status.success("✅ تم التحديث!")

# --- 6. العرض (Dashboard) ---
if st.session_state['data']:
    df = pd.DataFrame(st.session_state['data'])
    link_col = st.column_config.LinkColumn("شارت", display_text="Open TV")

    if selected_tab == "الرئيسية":
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("عدد الشركات", len(df))
        k2.metric("انفراجات", len(df[df['Divergence'] != "لا يوجد"]))
        k3.metric("صناديق ذهبية", len(df[df['AI_Score'] >= 70]))
        k4.metric("سيولة عالية", len(df[df['RVOL'] > 2.0]))
        
        st.markdown("### 📊 ملخص السوق")
        st.dataframe(
            df[['Name', 'Price', 'Change', 'RSI', 'RVOL', 'Divergence', 'TV']].style
            .format({"Price": "{:.2f}", "Change": "{:.2f}%", "RSI": "{:.1f}", "RVOL": "{:.1f}x"})
            .background_gradient(cmap='RdYlGn', subset=['Change'])
            .applymap(lambda v: 'color: #00e676; font-weight: bold;' if "إيجابي" in str(v) else '', subset=['Divergence']),
            column_config={"TV": link_col}, use_container_width=True
        )

    elif selected_tab == "الماسح الذكي":
        st.markdown("### 📦 الصناديق الذكية")
        score_filter = st.slider("تقييم الذكاء", 0, 100, 60)
        filtered = df[(df['AI_Score'] >= score_filter) & (df['Box_Status'] != "لا يوجد")]
        st.dataframe(
            filtered[['Name', 'Price', 'AI_Score', 'Box_Status', 'Box_Age', 'TV']].sort_values('AI_Score', ascending=False)
            .style.format({"Price": "{:.2f}", "AI_Score": "{:.0f}"}).background_gradient(cmap='Greens', subset=['AI_Score']),
            column_config={"TV": link_col}, use_container_width=True
        )

    elif selected_tab == "كاشف الانفراجات":
        st.markdown("### 🦅 فرص الانفراج (RSI Divergence)")
        div_df = df[df['Divergence'] != "لا يوجد"]
        st.dataframe(
            div_df[['Name', 'Price', 'RSI', 'Divergence', 'TV']].style.format({"Price": "{:.2f}", "RSI": "{:.1f}"})
            .applymap(lambda v: 'background-color: #1b5e20; color: white;', subset=['Divergence']),
            column_config={"TV": link_col}, use_container_width=True
        )

    elif selected_tab == "الخريطة":
        fig = px.treemap(
            df, path=[px.Constant("السوق"), 'Sector', 'Name'], values='Price',
            color='Change', color_continuous_scale='RdYlGn', color_continuous_midpoint=0,
            custom_data=['Symbol', 'Price', 'Change']
        )
        fig.update_traces(hovertemplate="<b>%{label}</b><br>السعر: %{customdata[1]:.2f}<br>التغير: %{customdata[2]:.2f}%")
        fig.update_layout(margin=dict(t=0, l=0, r=0, b=0), height=600, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    # --- الشارت الاحترافي (TradingView Logic) ---
    elif selected_tab == "الشارت":
        c_sel, _ = st.columns([1, 3])
        with c_sel:
            sel_stock = st.selectbox("اختر السهم:", df['Name'].unique())
        
        if sel_stock:
            hist = st.session_state['history'][sel_stock]
            
            # 1. إعداد Layout متقدم (Shared Axis)
            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.8, 0.2],
                specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
            )

            # 2. الشموع (Candles)
            fig.add_trace(go.Candlestick(
                x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'],
                name='Price',
                increasing_line_color='#089981', decreasing_line_color='#f23645',
                increasing_fillcolor='#089981', decreasing_fillcolor='#f23645'
            ), row=1, col=1, secondary_y=False)

            # 3. المتوسطات (EMAs)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['EMA'], line=dict(color='#2962ff', width=1.5), name=f'EMA {EMA_PERIOD}'), row=1, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['EMA50'], line=dict(color='#ff9800', width=1.5), name='EMA 50'), row=1, col=1, secondary_y=False)

            # 4. الصناديق (Transparent Boxes)
            box_res = check_bullish_box(hist, hist['ATR'])
            if box_res:
                latest = box_res[-1]
                mid = (latest['Box_Top'] + latest['Box_Bottom']) / 2
                # رسم المستطيل المظلل
                fig.add_shape(
                    type="rect",
                    x0=hist.index[-latest['Days_Ago']], x1=hist.index[-1],
                    y0=latest['Box_Bottom'], y1=latest['Box_Top'],
                    line=dict(color="rgba(8, 153, 129, 0.4)", width=1),
                    fillcolor="rgba(8, 153, 129, 0.1)",
                    row=1, col=1, secondary_y=False
                )
                # رسم خط المنتصف
                fig.add_shape(type="line", x0=hist.index[-latest['Days_Ago']], x1=hist.index[-1], y0=mid, y1=mid,
                              line=dict(color="#2962ff", width=1, dash="dot"), row=1, col=1, secondary_y=False)

            # 5. الفوليوم (مدمج في الخلفية)
            colors_vol = ['rgba(8, 153, 129, 0.5)' if c >= o else 'rgba(242, 54, 69, 0.5)' for c, o in zip(hist['Close'], hist['Open'])]
            fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], marker_color=colors_vol, name='Volume', showlegend=False), 
                          row=1, col=1, secondary_y=True)

            # 6. مؤشر RSI (في الأسفل)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['RSI'], line=dict(color='#b2b5be', width=1.5), name='RSI'), row=2, col=1)
            fig.add_shape(type="rect", x0=hist.index[0], x1=hist.index[-1], y0=30, y1=70, fillcolor="rgba(255, 255, 255, 0.05)", line_width=0, layer="below", row=2, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="#f23645", row=2, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="#089981", row=2, col=1)

            # --- 7. التنسيق السحري (TradingView Feel) ---
            fig.update_layout(
                template="plotly_dark", height=650, 
                dragmode='pan', # التفعيل الافتراضي للسحب
                hovermode='x unified', # المؤشر الموحد
                xaxis_rangeslider_visible=False,
                paper_bgcolor='#131722', plot_bgcolor='#131722',
                margin=dict(l=0, r=50, t=30, b=0), # ترك مساحة لليمين لمحور السعر
                legend=dict(orientation="h", y=1, x=0, bgcolor='rgba(0,0,0,0)'),
            )
            
            # إعدادات المحاور
            fig.update_xaxes(showgrid=False, gridcolor='#2a2e39')
            # محور السعر (يمين)
            fig.update_yaxes(side='right', showgrid=True, gridcolor='#2a2e39', zeroline=False, row=1, col=1, secondary_y=False)
            # محور الفوليوم (مخفي الأرقام، مصغر)
            fig.update_yaxes(showticklabels=False, range=[0, hist['Volume'].max()*4], row=1, col=1, secondary_y=True)
            # محور RSI
            fig.update_yaxes(side='right', showgrid=False, range=[0, 100], row=2, col=1)

            # تفعيل إعدادات الكونفيق (Config) المهمة جداً
            st.plotly_chart(fig, use_container_width=True, config={
                'scrollZoom': True, # التكبير بالعجلة
                'displayModeBar': True,
                'modeBarButtonsToRemove': ['zoomIn2d', 'zoomOut2d', 'autoScale2d'],
                'displaylogo': False
            })

else:
    st.info("👋 جاهز! اضغط زر التحديث.")
