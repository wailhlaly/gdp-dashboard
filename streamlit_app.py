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
st.set_page_config(page_title="Saudi Pro AI", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    .stDataFrame { border: 1px solid #30333d; }
    div[data-testid="stDataFrame"] div[class*="css"] { background-color: #161b24; color: white; }
    div[data-testid="stMetric"] { background-color: #1d212b !important; border: 1px solid #30333d; padding: 10px; border-radius: 8px; }
    div[data-testid="stMetricValue"] { color: #ffffff !important; }
    div.stButton > button { background: linear-gradient(90deg, #00c853, #64dd17); color: white; border: none; width: 100%; font-weight: bold; padding: 12px; border-radius: 8px; font-size: 16px; }
    div.stButton > button:hover { background: linear-gradient(90deg, #009624, #00c853); }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: #1d212b; color: #e0e0e0; border-radius: 4px; border: 1px solid #333; }
    .stTabs [aria-selected="true"] { background-color: #00c853 !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# --- 2. الإعدادات ---
with st.sidebar:
    st.header("⚙️ المحرك الذكي")
    RSI_PERIOD = st.number_input("RSI Period", value=24)
    EMA_PERIOD = st.number_input("EMA Period", value=20) # غيرناه لـ 20 ليكون أقوى كترند
    st.divider()
    ATR_MULT = st.number_input("ATR Mult", value=1.5)
    BOX_LOOKBACK = st.slider("Box Age (Days)", 5, 60, 25)

# --- 3. الدوال الفنية (المطورة) ---
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
    df['RVOL'] = df['Volume'] / df['Vol_Avg'] # سيولة اليوم
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/24, min_periods=24, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/24, min_periods=24, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # EMA & MACD
    df['EMA'] = df['Close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Trend Score (للمتوسطات)
    df['EMA8'] = df['Close'].ewm(span=8, adjust=False).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA40'] = df['Close'].ewm(span=40, adjust=False).mean()
    df['EMA86'] = df['Close'].ewm(span=86, adjust=False).mean()
    
    df['Trend_Score'] = (
        (df['Close'] > df['EMA8']).astype(int) + 
        (df['Close'] > df['EMA20']).astype(int) + 
        (df['Close'] > df['EMA40']).astype(int) + 
        (df['Close'] > df['EMA86']).astype(int)
    )
    return df

# 🔥 دالة كشف الصناديق (المطورة لتحليل السيولة الداخلية)
def check_bullish_box_advanced(df):
    in_series = False; is_bullish = False; start_open = 0.0; end_close = 0.0
    start_idx = 0; found_boxes = []
    
    # تحويل للدخول في Loop
    prices = df.iloc[-100:].reset_index() if len(df) > 100 else df.reset_index()
    atrs = df['ATR'].iloc[-100:].values if len(df) > 100 else df['ATR'].values
    rvols = df['RVOL'].iloc[-100:].values if len(df) > 100 else df['RVOL'].values # نحتاج سيولة كل شمعة
    
    for i in range(len(prices)):
        row = prices.iloc[i]; close = row['Close']; open_p = row['Open']
        is_green = close > open_p; is_red = close < open_p
        current_atr = atrs[i]
        
        if np.isnan(current_atr): continue
        
        if not in_series:
            if is_green:
                in_series = True; is_bullish = True; start_open = open_p; start_idx = i
            elif is_red:
                in_series = True; is_bullish = False; start_open = open_p; start_idx = i
        elif in_series:
            if is_bullish and is_green: end_close = close
            elif not is_bullish and is_red: end_close = close
            elif (is_bullish and is_red) or (not is_bullish and is_green):
                # نهاية السلسلة
                final_close = end_close if end_close != 0 else start_open
                price_move = abs(final_close - start_open)
                
                if price_move >= current_atr * ATR_MULT and is_bullish:
                    days_ago = len(prices) - i
                    if days_ago <= BOX_LOOKBACK:
                        # 🧠 الذكاء: حساب متوسط السيولة للشموع المكونة للصندوق
                        box_rvols = rvols[start_idx:i] # شريحة السيولة للصندوق
                        avg_box_rvol = np.mean(box_rvols) if len(box_rvols) > 0 else 1.0
                        
                        found_boxes.append({
                            "Box_Top": max(start_open, final_close),
                            "Box_Bottom": min(start_open, final_close),
                            "Days_Ago": days_ago,
                            "Box_Avg_RVOL": avg_box_rvol # تخزين جودة سيولة الصندوق
                        })
                
                # إعادة تعيين
                in_series = True; is_bullish = is_green; start_open = open_p; end_close = close; start_idx = i
                
    return found_boxes

# 🧠 دالة حساب التقييم الذكي (AI Score)
def calculate_ai_score(last_row, box_info):
    score = 0
    reasons = []
    
    # 1. جودة سيولة الصندوق (30%)
    if box_info['Box_Avg_RVOL'] >= 1.5:
        score += 30; reasons.append("سيولة صندوق عالية 🔥")
    elif box_info['Box_Avg_RVOL'] >= 1.0:
        score += 15
        
    # 2. موقع السعر (20%)
    mid = (box_info['Box_Top'] + box_info['Box_Bottom']) / 2
    if last_row['Close'] > mid:
        score += 20; reasons.append("فوق المنتصف")
        
    # 3. الاتجاه EMA (20%)
    if last_row['Close'] > last_row['EMA']:
        score += 20; reasons.append("فوق EMA")
        
    # 4. الزخم RSI (15%)
    if last_row['RSI'] > 50:
        score += 15
        
    # 5. الماكد (15%)
    if last_row['MACD'] > last_row['Signal']:
        score += 15; reasons.append("MACD إيجابي")
        
    return score, reasons

# --- 4. التشغيل ---
st.title("🤖 Saudi Pro AI (المحلل الذكي)")

# تهيئة
for k in ['data', 'signals', 'boxes', 'history']:
    if k not in st.session_state: st.session_state[k] = []

if st.button("🚀 تشغيل الذكاء الاصطناعي وتحليل السوق"):
    # تصفير
    st.session_state['data'] = []
    st.session_state['signals'] = []
    st.session_state['boxes'] = []
    st.session_state['history'] = {}
    
    prog = st.progress(0)
    status = st.empty()
    tickers_list = list(TICKERS.keys())
    
    chunk_size = 25
    for i in range(0, len(tickers_list), chunk_size):
        chunk = tickers_list[i:i + chunk_size]
        status.text(f"جاري معالجة البيانات... {i//chunk_size + 1}")
        
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
                                st.session_state['data'].append({
                                    "Name": name, "Symbol": sym, "Sector": SECTORS.get(name, "عام"),
                                    "Price": last['Close'], "Change": last['Change'], 
                                    "RSI": last['RSI'], "Trend_Score": last['Trend_Score'],
                                    "TV": link
                                })
                                
                                # --- تحليل الصناديق الذكي ---
                                boxes = check_bullish_box_advanced(df)
                                if boxes:
                                    latest = boxes[-1]
                                    # شرط: السعر ما زال داخل أو فوق الصندوق (لم يكسره)
                                    if last['Close'] >= latest['Box_Bottom']:
                                        # حساب التقييم
                                        ai_score, ai_reasons = calculate_ai_score(last, latest)
                                        
                                        st.session_state['boxes'].append({
                                            "الاسم": name, "السعر": last['Close'],
                                            "نوع الصندوق": "صاعد 🟢",
                                            "AI Score": ai_score, # النتيجة
                                            "الأسباب": ", ".join(ai_reasons),
                                            "سيولة الصندوق": f"x{latest['Box_Avg_RVOL']:.1f}",
                                            "منذ": latest['Days_Ago'],
                                            "TV": link
                                        })
                                        
                                # Sniper Logic
                                t = df.tail(4)
                                if len(t) == 4:
                                    rsi_x = False; ema_x = False
                                    for x in range(1, 4):
                                        if t['RSI'].iloc[x-1] <= 30 and t['RSI'].iloc[x] > 30: rsi_x = True
                                        if t['Close'].iloc[x-1] <= t['EMA'].iloc[x-1] and t['Close'].iloc[x] > t['EMA'].iloc[x]: ema_x = True
                                    if rsi_x and ema_x:
                                        st.session_state['signals'].append({
                                            "الاسم": name, "السعر": last['Close'], "RSI": last['RSI'], 
                                            "TV": link
                                        })
                    except: continue
        except: pass
        prog.progress(min((i + chunk_size) / len(tickers_list), 1.0))
    
    prog.empty()
    status.success("اكتمل التحليل الذكي!")

# --- 5. العرض ---
if st.session_state['data']:
    df = pd.DataFrame(st.session_state['data'])
    
    # KPI
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("عدد الشركات", len(df))
    # حساب عدد الفرص القوية (AI > 70)
    high_quality_boxes = len([b for b in st.session_state['boxes'] if b['AI Score'] >= 70])
    c2.metric("صناديق ذهبية (Score > 70)", high_quality_boxes)
    c3.metric("إشارات القناص", len(st.session_state['signals']))
    c4.metric("ترند قوي", len(df[df['Trend_Score'] == 4]))
    
    st.divider()
    
    tabs = st.tabs(["💎 الصناديق الذكية (AI)", "🎯 القناص", "🗺️ الخريطة", "📋 السوق", "📈 الشارت"])
    link_col = st.column_config.LinkColumn("شارت", display_text="Open TV")
    
    # --- TAB 1: AI BOXES ---
    with tabs[0]:
        if st.session_state['boxes']:
            st.markdown("### 🧠 تقييم الذكاء الاصطناعي للصناديق")
            st.caption("الترتيب يعتمد على الـ AI Score (من 100). الدرجة الأعلى تعني توافق السيولة + الاتجاه + الزخم.")
            
            df_ai = pd.DataFrame(st.session_state['boxes'])
            df_ai = df_ai.sort_values(by="AI Score", ascending=False) # الأفضل في الأعلى
            
            # تلوين النتيجة
            def color_score(val):
                if val >= 80: return 'background-color: #004d40; color: #b2dfdb; font-weight: bold' # ممتاز
                elif val >= 60: return 'color: #69f0ae; font-weight: bold' # جيد
                elif val < 40: return 'color: #ff5252' # ضعيف
                return ''
            
            st.dataframe(
                df_ai.style.format({"السعر": "{:.2f}", "AI Score": "{:.0f}"})
                .map(color_score, subset=['AI Score']),
                column_config={"TV": link_col, "الأسباب": st.column_config.ListColumn("نقاط القوة")},
                use_container_width=True
            )
        else: st.info("لا توجد صناديق نشطة حالياً.")

    # --- TAB 2: Sniper ---
    with tabs[1]:
        if st.session_state['signals']:
            st.dataframe(pd.DataFrame(st.session_state['signals']), column_config={"TV": link_col}, use_container_width=True)
        else: st.info("لا توجد إشارات.")

    # --- TAB 3: Map ---
    with tabs[2]:
        fig_ema = px.treemap(
            df, path=[px.Constant("السوق"), 'Sector', 'Name'], values='Price',
            color='Trend_Score', color_continuous_scale='RdYlGn', range_color=[0, 4],
            custom_data=['Symbol', 'TV', 'Price', 'Name']
        )
        fig_ema.update_traces(hovertemplate="<b>%{label}</b><br>السعر: %{customdata[2]:.2f}<br>الترند: %{color:.0f}/4")
        fig_ema.update_layout(margin=dict(t=0, l=0, r=0, b=0), height=500)
        st.plotly_chart(fig_ema, use_container_width=True)

    # --- TAB 4: Market ---
    with tabs[3]:
        st.dataframe(df.style.format({"Price": "{:.2f}", "Change": "{:.2f}%"}).background_gradient(cmap='RdYlGn', subset=['Change']), column_config={"TV": link_col}, use_container_width=True)

    # --- TAB 5: Chart ---
    with tabs[4]:
        sel = st.selectbox("سهم:", df['Name'].unique())
        if sel:
            hist = st.session_state['history'][sel]
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name='Price'), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['EMA'], line=dict(color='orange'), name='EMA'), row=1, col=1)
            
            # رسم الصندوق مع تقييم الذكاء
            box_res = check_bullish_box_advanced(hist)
            if box_res:
                latest = box_res[-1]
                # نحسب السكور لهذا السهم تحديداً
                score, _ = calculate_ai_score(hist.iloc[-1], latest)
                color_box = "green" if score >= 60 else "gray"
                
                fig.add_shape(type="rect", x0=hist.index[-latest['Days_Ago']-2], x1=hist.index[-1], y0=latest['Box_Bottom'], y1=latest['Box_Top'], 
                              line=dict(color=color_box, width=2), fillcolor=f"rgba(0,255,0,0.1)", row=1, col=1)
                
            fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False, paper_bgcolor='#161b24', plot_bgcolor='#161b24')
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("👋 جاهز للعمل! اضغط الزر الأخضر.")
