import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import random
import math

# --- استيراد البيانات ---
try:
    from data.saudi_tickers import STOCKS_DB
except ImportError:
    st.error("🚨 ملف البيانات مفقود.")
    st.stop()

TICKERS = {item['symbol']: item['name'] for item in STOCKS_DB}
SECTORS_MAP = {item['name']: item['sector'] for item in STOCKS_DB}

# --- 1. إعداد الصفحة ---
st.set_page_config(page_title="TASI Galaxy Pro", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif; }
    
    /* خلفية سوداء تماماً لتباين أفضل */
    .stApp { background-color: #000000; color: #ffffff; }
    
    /* زر الإطلاق بتأثير نيون */
    div.stButton > button {
        background: radial-gradient(circle, #2962ff 0%, #000000 100%);
        border: 1px solid #2962ff; color: white;
        padding: 15px 40px; border-radius: 50px;
        font-weight: bold; font-size: 22px; width: 100%;
        box-shadow: 0 0 30px rgba(41, 98, 255, 0.4);
        transition: all 0.3s;
    }
    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 50px rgba(41, 98, 255, 0.8);
        border-color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. الإعدادات ---
with st.sidebar:
    st.header("⚙️ التحكم")
    ATR_MULT = st.number_input("ATR Multiplier", 1.0, 3.0, 1.5)
    BOX_LOOKBACK = st.slider("نطاق البحث", 5, 50, 20)

# --- 3. الدوال الفنية ---
def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    return ranges.max(axis=1).ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def get_box_status(df, lookback):
    if len(df) < 30: return "---"
    df['ATR'] = calculate_atr(df)
    prices = df.iloc[-lookback:].reset_index(); atrs = df['ATR'].iloc[-lookback:].values
    latest_status = "---"
    
    in_series = False; mode = None; start_open = 0.0; end_close = 0.0
    
    for i in range(len(prices)):
        row = prices.iloc[i]; close = row['Close']; open_p = row['Open']
        is_green = close > open_p; is_red = close < open_p
        current_atr = atrs[i]
        if np.isnan(current_atr): continue
        
        if not in_series:
            if is_green: in_series = True; mode = 'bull'; start_open = open_p
            elif is_red: in_series = True; mode = 'bear'; start_open = open_p
        elif in_series:
            if mode == 'bull' and is_green: end_close = close
            elif mode == 'bear' and is_red: end_close = close
            elif (mode == 'bull' and is_red) or (mode == 'bear' and is_green):
                final_close = end_close if end_close != 0 else start_open
                price_move = abs(final_close - start_open)
                if price_move >= current_atr * ATR_MULT:
                    current_price = prices.iloc[-1]['Close']
                    box_top = max(start_open, final_close); box_bottom = min(start_open, final_close)
                    if mode == 'bull':
                        if current_price >= box_bottom: latest_status = "Bull"
                    else:
                        if current_price <= box_top: latest_status = "Bear"
                in_series = True; mode = 'bull' if is_green else 'bear'; start_open = open_p; end_close = close
    return latest_status

def get_color(status):
    if status == "Bull": return "#00e676" # أخضر ساطع
    elif status == "Bear": return "#d50000" # أحمر دموي
    else: return "#37474f" # رمادي كحلي

# --- 4. المحرك الرئيسي ---
st.title("🌌 TASI Galaxy (Touch & Zoom)")

if 'galaxy_data' not in st.session_state: st.session_state['galaxy_data'] = []

if st.button("🪐 استكشاف الكون (Scan Universe)"):
    st.session_state['galaxy_data'] = []
    progress = st.progress(0); status = st.empty()
    tickers = list(TICKERS.keys())
    
    chunk_size = 30
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        status.text(f"جاري بناء المجرات... {i//chunk_size + 1}")
        try:
            raw_daily = yf.download(chunk, period="2y", interval="1d", group_by='ticker', auto_adjust=False, threads=True, progress=False)
            if not raw_daily.empty:
                for sym in chunk:
                    try:
                        name = TICKERS[sym]; sector = SECTORS_MAP.get(name, "أخرى")
                        try: df_d = raw_daily[sym].copy()
                        except: continue
                        
                        col = 'Close' if 'Close' in df_d.columns else 'Adj Close'
                        if col in df_d.columns:
                            df_d = df_d.rename(columns={col: 'Close'}); df_d = df_d.dropna()
                            if len(df_d) > 50:
                                s_d = get_box_status(df_d, BOX_LOOKBACK)
                                df_w = df_d.resample('W').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
                                s_w = get_box_status(df_w, BOX_LOOKBACK)
                                df_m = df_d.resample('ME').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
                                s_m = get_box_status(df_m, BOX_LOOKBACK)
                                
                                st.session_state['galaxy_data'].append({
                                    "Name": name, "Sector": sector,
                                    "Daily": s_d, "Weekly": s_w, "Monthly": s_m,
                                    "Price": df_d['Close'].iloc[-1]
                                })
                    except: continue
        except: pass
        progress.progress(min((i + chunk_size) / len(tickers), 1.0))
    progress.empty(); status.success("المجرة جاهزة للاستكشاف!")

# --- 5. الرسم (تحسينات اللمس والعرض) ---
if st.session_state['galaxy_data']:
    df = pd.DataFrame(st.session_state['galaxy_data'])
    fig = go.Figure()
    
    # 0. خلفية النجوم (طبقتين لعمق أكبر)
    # نجوم بعيدة (صغيرة وكثيرة)
    fig.add_trace(go.Scatter(
        x=[random.uniform(-180, 180) for _ in range(500)],
        y=[random.uniform(-180, 180) for _ in range(500)],
        mode='markers', marker=dict(size=1, color='white', opacity=0.3), hoverinfo='skip'
    ))
    # نجوم قريبة (أكبر وألمع)
    fig.add_trace(go.Scatter(
        x=[random.uniform(-180, 180) for _ in range(100)],
        y=[random.uniform(-180, 180) for _ in range(100)],
        mode='markers', marker=dict(size=2.5, color='#e0f7fa', opacity=0.6), hoverinfo='skip'
    ))

    # 1. الشمس (TASI) مع توهج
    # التوهج
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode='markers',
        marker=dict(size=140, color='#ffab00', opacity=0.2), hoverinfo='skip'
    ))
    # الجسم الأساسي
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode='markers+text',
        marker=dict(size=80, color='#ffab00', line=dict(color='#ffd600', width=4)),
        text=["<b>TASI</b>"], textposition="middle center",
        textfont=dict(color='black', size=20, family="Cairo", weight="bold"),
        hoverinfo='skip'
    ))
    
    sectors = df['Sector'].unique()
    sector_base_radius = 80 # نصف قطر المدار الأول
    
    for i, sec in enumerate(sectors):
        # توزيع القطاعات في مدارات مختلفة لتقليل الازدحام
        # كل قطاعين يأخذان مساراً أبعد قليلاً
        current_orbit_radius = sector_base_radius + (i % 2) * 30 
        
        sec_angle = (2 * math.pi * i) / len(sectors)
        sec_x = current_orbit_radius * math.cos(sec_angle)
        sec_y = current_orbit_radius * math.sin(sec_angle)
        
        # رسم خط المدار (خافت جداً) لربط القطاع بالشمس
        fig.add_trace(go.Scatter(
            x=[0, sec_x], y=[0, sec_y], mode='lines',
            line=dict(color='rgba(255, 255, 255, 0.05)', width=1),
            hoverinfo='skip'
        ))
        
        # كوكب القطاع
        fig.add_trace(go.Scatter(
            x=[sec_x], y=[sec_y], mode='markers+text',
            marker=dict(size=40, color='#2962ff', line=dict(color='white', width=1), opacity=0.9),
            text=[sec], textposition="bottom center",
            textfont=dict(color='#bbdefb', size=14, weight="bold"),
            hoverinfo='none'
        ))
        
        # الأسهم
        sec_stocks = df[df['Sector'] == sec]
        num_stocks = len(sec_stocks)
        
        stk_xs, stk_ys, stk_cols, stk_txts = [], [], [], []
        halo_w_x, halo_w_y, halo_w_c = [], [], []
        halo_m_x, halo_m_y, halo_m_c = [], [], []
        
        for j, (_, stock) in enumerate(sec_stocks.iterrows()):
            stock_angle = (2 * math.pi * j) / num_stocks
            # مسافة انتشار الأسهم حول القطاع (عشوائية قليلاً لتبدو طبيعية)
            dist = random.uniform(15, 28) 
            
            sx = sec_x + dist * math.cos(stock_angle)
            sy = sec_y + dist * math.sin(stock_angle)
            
            # نص المعلومات (HTML formatted)
            tooltip = f"""
            <span style='font-size:18px; font-weight:bold; color:white'>{stock['Name']}</span><br>
            <span style='color:#b0bec5'>السعر: {stock['Price']:.2f}</span><br>
            <span style='color:{get_color(stock['Daily'])}'>● يومي</span>
            <span style='color:{get_color(stock['Weekly'])}'>● أسبوعي</span>
            <span style='color:{get_color(stock['Monthly'])}'>● شهري</span>
            """
            
            stk_xs.append(sx); stk_ys.append(sy)
            stk_cols.append(get_color(stock['Daily']))
            stk_txts.append(tooltip)
            
            # بيانات الهالات
            halo_w_x.append(sx); halo_w_y.append(sy); halo_w_c.append(get_color(stock['Weekly']))
            halo_m_x.append(sx); halo_m_y.append(sy); halo_m_c.append(get_color(stock['Monthly']))

        # رسم الهالات (شهري - أسبوعي)
        fig.add_trace(go.Scatter(
            x=halo_m_x, y=halo_m_y, mode='markers',
            marker=dict(size=30, color=halo_m_c, opacity=0.2), hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=halo_w_x, y=halo_w_y, mode='markers',
            marker=dict(size=20, color=halo_w_c, opacity=0.5), hoverinfo='skip'
        ))
        
        # رسم الأنوية (اليومي - المتفاعلة)
        fig.add_trace(go.Scatter(
            x=stk_xs, y=stk_ys, mode='markers',
            marker=dict(size=12, color=stk_cols, line=dict(color='white', width=1)),
            text=stk_txts, hoverinfo='text',
            hoverlabel=dict(bgcolor="#1c1c1c", bordercolor="white", font=dict(color="white"))
        ))

    # --- إعدادات اللمس والتفاعل (The Magic Config) ---
    fig.update_layout(
        template="plotly_dark",
        height=900, width=900, # مربع ليكون متناسقاً
        paper_bgcolor='#000000', plot_bgcolor='#000000',
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False, fixedrange=False), 
        yaxis=dict(visible=False, fixedrange=False),
        
        # إعدادات التفاعل المحسنة
        hovermode='closest', # يلتقط أقرب عنصر
        hoverdistance=50,    # مسافة التقاط معقولة
        dragmode='pan'       # الوضع الافتراضي هو التحريك (Pan)
    )
    
    # تفعيل خيارات التكبير باللمس (Pinch) والعجلة
    config = {
        'scrollZoom': True,       # تفعيل عجلة الماوس
        'displayModeBar': True,   # إظهار شريط الأدوات (مهم للجوال للتبديل بين Pan و Zoom)
        'doubleClick': 'reset',   # نقرتين لإعادة الضبط
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'], # إزالة أدوات غير مفيدة
        'responsive': True        # استجابة لحجم الشاشة
    }
    
    st.plotly_chart(fig, use_container_width=True, config=config)
    st.markdown("""
    <div style="text-align: center; color: gray; padding: 10px;">
    🤏 <b>للجوال:</b> استخدم إصبعين للتقريب (Pinch) وإصبع واحد للتحريك.<br>
    🖱️ <b>للكمبيوتر:</b> استخدم عجلة الماوس للتقريب.
    </div>
    """, unsafe_allow_html=True)

else:
    st.write("")
