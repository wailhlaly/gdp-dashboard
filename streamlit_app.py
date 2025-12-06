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
st.set_page_config(page_title="TASI Galaxy Touch", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif; }
    
    .stApp { background-color: #000000; color: #ffffff; }
    
    div.stButton > button {
        background: radial-gradient(circle, #00e676 0%, #000000 100%);
        color: white; border: 1px solid #00e676;
        padding: 15px 30px; border-radius: 50px;
        font-weight: bold; font-size: 20px; width: 100%;
        box-shadow: 0 0 25px rgba(0, 230, 118, 0.5);
        transition: transform 0.2s;
    }
    div.stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 45px rgba(0, 230, 118, 0.8);
    }
</style>
""", unsafe_allow_html=True)

# --- 2. الإعدادات ---
with st.sidebar:
    st.header("⚙️ إعدادات الرادار")
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
    if status == "Bull": return "#00e676" 
    elif status == "Bear": return "#ff1744" 
    else: return "#37474f" 

# --- 4. المحرك الرئيسي ---
st.title("🌌 TASI Galaxy (Touch Enabled)")

if 'galaxy_data' not in st.session_state: st.session_state['galaxy_data'] = []

if st.button("🚀 إطلاق المسح (Scan)"):
    st.session_state['galaxy_data'] = []
    progress = st.progress(0); status = st.empty()
    tickers = list(TICKERS.keys())
    
    chunk_size = 30
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        status.text(f"جاري التحديث... {i//chunk_size + 1}")
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
    progress.empty(); status.success("جاهز!")

# --- 5. الرسم (تعديلات اللمس) ---
if st.session_state['galaxy_data']:
    df = pd.DataFrame(st.session_state['galaxy_data'])
    
    fig = go.Figure()
    
    # نجوم الخلفية
    star_x = [random.uniform(-150, 150) for _ in range(400)]
    star_y = [random.uniform(-150, 150) for _ in range(400)]
    fig.add_trace(go.Scatter(
        x=star_x, y=star_y, mode='markers',
        marker=dict(size=1.5, color='white', opacity=0.2),
        hoverinfo='skip' # تجاهل النجوم عند اللمس
    ))

    # الشمس (تاسي)
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode='markers+text',
        marker=dict(size=90, color='#ffab00', line=dict(color='#ffd600', width=4), opacity=1),
        text=["<b>TASI</b>"], textposition="middle center",
        textfont=dict(color='black', size=18, family="Cairo"),
        hoverinfo='skip'
    ))
    
    sectors = df['Sector'].unique()
    sector_radius = 65 
    
    for i, sec in enumerate(sectors):
        sec_angle = (2 * math.pi * i) / len(sectors)
        sec_x = sector_radius * math.cos(sec_angle)
        sec_y = sector_radius * math.sin(sec_angle)
        
        # الكوكب (القطاع)
        fig.add_trace(go.Scatter(
            x=[sec_x], y=[sec_y], mode='markers+text',
            marker=dict(size=40, color='#2962ff', line=dict(color='#82b1ff', width=2), opacity=0.9),
            text=[sec], textposition="bottom center",
            textfont=dict(color='#e3f2fd', size=16, weight="bold"),
            hoverinfo='none' # لا نريد تفاصيل للقطاع نفسه عند اللمس، بل للأسهم
        ))
        
        sec_stocks = df[df['Sector'] == sec]
        num_stocks = len(sec_stocks)
        
        # تجميع إحداثيات وبيانات الأسهم لرسمها في Trace واحد لكل قطاع (أسرع وأخف)
        stk_xs = []; stk_ys = []; stk_colors = []; stk_sizes = []; stk_texts = []
        halo_w_x = []; halo_w_y = []; halo_w_c = []
        halo_m_x = []; halo_m_y = []; halo_m_c = []
        
        for j, (_, stock) in enumerate(sec_stocks.iterrows()):
            stock_angle = (2 * math.pi * j) / num_stocks
            dist = random.uniform(12, 25) # مسافة الانتشار
            
            sx = sec_x + dist * math.cos(stock_angle)
            sy = sec_y + dist * math.sin(stock_angle)
            
            # بيانات السهم (للمس)
            # تنسيق HTML للنص ليظهر بشكل جميل
            ht = f"""
            <b style='font-size:16px'>{stock['Name']}</b><br>
            السعر: {stock['Price']:.2f}<br>
            📅 يومي: {stock['Daily']}<br>
            🗓️ أسبوعي: {stock['Weekly']}<br>
            📆 شهري: {stock['Monthly']}
            """
            
            # النواة
            stk_xs.append(sx); stk_ys.append(sy)
            stk_colors.append(get_color(stock['Daily']))
            stk_texts.append(ht)
            
            # الهالات
            halo_w_x.append(sx); halo_w_y.append(sy); halo_w_c.append(get_color(stock['Weekly']))
            halo_m_x.append(sx); halo_m_y.append(sy); halo_m_c.append(get_color(stock['Monthly']))

        # رسم الهالات أولاً (لتبقى في الخلف)
        fig.add_trace(go.Scatter(
            x=halo_m_x, y=halo_m_y, mode='markers',
            marker=dict(size=28, color=halo_m_c, opacity=0.3),
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=halo_w_x, y=halo_w_y, mode='markers',
            marker=dict(size=18, color=halo_w_c, opacity=0.6),
            hoverinfo='skip'
        ))
        
        # رسم الأنوية (التي تتفاعل مع اللمس)
        fig.add_trace(go.Scatter(
            x=stk_xs, y=stk_ys, mode='markers',
            marker=dict(size=10, color=stk_colors, line=dict(color='white', width=1)),
            text=stk_texts,
            hoverinfo='text', # عرض النص المخصص فقط
            hovertemplate="%{text}<extra></extra>", # إزالة اسم الـ Trace المزعج
            name=sec
        ))

    # --- إعدادات اللمس والتفاعل ---
    fig.update_layout(
        template="plotly_dark",
        height=1000,
        paper_bgcolor='#000000',
        plot_bgcolor='#000000',
        showlegend=False,
        xaxis=dict(visible=False, fixedrange=False), # fixedrange=False يسمح بالزوم
        yaxis=dict(visible=False, fixedrange=False),
        margin=dict(l=0, r=0, t=0, b=0),
        
        # --- السحر هنا (إعدادات اللمس) ---
        hovermode='closest', # يلتقط أقرب نقطة للإصبع
        hoverdistance=100,   # مسافة الالتقاط (كبيرة لتسهيل اللمس)
        spikedistance=100,   # تحسين الاستجابة
        dragmode='pan'       # التحرك بالسحب
    )
    
    st.plotly_chart(fig, use_container_width=True, config={
        'scrollZoom': True,
        'displayModeBar': False, # إخفاء الشريط العلوي لتجربة أنظف
        'doubleClick': 'reset'
    })
    
    st.info("💡 **نصيحة:** اقترب من أي كوكب أخضر لترى تفاصيل السهم فوراً.")

else:
    st.write("")
