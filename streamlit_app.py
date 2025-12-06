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
st.set_page_config(page_title="TASI 3D Touch", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif; }
    
    .stApp { background-color: #000000; color: #ffffff; }
    
    div.stButton > button {
        background: radial-gradient(circle, #00b0ff 0%, #000000 100%);
        border: 1px solid #40c4ff; color: white;
        padding: 15px 30px; border-radius: 50px;
        font-weight: bold; font-size: 20px; width: 100%;
        box-shadow: 0 0 30px rgba(0, 176, 255, 0.5);
    }
    div.stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 50px rgba(0, 176, 255, 0.8);
    }
</style>
""", unsafe_allow_html=True)

# --- 2. الإعدادات ---
with st.sidebar:
    st.header("⚙️ التحكم")
    # خيار جديد لإظهار/إخفاء الأسماء
    SHOW_LABELS = st.checkbox("إظهار أسماء الأسهم", value=False, help="قد يسبب ازدحاماً إذا كانت الأسهم كثيرة")
    st.divider()
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

def get_color_hex(status):
    if status == "Bull": return "#00e676" 
    elif status == "Bear": return "#ff1744" 
    else: return "#607d8b" 

# --- 4. المحرك الرئيسي ---
st.title("🌌 TASI 3D Universe (Interactive)")

if 'galaxy_data_v17' not in st.session_state: st.session_state['galaxy_data_v17'] = []

if st.button("🪐 استكشاف الكون (Build Galaxy)"):
    st.session_state['galaxy_data_v17'] = []
    progress = st.progress(0); status = st.empty()
    tickers = list(TICKERS.keys())
    
    chunk_size = 30
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        status.text(f"معالجة البيانات... {i//chunk_size + 1}")
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
                                
                                st.session_state['galaxy_data_v17'].append({
                                    "Name": name, "Sector": sector,
                                    "Daily": s_d, "Weekly": s_w, "Monthly": s_m,
                                    "Price": df_d['Close'].iloc[-1]
                                })
                    except: continue
        except: pass
        progress.progress(min((i + chunk_size) / len(tickers), 1.0))
    progress.empty(); status.success("تم!")

# --- 5. الرسم (تفعيل الأسماء واللمس) ---
if st.session_state['galaxy_data_v17']:
    df = pd.DataFrame(st.session_state['galaxy_data_v17'])
    
    # 1. الشمس
    fig = go.Figure(data=[go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers+text',
        marker=dict(size=60, color='#ffab00', opacity=1),
        text=["<b>TASI</b>"], textfont=dict(size=24, color='white'),
        hoverinfo='none'
    )])
    
    sectors = df['Sector'].unique()
    sector_radius = 450 
    
    for i, sec in enumerate(sectors):
        sec_angle = (2 * math.pi * i) / len(sectors)
        sec_x = sector_radius * math.cos(sec_angle)
        sec_y = sector_radius * math.sin(sec_angle)
        sec_z = 0
        
        # الكوكب (القطاع)
        fig.add_trace(go.Scatter3d(
            x=[sec_x], y=[sec_y], z=[sec_z],
            mode='markers+text',
            marker=dict(size=25, color='#2962ff', opacity=0.8),
            text=[sec], textposition="top center",
            textfont=dict(color='#82b1ff', size=14, weight="bold"),
            hoverinfo='none'
        ))
        
        # الأسهم
        sec_stocks = df[df['Sector'] == sec]
        
        xs, ys, zs, colors, sizes, texts, labels = [], [], [], [], [], [], []
        
        for _, stock in sec_stocks.iterrows():
            r = random.uniform(40, 100) 
            theta = random.uniform(0, 2*math.pi)
            phi = random.uniform(0, math.pi)
            
            dx = r * math.sin(phi) * math.cos(theta)
            dy = r * math.sin(phi) * math.sin(theta)
            dz = r * math.cos(phi) * 0.4
            
            xs.append(sec_x + dx)
            ys.append(sec_y + dy)
            zs.append(sec_z + dz)
            
            colors.append(get_color_hex(stock['Daily']))
            
            size = 5
            if stock['Daily'] == 'Bull' and stock['Weekly'] == 'Bull': size = 12
            sizes.append(size)
            
            # النص عند التحويم (Tooltip)
            tooltip = f"<b>{stock['Name']}</b><br>{stock['Price']:.2f}<br>D:{stock['Daily']} W:{stock['Weekly']}"
            texts.append(tooltip)
            # النص الظاهر (Label)
            labels.append(stock['Name'])
            
        # تحديد وضع العرض (نص أم بدون نص)
        mode_setting = 'markers+text' if SHOW_LABELS else 'markers'
        
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode=mode_setting, # تفعيل النص بناءً على الخيار
            marker=dict(size=sizes, color=colors, opacity=0.9, line=dict(width=0)),
            text=labels if SHOW_LABELS else texts, # إذا أظهرنا الأسماء نضع الاسم، وإلا نضع التلميح
            hovertext=texts, # التلميح يظهر دائماً عند اللمس
            hoverinfo='text',
            textfont=dict(size=10, color='rgba(255,255,255,0.8)'), # تنسيق الأسماء
            name=sec
        ))

    # --- إعدادات الكاميرا واللمس ---
    fig.update_layout(
        height=900,
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor='black',
        showlegend=False,
        scene=dict(
            xaxis=dict(visible=False, showbackground=False),
            yaxis=dict(visible=False, showbackground=False),
            zaxis=dict(visible=False, showbackground=False),
            bgcolor='black',
            dragmode='orbit', # أفضل وضع للمس (يدور حول المركز)
            aspectmode='data'
        )
    )
    
    # تفعيل خيارات التفاعل للجوال
    config = {
        'scrollZoom': True,
        'displayModeBar': False,
        'responsive': True
    }
    
    st.plotly_chart(fig, use_container_width=True, config=config)
    
    if not SHOW_LABELS:
        st.caption("💡 **تلميح:** فعّل خيار 'إظهار أسماء الأسهم' من القائمة الجانبية لرؤية الشركات بدون لمس.")

else:
    st.write("")
