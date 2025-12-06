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
    
    /* خلفية الفضاء العميق */
    .stApp { background-color: #000000; color: #ffffff; }
    
    /* زر الإطلاق */
    div.stButton > button {
        background: radial-gradient(circle, #2962ff 0%, #000000 100%);
        color: white; border: 1px solid #2962ff;
        padding: 15px 30px; border-radius: 50px;
        font-weight: bold; font-size: 20px; width: 100%;
        box-shadow: 0 0 25px rgba(41, 98, 255, 0.5);
        transition: transform 0.2s;
    }
    div.stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 45px rgba(41, 98, 255, 0.8);
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
    if status == "Bull": return "#00e676" # أخضر مشع
    elif status == "Bear": return "#ff1744" # أحمر مشع
    else: return "#263238" # رمادي غامق جداً (مطفى)

# --- 4. المحرك الرئيسي ---
st.title("🌌 TASI Galaxy (Live Space View)")

if 'galaxy_data' not in st.session_state: st.session_state['galaxy_data'] = []

if st.button("🪐 إطلاق المسبار الفضائي (Scan)"):
    st.session_state['galaxy_data'] = []
    progress = st.progress(0); status = st.empty()
    tickers = list(TICKERS.keys())
    
    chunk_size = 30
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        status.text(f"جاري مسح النظام الشمسي... {i//chunk_size + 1}")
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
    progress.empty(); status.success("تم بناء المجرة!")

# --- 5. رسم المجرة (Advanced Solar System) ---
if st.session_state['galaxy_data']:
    df = pd.DataFrame(st.session_state['galaxy_data'])
    
    fig = go.Figure()
    
    # 0. خلفية النجوم (Starfield) لإعطاء عمق
    # ننشئ 200 نقطة عشوائية بيضاء في الخلفية
    star_x = [random.uniform(-150, 150) for _ in range(300)]
    star_y = [random.uniform(-150, 150) for _ in range(300)]
    fig.add_trace(go.Scatter(
        x=star_x, y=star_y, mode='markers',
        marker=dict(size=2, color='white', opacity=0.3),
        hoverinfo='none'
    ))

    # 1. الشمس (TASI)
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode='markers+text',
        marker=dict(size=80, color='#ffab00', line=dict(color='#ffd600', width=6), opacity=0.9),
        text=["<b>TASI</b>"], textposition="middle center",
        textfont=dict(color='black', size=16, family="Cairo"),
        hoverinfo='none'
    ))
    
    # توزيع القطاعات
    sectors = df['Sector'].unique()
    # زيادة نصف القطر بشكل كبير لإعطاء مساحة
    sector_radius = 65 
    
    for i, sec in enumerate(sectors):
        # زاوية القطاع
        sec_angle = (2 * math.pi * i) / len(sectors)
        sec_x = sector_radius * math.cos(sec_angle)
        sec_y = sector_radius * math.sin(sec_angle)
        
        # رسم كوكب القطاع
        fig.add_trace(go.Scatter(
            x=[sec_x], y=[sec_y], mode='markers+text',
            marker=dict(size=35, color='#2962ff', line=dict(color='#82b1ff', width=2), opacity=0.9),
            text=[sec], textposition="bottom center",
            textfont=dict(color='#e3f2fd', size=14, weight="bold"),
            hoverinfo='none'
        ))
        
        # توزيع أسهم القطاع (سحابة حول الكوكب)
        sec_stocks = df[df['Sector'] == sec]
        num_stocks = len(sec_stocks)
        
        for j, (_, stock) in enumerate(sec_stocks.iterrows()):
            stock_angle = (2 * math.pi * j) / num_stocks
            
            # --- التحسين الجوهري: تباعد عشوائي ---
            # بدلاً من نصف قطر ثابت، نجعله يتراوح بين قيمة دنيا وعليا
            # هذا يخلق شكل "سحابة" أو "حزام" بدلاً من خط دائري نحيف
            random_scatter = random.uniform(10, 22) # مسافة عشوائية عن مركز القطاع
            
            stk_x = sec_x + random_scatter * math.cos(stock_angle)
            stk_y = sec_y + random_scatter * math.sin(stock_angle)
            
            # معلومات السهم للنقر
            hover_text = f"<b>{stock['Name']}</b><br>السعر: {stock['Price']:.2f}<br>يومي: {stock['Daily']}<br>أسبوعي: {stock['Weekly']}<br>شهري: {stock['Monthly']}"
            
            # النواة (اليومي)
            fig.add_trace(go.Scatter(
                x=[stk_x], y=[stk_y], mode='markers',
                marker=dict(size=8, color=get_color(stock['Daily']), line=dict(color='white', width=1)),
                text=hover_text, hoverinfo='text', name=stock['Name']
            ))
            
            # الهالات (أسبوعي وشهري) - تكبير الحجم قليلاً
            fig.add_trace(go.Scatter(
                x=[stk_x], y=[stk_y], mode='markers',
                marker=dict(size=16, color=get_color(stock['Weekly']), opacity=0.6),
                hoverinfo='none', showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=[stk_x], y=[stk_y], mode='markers',
                marker=dict(size=26, color=get_color(stock['Monthly']), opacity=0.3),
                hoverinfo='none', showlegend=False
            ))
            
            # خط خافت جداً يربط السهم بالقطاع (اختياري، لربط بصري)
            fig.add_trace(go.Scatter(
                x=[sec_x, stk_x], y=[sec_y, stk_y], mode='lines',
                line=dict(color='rgba(255,255,255,0.05)', width=0.5),
                hoverinfo='none', showlegend=False
            ))

    # --- إعدادات العرض النهائية ---
    fig.update_layout(
        template="plotly_dark",
        height=1000, # شاشة عملاقة
        width=1000,
        paper_bgcolor='#000000',
        plot_bgcolor='#000000',
        showlegend=False,
        # إخفاء المحاور تماماً
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, visible=False),
        # إعدادات التفاعل (Pan & Zoom)
        dragmode='pan',
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    # تفعيل التقريب بالعجلة
    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})
    
    st.info("💡 **طريقة الاستخدام:** استخدم عجلة الماوس للتقريب (Zoom) والدخول داخل القطاعات. الأسهم متباعدة لسهولة النقر.")

else:
    st.write("") # فراغ حتى الضغط
