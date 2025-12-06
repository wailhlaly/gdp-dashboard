import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pydeck as pdk # المكتبة الجديدة 🌟
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
st.set_page_config(page_title="TASI Galaxy 3D (PyDeck)", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif; }
    .stApp { background-color: #000000; color: #ffffff; }
    div.stButton > button {
        background: linear-gradient(45deg, #7c4dff, #2962ff); color: white; border: none;
        padding: 15px 40px; border-radius: 50px; font-weight: bold; font-size: 22px; width: 100%;
        box-shadow: 0 0 30px rgba(124, 77, 255, 0.5);
    }
    /* إخفاء عناصر التحكم الافتراضية لـ PyDeck لجعلها أنظف */
    .deckgl-control { display: none !important; }
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

# دالة تحويل الحالة إلى لون RGB لـ PyDeck
def get_color_rgb(status):
    if status == "Bull": return [0, 230, 118, 255] # أخضر نيون
    elif status == "Bear": return [255, 23, 68, 255] # أحمر نيون
    else: return [55, 71, 79, 150] # رمادي شفاف

# --- 4. المحرك الرئيسي ---
st.title("🌌 TASI Galaxy 3D (Powered by PyDeck)")

if 'galaxy_data_3d' not in st.session_state: st.session_state['galaxy_data_3d'] = []

if st.button("🚀 إطلاق المحرك ثلاثي الأبعاد (Scan 3D)"):
    st.session_state['galaxy_data_3d'] = []
    progress = st.progress(0); status = st.empty()
    tickers = list(TICKERS.keys())
    
    chunk_size = 30
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        status.text(f"جاري بناء النموذج ثلاثي الأبعاد... {i//chunk_size + 1}")
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
                                
                                st.session_state['galaxy_data_3d'].append({
                                    "Name": name, "Sector": sector,
                                    "Daily": s_d, "Weekly": s_w, "Monthly": s_m,
                                    "Price": df_d['Close'].iloc[-1]
                                })
                    except: continue
        except: pass
        progress.progress(min((i + chunk_size) / len(tickers), 1.0))
    progress.empty(); status.success("المجرة 3D جاهزة!")

# --- 5. رسم المجرة باستخدام PyDeck 🌟 ---
if st.session_state['galaxy_data_3d']:
    df_res = pd.DataFrame(st.session_state['galaxy_data_3d'])
    
    # --- تجهيز بيانات PyDeck ---
    layers = []
    
    # 1. الشمس (TASI) - مركز ثابت
    sun_data = [{"name": "TASI (المؤشر العام)", "pos": [0, 0, 0], "color": [255, 171, 0, 255], "radius": 150}]
    layers.append(pdk.Layer(
        "ScatterplotLayer",
        data=sun_data,
        get_position="pos",
        get_color="color",
        get_radius="radius",
        pickable=True,
        opacity=0.9,
        stroked=True, filled=True, radius_scale=1, line_width_min_pixels=5, get_line_color=[255, 214, 0]
    ))

    # 2. القطاعات (كواكب تدور حول الشمس)
    sectors = df_res['Sector'].unique()
    sector_radius_base = 400 # توسيع المدار
    sector_positions = {}

    sector_plot_data = []
    for i, sec in enumerate(sectors):
        angle = (2 * math.pi * i) / len(sectors)
        # إضافة تنوع بسيط في الارتفاع (Z) لجعل المدارات متموجة
        z_offset = 50 * math.sin(angle * 3) 
        sx = sector_radius_base * math.cos(angle)
        sy = sector_radius_base * math.sin(angle)
        sz = z_offset
        sector_positions[sec] = (sx, sy, sz)
        
        sector_plot_data.append({
            "name": sec, "pos": [sx, sy, sz], "color": [41, 98, 255, 200], "radius": 60
        })

    layers.append(pdk.Layer(
        "ScatterplotLayer",
        data=sector_plot_data,
        get_position="pos", get_color="color", get_radius="radius",
        pickable=True, opacity=0.8, stroked=True, line_width_min_pixels=2, get_line_color=[130, 177, 255]
    ))

    # 3. الأسهم (أعمدة مكدسة تدور حول القطاعات)
    stock_plot_data = []
    
    # ارتفاع الطبقات في العمود
    Z_DAILY = 0
    Z_WEEKLY = 25
    Z_MONTHLY = 50
    STOCK_RADIUS = 12

    for i, row in df_res.iterrows():
        sec_pos = sector_positions[row['Sector']]
        
        # توزيع عشوائي ثلاثي الأبعاد حول القطاع (سحابة كروية)
        # نستخدم إحداثيات كروية لتوزيع طبيعي
        phi = random.uniform(0, 2 * math.pi)
        theta = random.uniform(0, math.pi)
        dist = random.uniform(80, 180) # مسافة الانتشار عن القطاع
        
        dx = dist * math.sin(theta) * math.cos(phi)
        dy = dist * math.sin(theta) * math.sin(phi)
        dz = dist * math.cos(theta) * 0.5 # ضغط الانتشار العمودي قليلاً

        base_x = sec_pos[0] + dx
        base_y = sec_pos[1] + dy
        base_z = sec_pos[2] + dz
        
        tooltip_text = f"{row['Name']} \n السعر: {row['Price']:.2f} \n يومي: {row['Daily']} \n أسبوعي: {row['Weekly']} \n شهري: {row['Monthly']}"

        # الطبقة 1: اليومي (القاعدة)
        stock_plot_data.append({
            "name": row['Name'], "pos": [base_x, base_y, base_z + Z_DAILY],
            "color": get_color_rgb(row['Daily']), "radius": STOCK_RADIUS, "info": tooltip_text, "frame": "يومي"
        })
        # الطبقة 2: الأسبوعي (الوسط)
        stock_plot_data.append({
            "name": row['Name'], "pos": [base_x, base_y, base_z + Z_WEEKLY],
            "color": get_color_rgb(row['Weekly']), "radius": STOCK_RADIUS * 0.9, "info": tooltip_text, "frame": "أسبوعي"
        })
        # الطبقة 3: الشهري (القمة)
        stock_plot_data.append({
            "name": row['Name'], "pos": [base_x, base_y, base_z + Z_MONTHLY],
            "color": get_color_rgb(row['Monthly']), "radius": STOCK_RADIUS * 0.8, "info": tooltip_text, "frame": "شهري"
        })

    # إضافة طبقة الأسهم
    layers.append(pdk.Layer(
        "ScatterplotLayer",
        data=stock_plot_data,
        get_position="pos", get_color="color", get_radius="radius",
        pickable=True, # مهم لظهور المعلومات عند التحويم
        opacity=1.0,
        stroked=True, line_width_min_pixels=1, get_line_color=[255,255,255, 50]
    ))

    # --- إعدادات الكاميرا والإضاءة (Cinematic View) ---
    view_state = pdk.ViewState(
        latitude=0, longitude=0, # مركز العالم
        zoom=0.5, # زوم بعيد لرؤية المجرة كاملة
        pitch=45, # زاوية نظر مائلة (سينمائية)
        bearing=0 # دوران الكاميرا
    )
    
    # إعدادات الإضاءة والجو العام
    r = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style=None, # خلفية سوداء تماماً بدون خريطة أرضية
        tooltip={"html": "<b>{info}</b>", "style": {"backgroundColor": "#1c1c1c", "color": "white", "fontSize": "14px", "borderRadius": "5px"}}
    )
    
    # عرض الشارت في Streamlit
    st.pydeck_chart(r, use_container_width=True)
    
    st.markdown("""
    <div style="text-align: center; color: #b0bec5; padding: 20px;">
    🖱️ <b>التحكم بالماوس (PC):</b><br>
    • <b>الزر الأيسر + السحب:</b> للتدوير (Rotate).<br>
    • <b>الزر الأيمن + السحب:</b> للتحريك الجانبي (Pan).<br>
    • <b>العجلة:</b> للتقريب والتبعيد (Zoom).<br><br>
    👆 <b>التحكم باللمس (Mobile):</b><br>
    • <b>إصبع واحد:</b> للتدوير.<br>
    • <b>إصبعين:</b> للتحريك الجانبي والتقريب (Pinch).
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("🌌 اضغط الزر البنفسجي لبناء المجرة ثلاثية الأبعاد.")
