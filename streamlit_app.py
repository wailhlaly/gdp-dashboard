import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

# --- استيراد البيانات ---
try:
    from data.saudi_tickers import STOCKS_DB
except ImportError:
    st.error("🚨 ملف البيانات مفقود.")
    st.stop()

TICKERS = {item['symbol']: item['name'] for item in STOCKS_DB}

# --- 1. إعداد الصفحة ---
st.set_page_config(page_title="TASI Matrix Pro", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif; direction: rtl; }
    
    .stApp { background-color: #0e1117; color: #ffffff; }
    .stDataFrame { border: 1px solid #30333d; border-radius: 8px; }
    
    /* تنسيق الخلايا الملونة */
    .bullish { background-color: #004d40; color: #b2dfdb; padding: 5px; border-radius: 4px; text-align: center; font-weight: bold; }
    .bearish { background-color: #3e2723; color: #ffccbc; padding: 5px; border-radius: 4px; text-align: center; font-weight: bold; }
    .neutral { color: #555; text-align: center; }
    
    /* زر التشغيل */
    div.stButton > button {
        background: linear-gradient(90deg, #2962ff, #0039cb); color: white; 
        border: none; padding: 10px 24px; border-radius: 8px; font-weight: bold; width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. الإعدادات ---
with st.sidebar:
    st.header("⚙️ الإعدادات")
    ATR_MULT = st.number_input("ATR Multiplier", 1.0, 3.0, 1.5)
    BOX_LOOKBACK = st.slider("نطاق البحث (شموع)", 5, 50, 20)

# --- 3. الدوال الفنية (Core Logic) ---

def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    return ranges.max(axis=1).ewm(alpha=1/period, min_periods=period, adjust=False).mean()

# دالة ذكية تفحص وجود صندوق (صاعد أو هابط) وترجع الحالة
def get_box_status(df, lookback):
    if len(df) < 50: return "---"
    
    # حساب ATR
    df['ATR'] = calculate_atr(df)
    
    prices = df.iloc[-lookback:].reset_index()
    atrs = df['ATR'].iloc[-lookback:].values
    
    latest_status = "---" # الحالة الافتراضية
    
    # خوارزمية الصندوق
    in_series = False; mode = None # 'bull' or 'bear'
    start_open = 0.0; end_close = 0.0
    
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
                # نهاية السلسلة
                final_close = end_close if end_close != 0 else start_open
                price_move = abs(final_close - start_open)
                
                # التحقق من الشرط
                if price_move >= current_atr * ATR_MULT:
                    # تم اكتشاف صندوق! هل السعر الحالي ما زال يحترمه؟
                    current_price = prices.iloc[-1]['Close']
                    box_top = max(start_open, final_close)
                    box_bottom = min(start_open, final_close)
                    
                    if mode == 'bull':
                        # الصندوق الصاعد: يعتبر فعالاً إذا السعر فوق قاعه
                        if current_price >= box_bottom: 
                            latest_status = "🟢 صاعد"
                    else:
                        # الصندوق الهابط: يعتبر فعالاً إذا السعر تحت قمته
                        if current_price <= box_top:
                            latest_status = "🔴 هابط"
                            
                # إعادة تعيين
                in_series = True
                mode = 'bull' if is_green else 'bear'
                start_open = open_p; end_close = close
                
    return latest_status

# --- 4. المحرك الرئيسي ---
st.title("📊 مصفوفة الصناديق الشاملة (Matrix View)")

if 'matrix_data' not in st.session_state: st.session_state['matrix_data'] = []

if st.button("🚀 تحديث المصفوفة (Scan All Timeframes)"):
    st.session_state['matrix_data'] = []
    progress = st.progress(0); status = st.empty()
    tickers = list(TICKERS.keys())
    
    # نسحب بيانات يومية لمدة سنتين (تكفي لاشتقاق الأسبوعي والشهري)
    chunk_size = 30
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        status.text(f"معالجة الدفعة {i//chunk_size + 1}...")
        
        try:
            # سحب البيانات اليومية الخام
            raw_daily = yf.download(chunk, period="2y", interval="1d", group_by='ticker', auto_adjust=False, threads=True, progress=False)
            
            if not raw_daily.empty:
                for sym in chunk:
                    try:
                        name = TICKERS[sym]
                        try: df_d = raw_daily[sym].copy()
                        except: continue
                        
                        # تنظيف
                        col = 'Close' if 'Close' in df_d.columns else 'Adj Close'
                        if col in df_d.columns:
                            df_d = df_d.rename(columns={col: 'Close'})
                            df_d = df_d.dropna()
                            if len(df_d) > 50:
                                last_price = df_d['Close'].iloc[-1]
                                
                                # 1. تحليل اليومي (Daily)
                                status_d = get_box_status(df_d, BOX_LOOKBACK)
                                
                                # 2. اشتقاق وتحليل الأسبوعي (Weekly Resample)
                                df_w = df_d.resample('W').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
                                status_w = get_box_status(df_w, BOX_LOOKBACK)
                                
                                # 3. اشتقاق وتحليل الشهري (Monthly Resample)
                                df_m = df_d.resample('ME').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
                                status_m = get_box_status(df_m, BOX_LOOKBACK) # ننظر لعدد أقل من الشهور عادة
                                
                                # لا نعرض السهم إلا إذا كان فيه صندوق واحد على الأقل
                                if "---" not in [status_d, status_w, status_m] or status_d != "---" or status_w != "---" or status_m != "---":
                                    link = f"https://www.tradingview.com/chart/?symbol=TADAWUL:{sym.replace('.SR','')}"
                                    
                                    st.session_state['matrix_data'].append({
                                        "الاسم": name,
                                        "السعر": last_price,
                                        "يومي": status_d,
                                        "أسبوعي": status_w,
                                        "شهري": status_m,
                                        "TV_Url": link
                                    })
                    except: continue
        except: pass
        progress.progress(min((i + chunk_size) / len(tickers), 1.0))
        
    progress.empty(); status.success("تم بناء المصفوفة!")

# --- 5. العرض (الجدول الموحد) ---
if st.session_state['matrix_data']:
    df = pd.DataFrame(st.session_state['matrix_data'])
    
    # إحصائيات سريعة
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("عدد الفرص", len(df))
    # حساب الشركات التي لديها توافق (يومي + أسبوعي صاعد)
    confluence = df[(df['يومي'] == "🟢 صاعد") & (df['أسبوعي'] == "🟢 صاعد")]
    c2.metric("توافق صاعد (D+W)", len(confluence))
    
    st.markdown("### 📋 التحليل الشامل (Matrix)")
    
    # دالة لتلوين الخلايا
    def style_matrix(val):
        if val == "🟢 صاعد":
            return 'background-color: #004d40; color: #e0f2f1; font-weight: bold; text-align: center;'
        elif val == "🔴 هابط":
            return 'background-color: #3e2723; color: #fbe9e7; font-weight: bold; text-align: center;'
        else:
            return 'color: #555; text-align: center;'

    # إعداد رابط الشارت
    link_config = st.column_config.LinkColumn("الشارت", display_text="Open TV")

    # عرض الجدول
    st.dataframe(
        df.style
        .format({"السعر": "{:.2f}"})
        .map(style_matrix, subset=['يومي', 'أسبوعي', 'شهري']), # تلوين الأعمدة الثلاثة
        column_config={"TV_Url": link_config},
        use_container_width=True,
        height=700
    )
else:
    st.info("اضغط الزر للبدء. سيتم فحص 260+ شركة على 3 فريمات.")
