import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# --- 1. إعداد الصفحة والوضع الليلي ---
st.set_page_config(page_title="Saudi Pro Ultimate", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    .stDataFrame { border: 1px solid #30333d; }
    div[data-testid="stDataFrame"] div[class*="css"] { background-color: #161b24; color: white; }
    div[data-testid="stMetric"] { background-color: #1d212b !important; border: 1px solid #30333d; padding: 15px; border-radius: 8px; color: white !important; }
    div[data-testid="stMetricLabel"] { color: #b0b3b8 !important; }
    div[data-testid="stMetricValue"] { color: #ffffff !important; }
    div.stButton > button { background-color: #2962ff; color: white; border: none; width: 100%; font-weight: bold; }
    div.stButton > button:hover { background-color: #1e53e5; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; background-color: transparent; }
    .stTabs [data-baseweb="tab"] { background-color: #1d212b; color: #e0e0e0; border-radius: 4px; }
    .stTabs [aria-selected="true"] { background-color: #2962ff !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# --- 2. الإعدادات ---
with st.sidebar:
    st.title("⚙️ الإعدادات")
    RSI_PERIOD = st.number_input("فترة RSI", value=24)
    EMA_PERIOD = st.number_input("فترة EMA", value=8)
    st.divider()
    st.markdown("### 📦 إعدادات الصندوق")
    ATR_LENGTH = st.number_input("طول ATR", value=14)
    ATR_MULT = st.number_input("مضاعف ATR", value=1.5)
    BOX_LOOKBACK = st.slider("بحث في آخر (شمعة)", 10, 50, 20)
    
    st.info("اضغط زر التشغيل لبدء التحليل الشامل.")

# --- 3. القائمة الشاملة للسوق السعودي (مصححة ومدققة) ---
TICKERS = {
    # === الطاقة ===
    "2222.SR": "أرامكو", "2030.SR": "المصافي", "4200.SR": "الدريس", "5110.SR": "الكهرباء", 
    "2080.SR": "الغاز", "4030.SR": "البحري", "2380.SR": "رابغ", "2381.SR": "الحفر العربية", "2382.SR": "أديس",
    
    # === المواد الأساسية (بتروكيماويات) ===
    "2010.SR": "سابك", "1211.SR": "معادن", "2020.SR": "سابك للمغذيات", "2310.SR": "سبكيم", 
    "2060.SR": "التصنيع", "2290.SR": "ينساب", "2001.SR": "كيمانول", "2170.SR": "اللجين", 
    "2330.SR": "المتقدمة", "2350.SR": "كيان", "2090.SR": "جبسكو", "2150.SR": "زجاج", 
    "2180.SR": "فيبكو", "2200.SR": "أنابيب", "2210.SR": "نما", "2230.SR": "الكيميائية", 
    "2240.SR": "الزامل", "2250.SR": "المجموعة", "2300.SR": "صناعة الورق", "2320.SR": "البابطين", 
    "2340.SR": "العبداللطيف", "2360.SR": "الفخارية", "2370.SR": "مسك", "1301.SR": "أسلاك", 
    "1304.SR": "اليمامة للحديد", "1320.SR": "أنابيب الشرق", "1321.SR": "أنابيب السعودية", 
    "1322.SR": "المطاحن الأولى", "1201.SR": "تكوين", "1202.SR": "مبكو", "1210.SR": "بي سي آي",
    
    # === الأسمنتات ===
    "3030.SR": "أسمنت السعودية", "3040.SR": "أسمنت القصيم", "3050.SR": "أسمنت الجنوب", 
    "3060.SR": "أسمنت ينبع", "3010.SR": "أسمنت العربية", "3020.SR": "أسمنت اليمامة", 
    "3080.SR": "أسمنت الشرقية", "3090.SR": "أسمنت تبوك", "3091.SR": "أسمنت الجوف", 
    "3001.SR": "أسمنت حائل", "3002.SR": "أسمنت نجران", "3003.SR": "أسمنت المدينة", 
    "3004.SR": "أسمنت الشمالية", "3005.SR": "أسمنت أم القرى", "3007.SR": "زهرة الواحة", 
    "3008.SR": "الكثيري",
    
    # === البنوك ===
    "1120.SR": "الراجحي", "1180.SR": "الأهلي", "1010.SR": "الرياض", "1150.SR": "الإنماء", 
    "1060.SR": "الأول", "1140.SR": "البلاد", "1030.SR": "الاستثمار", "1020.SR": "الجزيرة", 
    "1080.SR": "العربي", "1050.SR": "الفرنسي", "1111.SR": "تداول", "1183.SR": "سهل", 
    "4081.SR": "النايفات", "1182.SR": "أملاك", "4280.SR": "المملكة",
    
    # === الاتصالات وتقنية المعلومات (مصححة) ===
    "7010.SR": "STC", "7020.SR": "موبايلي", "7030.SR": "زين", "7040.SR": "عذيب", 
    "7202.SR": "سلوشنز", # تم التصحيح
    "7203.SR": "علم",     # تم التصحيح
    "7200.SR": "المعمر (MIS)", # تم التصحيح
    "7201.SR": "بحر العرب", "7204.SR": "توبي",
    
    # === التجزئة والأغذية ===
    "4190.SR": "جرير", "4001.SR": "العثيم", "4003.SR": "إكسترا", "4164.SR": "النهدي", 
    "2280.SR": "المراعي", "2270.SR": "سدافكو", "6002.SR": "هرفي", "6004.SR": "كاتريون", 
    "6010.SR": "نادك", "6020.SR": "جاكو", "6040.SR": "تبوك الزراعية", "6050.SR": "الأسماك", 
    "6060.SR": "الشرقية الزراعية", "6070.SR": "الجوف", "6090.SR": "جازادكو", "1810.SR": "سيرا", 
    "1830.SR": "وقت اللياقة", "4161.SR": "بن داود", "4162.SR": "المنجم", "4163.SR": "الدواء", 
    "4006.SR": "المزرعة", "4061.SR": "أنعام", "4100.SR": "مكة", "4170.SR": "شمس", 
    "4180.SR": "فتيحي", "6001.SR": "حلواني", "6012.SR": "ريدان", "4191.SR": "السيف غاليري",
    
    # === الصحة والتأمين ===
    "4002.SR": "المواساة", "4004.SR": "دلة", "4007.SR": "الحمادي", "4009.SR": "الألماني", 
    "4013.SR": "سليمان الحبيب", "4015.SR": "جمجوم فارما", "8010.SR": "التعاونية", "8210.SR": "بوبا", 
    "8230.SR": "الراجحي تكافل", "8012.SR": "جزيرة تكافل", "8020.SR": "ملاذ", "8030.SR": "ميدغلف", 
    "8040.SR": "أليانز", "8050.SR": "سلامة", "8060.SR": "ولاء", "8070.SR": "الدرع العربي", 
    "8100.SR": "سايكو", "8120.SR": "اتحاد الخليج", "8150.SR": "أسيج", "8160.SR": "التأمين العربية", 
    "8170.SR": "الاتحاد", "8180.SR": "الصقر", "8190.SR": "المتحدة", "8200.SR": "إعادة", 
    "8240.SR": "تشب", "8250.SR": "جي جي", "8260.SR": "الخليجية", "8270.SR": "بروج", 
    "8280.SR": "العالمية", "8300.SR": "الوطنية", "8310.SR": "أمانة", "8311.SR": "عناية",
    
    # === العقار والريت ===
    "4300.SR": "دار الأركان", "4250.SR": "جبل عمر", "4220.SR": "إعمار", "4321.SR": "سينومي سنترز", 
    "4230.SR": "البحر الأحمر", "4090.SR": "طيبة", "4150.SR": "التعمير", "4310.SR": "مدينة المعرفة", 
    "4320.SR": "الأندلس", "4322.SR": "رتال", "4323.SR": "سمو", "4330.SR": "الرياض ريت", 
    "4340.SR": "الراجحي ريت", "4342.SR": "جدوى ريت السعودية", "4344.SR": "سدكو كابيتال ريت",
    
    # === السلع الرأسمالية والخدمات ===
    "1212.SR": "أسترا", "1214.SR": "شاكر", "1302.SR": "بوان", "1303.SR": "الصناعات الكهربائية", 
    "1831.SR": "مهارة", "2040.SR": "الخزف", "2110.SR": "الكابلات", "4020.SR": "العقارية", 
    "4040.SR": "الجماعي", "4050.SR": "ساسكو", "4260.SR": "بدجت", "4261.SR": "ذيب", 
    "4031.SR": "الخدمات الأرضية", "4263.SR": "سال", "4142.SR": "الرياض للحديد", "4072.SR": "مجموعة MBC",
    
    # === المؤشر العام ===
    "^TASI.SR": "المؤشر العام (TASI)"
}

# --- 4. الدوال الفنية (بما فيها منطق الصندوق الجديد) ---

def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

# دالة كشف الصناديق (محاكاة Pine Script)
def check_bullish_box(df, atr_series):
    in_series = False
    is_bullish = False
    start_open = 0.0
    end_close = 0.0
    start_index = 0
    found_boxes = []
    
    lookback_slice = df.iloc[-100:].copy() if len(df) > 100 else df.copy()
    atr_slice = atr_series.iloc[-100:] if len(df) > 100 else atr_series
    prices = lookback_slice.reset_index()
    atrs = atr_slice.values
    
    for i in range(len(prices)):
        row = prices.iloc[i]
        close = row['Close']
        open_p = row['Open']
        
        is_green = close > open_p
        is_red = close < open_p
        
        current_atr = atrs[i]
        if np.isnan(current_atr): continue
        
        if not in_series:
            if is_green:
                in_series = True
                is_bullish = True
                start_open = open_p
                start_index = i
            elif is_red:
                in_series = True
                is_bullish = False
                start_open = open_p
        elif in_series:
            if is_bullish and is_green:
                end_close = close
            elif not is_bullish and is_red:
                end_close = close
            elif (is_bullish and is_red) or (not is_bullish and is_green):
                final_close = end_close if end_close != 0 else start_open
                price_move = abs(final_close - start_open)
                threshold = current_atr * ATR_MULT
                
                if price_move >= threshold:
                    if is_bullish:
                        days_ago = len(prices) - i
                        if days_ago <= BOX_LOOKBACK:
                            found_boxes.append({
                                "Price": close,
                                "Box_Top": max(start_open, final_close),
                                "Box_Bottom": min(start_open, final_close),
                                "Days_Ago": days_ago
                            })
                in_series = True
                is_bullish = is_green
                start_open = open_p
                end_close = close
                start_index = i

    return found_boxes

def calculate_indicators(df):
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['EMA'] = df['Close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Change'] = df['Close'].pct_change() * 100
    df['ATR'] = calculate_atr(df, ATR_LENGTH)
    return df

# --- 5. المنطق والتشغيل ---
st.title("📊 محلل السوق السعودي (القائمة الكاملة)")

if 'data' not in st.session_state: st.session_state['data'] = []
if 'signals' not in st.session_state: st.session_state['signals'] = []
if 'boxes' not in st.session_state: st.session_state['boxes'] = [] 
if 'history' not in st.session_state: st.session_state['history'] = {}

if st.button("🚀 تشغيل المسح الشامل (All Tickers)"):
    st.session_state['data'] = []
    st.session_state['signals'] = []
    st.session_state['boxes'] = []
    st.session_state['history'] = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    tickers_list = list(TICKERS.keys())
    
    chunk_size = 50
    total_tickers = len(tickers_list)
    
    for i in range(0, total_tickers, chunk_size):
        chunk = tickers_list[i:i + chunk_size]
        status_text.text(f"جاري تحليل الدفعة {i//chunk_size + 1}...")
        
        try:
            raw_data = yf.download(chunk, period="1y", interval="1d", group_by='ticker', auto_adjust=False, threads=True, progress=False)
            
            if not raw_data.empty:
                for symbol in chunk:
                    try:
                        name = TICKERS[symbol]
                        try: df = raw_data[symbol].copy()
                        except: continue

                        col = 'Close' if 'Close' in df.columns else 'Adj Close'
                        if col in df.columns:
                            df = df.rename(columns={col: 'Close'})
                            df = df.dropna()
                            if len(df) > 60:
                                df = calculate_indicators(df)
                                last_row = df.iloc[-1]
                                
                                st.session_state['history'][name] = df
                                
                                st.session_state['data'].append({
                                    "Name": name, "Symbol": symbol, "Price": last_row['Close'],
                                    "Change": last_row['Change'], "RSI": last_row['RSI'],
                                    "MACD": last_row['MACD']
                                })
                                
                                found_boxes = check_bullish_box(df, df['ATR'])
                                if found_boxes:
                                    latest_box = found_boxes[-1]
                                    st.session_state['boxes'].append({
                                        "الاسم": name, "السعر": last_row['Close'],
                                        "قمة الصندوق": latest_box['Box_Top'],
                                        "قاع الصندوق": latest_box['Box_Bottom'],
                                        "منذ (شمعة)": latest_box['Days_Ago']
                                    })

                                tail = df.tail(4)
                                if len(tail) == 4:
                                    rsi_break = False
                                    ema_break = False
                                    for idx in range(1, 4):
                                        if tail['RSI'].iloc[idx-1] <= 30 and tail['RSI'].iloc[idx] > 30: rsi_break = True
                                        if tail['Close'].iloc[idx-1] <= tail['EMA'].iloc[idx-1] and tail['Close'].iloc[idx] > tail['EMA'].iloc[idx]: ema_break = True
                                    
                                    if rsi_break and ema_break:
                                        macd_status = "✅" if last_row['MACD'] > last_row['Signal_Line'] else "⚠️"
                                        st.session_state['signals'].append({
                                            "الاسم": name, "السعر": last_row['Close'], "RSI": last_row['RSI'], "MACD": macd_status
                                        })
                    except: continue
        except: pass
        progress_bar.progress(min((i + chunk_size) / total_tickers, 1.0))
        
    progress_bar.empty()
    status_text.success("تم الانتهاء!")

# --- 6. العرض ---
if st.session_state['data']:
    df_all = pd.DataFrame(st.session_state['data'])
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("عدد الشركات", len(df_all))
    k2.metric("فرص القناص", len(st.session_state['signals']))
    k3.metric("صناديق صاعدة 📦", len(st.session_state['boxes']))
    bullish = len(df_all[df_all['Change'] > 0])
    k4.metric("السوق أخضر", bullish)
    
    st.markdown("---")
    t1, t2, t3, t4 = st.tabs(["📦 كاشف الصناديق", "🎯 إشارات القناص", "📋 السوق الشامل", "📈 الشارت"])
    
    with t1:
        if st.session_state['boxes']:
            st.markdown(f"### شركات كونت 'صندوق صعودي' (Bullish Box)")
            df_boxes = pd.DataFrame(st.session_state['boxes'])
            df_boxes = df_boxes.sort_values(by="منذ (شمعة)", ascending=True)
            st.dataframe(df_boxes.style.format({"السعر": "{:.2f}", "قمة الصندوق": "{:.2f}", "قاع الصندوق": "{:.2f}"}).background_gradient(cmap='Blues', subset=['منذ (شمعة)']), use_container_width=True)
        else:
            st.info(f"لم يتم العثور على صناديق صعودية مكتملة.")

    with t2:
        if st.session_state['signals']:
            st.dataframe(pd.DataFrame(st.session_state['signals']), use_container_width=True)
        else:
            st.info("لا توجد إشارات RSI+EMA حالياً.")
            
    with t3:
        display_df = df_all.copy().rename(columns={"Name": "الاسم", "Price": "السعر", "Change": "التغير %", "RSI": f"RSI ({RSI_PERIOD})", "MACD": "MACD"})
        cols_to_show = ["الاسم", "السعر", "التغير %", f"RSI ({RSI_PERIOD})", "MACD"]
        st.dataframe(display_df[cols_to_show].style.format({"السعر": "{:.2f}", "التغير %": "{:.2f}%", f"RSI ({RSI_PERIOD})": "{:.2f}"}).background_gradient(cmap='RdYlGn', subset=['التغير %']), use_container_width=True, height=500)
        
    with t4:
        sel = st.selectbox("اختر سهم:", df_all['Name'].unique())
        if sel:
            df_chart = st.session_state['history'][sel]
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_width=[0.3, 0.7])
            fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name='Price'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA'], line=dict(color='orange'), name='EMA'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            box_res = check_bullish_box(df_chart, df_chart['ATR'])
            if box_res:
                latest = box_res[-1]
                if latest['Days_Ago'] <= 50:
                    fig.add_shape(type="rect", x0=df_chart.index[-latest['Days_Ago']-5], x1=df_chart.index[-latest['Days_Ago']], y0=latest['Box_Bottom'], y1=latest['Box_Top'], line=dict(color="green", width=2), fillcolor="rgba(0,255,0,0.1)", row=1, col=1)

            fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False, paper_bgcolor='#161b24', plot_bgcolor='#161b24')
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("اضغط زر التحديث للبدء.")
