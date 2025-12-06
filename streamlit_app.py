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

# القائمة الشاملة (عينة كبيرة)
TICKERS = {
    # طاقة
    "2222.SR": "أرامكو", "2030.SR": "المصافي", "4200.SR": "الدريس", "5110.SR": "الكهرباء", "4030.SR": "البحري",
    # مواد أساسية
    "2010.SR": "سابك", "1211.SR": "معادن", "2020.SR": "سابك للمغذيات", "2310.SR": "سبكيم", "2060.SR": "التصنيع",
    "2290.SR": "ينساب", "2350.SR": "كيان", "2380.SR": "رابغ", "2381.SR": "الحفر العربية",
    # بنوك
    "1120.SR": "الراجحي", "1180.SR": "الأهلي", "1010.SR": "الرياض", "1150.SR": "الإنماء", "1060.SR": "الأول",
    "1140.SR": "البلاد", "1030.SR": "الاستثمار", "1020.SR": "الجزيرة", "1050.SR": "الفرنسي",
    # اتصالات
    "7010.SR": "STC", "7020.SR": "موبايلي", "7030.SR": "زين", "7202.SR": "علم", "7200.SR": "سلوشنز",
    # تجزئة وخدمات
    "4190.SR": "جرير", "4001.SR": "العثيم", "4003.SR": "إكسترا", "4164.SR": "النهدي", "2280.SR": "المراعي",
    "4002.SR": "المواساة", "4013.SR": "سليمان الحبيب", "4261.SR": "ذيب", "1810.SR": "سيرا", "1830.SR": "وقت اللياقة",
    # تأمين
    "8010.SR": "التعاونية", "8210.SR": "بوبا", "8230.SR": "الراجحي تكافل",
    # عقار
    "4300.SR": "دار الأركان", "4250.SR": "جبل عمر", "4220.SR": "إعمار", "4090.SR": "طيبة",
    # مؤشر
    "^TASI.SR": "المؤشر العام"
}

# --- 3. الدوال الفنية (بما فيها منطق الصندوق الجديد) ---

def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

# 🔥 هذه الدالة تحاكي كود Pine Script بدقة
def check_bullish_box(df, atr_series):
    # المتغيرات لمحاكاة الحالة (State Machine)
    in_series = False
    is_bullish = False
    start_open = 0.0
    end_close = 0.0
    start_index = 0
    
    found_boxes = [] # لتخزين الصناديق المكتشفة
    
    # نحتاج للتكرار عبر البيانات (Loop)
    # لتحسين الأداء، نأخذ آخر 100 شمعة فقط للفحص
    lookback_slice = df.iloc[-100:].copy() if len(df) > 100 else df.copy()
    atr_slice = atr_series.iloc[-100:] if len(df) > 100 else atr_series
    
    # إعادة تعيين الفهرس للتكرار بالأرقام
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
        
        # 1. بداية سلسلة جديدة
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
        
        # 2. استمرار السلسلة
        elif in_series:
            # نحن في سلسلة خضراء وجاءت شمعة خضراء -> تمديد
            if is_bullish and is_green:
                end_close = close
            
            # نحن في سلسلة حمراء وجاءت شمعة حمراء -> تمديد
            elif not is_bullish and is_red:
                end_close = close
                
            # 3. انقطاع السلسلة (نهاية الصندوق المحتمل)
            elif (is_bullish and is_red) or (not is_bullish and is_green):
                # حساب حجم الحركة
                # ملاحظة: في Pine Script يتم استخدام آخر endClose تم تسجيله
                final_close = end_close if end_close != 0 else start_open # حماية
                price_move = abs(final_close - start_open)
                threshold = current_atr * ATR_MULT
                
                # التحقق هل هو صندوق صحيح؟
                if price_move >= threshold:
                    # نحن مهتمون فقط بالصناديق الصعودية (Bullish)
                    if is_bullish:
                        # التأكد أن الصندوق انتهى حديثاً (ضمن النطاق المحدد من المستخدم)
                        days_ago = len(prices) - i
                        if days_ago <= BOX_LOOKBACK:
                            found_boxes.append({
                                "Price": close,
                                "Box_Top": max(start_open, final_close),
                                "Box_Bottom": min(start_open, final_close),
                                "Days_Ago": days_ago
                            })
                
                # إعادة تعيين الحالة لبدء سلسلة جديدة فوراً
                in_series = True
                is_bullish = is_green # الشمعة الحالية تحدد نوع السلسلة الجديدة
                start_open = open_p
                end_close = close
                start_index = i

    return found_boxes

def calculate_indicators(df):
    # RSI & EMA & MACD (كما في الكود السابق)
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
    
    # ATR للصناديق
    df['ATR'] = calculate_atr(df, ATR_LENGTH)
    
    return df

# --- 4. المنطق والتشغيل ---
st.title("📊 محلل السوق السعودي (نسخة الصناديق الذكية)")

if 'data' not in st.session_state: st.session_state['data'] = []
if 'signals' not in st.session_state: st.session_state['signals'] = []
if 'boxes' not in st.session_state: st.session_state['boxes'] = [] # قائمة جديدة للصناديق
if 'history' not in st.session_state: st.session_state['history'] = {}

if st.button("🚀 تشغيل المسح (RSI + Boxes)"):
    st.session_state['data'] = []
    st.session_state['signals'] = []
    st.session_state['boxes'] = []
    st.session_state['history'] = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    tickers_list = list(TICKERS.keys())
    
    # نظام الدفعات
    chunk_size = 50
    total_tickers = len(tickers_list)
    
    for i in range(0, total_tickers, chunk_size):
        chunk = tickers_list[i:i + chunk_size]
        status_text.text(f"جاري تحليل الدفعة {i//chunk_size + 1}...")
        
        try:
            # نحتاج بيانات أكثر (سنة) لضمان دقة ATR
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
                                
                                # 1. تخزين البيانات العامة
                                st.session_state['data'].append({
                                    "Name": name, "Symbol": symbol, "Price": last_row['Close'],
                                    "Change": last_row['Change'], "RSI": last_row['RSI'],
                                    "MACD": last_row['MACD']
                                })
                                
                                # 2. كشف الصناديق (الميزة الجديدة)
                                # نمرر البيانات وسلسلة ATR
                                found_boxes = check_bullish_box(df, df['ATR'])
                                if found_boxes:
                                    # نأخذ أحدث صندوق تم العثور عليه
                                    latest_box = found_boxes[-1]
                                    st.session_state['boxes'].append({
                                        "الاسم": name,
                                        "السعر": last_row['Close'],
                                        "قمة الصندوق": latest_box['Box_Top'],
                                        "قاع الصندوق": latest_box['Box_Bottom'],
                                        "منذ (شمعة)": latest_box['Days_Ago']
                                    })

                                # 3. إشارات القناص القديمة
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

# --- 5. العرض ---
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
    
    # --- TAB 1: الصناديق (الميزة الجديدة) ---
    with t1:
        if st.session_state['boxes']:
            st.markdown(f"### شركات كونت 'صندوق صعودي' (Bullish Box) في آخر {BOX_LOOKBACK} شمعة")
            st.caption("يعتمد على مؤشر: حركة سعرية صاعدة > (1.5 * ATR)")
            
            df_boxes = pd.DataFrame(st.session_state['boxes'])
            # ترتيب حسب الأحدث
            df_boxes = df_boxes.sort_values(by="منذ (شمعة)", ascending=True)
            
            st.dataframe(
                df_boxes.style.format({"السعر": "{:.2f}", "قمة الصندوق": "{:.2f}", "قاع الصندوق": "{:.2f}"})
                .background_gradient(cmap='Blues', subset=['منذ (شمعة)']),
                use_container_width=True
            )
        else:
            st.info(f"لم يتم العثور على صناديق صعودية مكتملة في آخر {BOX_LOOKBACK} يوم.")

    # --- TAB 2: القناص ---
    with t2:
        if st.session_state['signals']:
            st.dataframe(pd.DataFrame(st.session_state['signals']), use_container_width=True)
        else:
            st.info("لا توجد إشارات RSI+EMA حالياً.")
            
    # --- TAB 3: السوق ---
    with t3:
        display_df = df_all.copy().rename(columns={"Name": "الاسم", "Price": "السعر", "Change": "التغير %", "RSI": f"RSI ({RSI_PERIOD})", "MACD": "MACD"})
        cols_to_show = ["الاسم", "السعر", "التغير %", f"RSI ({RSI_PERIOD})", "MACD"]
        st.dataframe(
            display_df[cols_to_show].style.format({"السعر": "{:.2f}", "التغير %": "{:.2f}%", f"RSI ({RSI_PERIOD})": "{:.2f}"})
            .background_gradient(cmap='RdYlGn', subset=['التغير %']),
            use_container_width=True, height=500
        )
        
    # --- TAB 4: الشارت ---
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
            
            # محاولة رسم الصندوق الأخير على الشارت إذا وجد
            # (اختياري لزيادة الجمالية)
            last_atr = df_chart['ATR'].iloc[-1]
            box_res = check_bullish_box(df_chart, df_chart['ATR'])
            if box_res:
                latest = box_res[-1]
                if latest['Days_Ago'] <= 50: # نرسمه فقط إذا كان قريباً
                    # رسم مستطيل يمثل الصندوق
                    fig.add_shape(type="rect",
                        x0=df_chart.index[-latest['Days_Ago']-5], x1=df_chart.index[-latest['Days_Ago']], 
                        y0=latest['Box_Bottom'], y1=latest['Box_Top'],
                        line=dict(color="green", width=2), fillcolor="rgba(0,255,0,0.1)",
                        row=1, col=1
                    )

            fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False, paper_bgcolor='#161b24', plot_bgcolor='#161b24')
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("اضغط زر التحديث للبدء.")
