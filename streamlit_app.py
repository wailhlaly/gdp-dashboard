import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time

# --- 1. إعداد الصفحة وتنسيق CSS ---
st.set_page_config(page_title="TASI Pro Dashboard", layout="wide", initial_sidebar_state="expanded")

# تخصيص المظهر (CSS)
st.markdown("""
<style>
    /* تحسين الخطوط والعناوين */
    h1 { color: #1f77b4; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    h2, h3 { color: #333; }
    
    /* تنسيق البطاقات */
    div[data-testid="stMetric"] {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    
    /* تنسيق الجداول */
    .stDataFrame { border: 1px solid #e6e6e6; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# --- 2. الثوابت والقوائم ---
RSI_PERIOD = 24
EMA_PERIOD = 8

TICKERS = {
    # (نفس القائمة الشاملة - سأضع عينة كبيرة لتعمل الكود، يمكنك إضافة الباقي كما كان)
    "^TASI.SR": "المؤشر العام", "1120.SR": "الراجحي", "1180.SR": "الأهلي", "2222.SR": "أرامكو", "2010.SR": "سابك",
    "7010.SR": "STC", "1150.SR": "الإنماء", "1211.SR": "معادن", "4030.SR": "البحري", "4200.SR": "الدريس",
    "4190.SR": "جرير", "2020.SR": "سابك للمغذيات", "2280.SR": "المراعي", "4002.SR": "المواساة", "8010.SR": "التعاونية",
    "1010.SR": "الرياض", "1060.SR": "الأول", "1140.SR": "البلاد", "2350.SR": "كيان", "2310.SR": "سبكيم",
    "4250.SR": "جبل عمر", "4300.SR": "دار الأركان", "4090.SR": "طيبة", "4321.SR": "المراكز", "4220.SR": "إعمار",
    "7020.SR": "موبايلي", "7030.SR": "زين", "7202.SR": "علم", "7200.SR": "سلوشنز", "4013.SR": "سليمان الحبيب",
    # ... (يمكنك لصق القائمة الطويلة جداً هنا)
}

# --- 3. الدوال الفنية ---
def calculate_rsi_rma(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

# --- 4. إدارة الذاكرة (Session State) ---
if 'summary' not in st.session_state: st.session_state['summary'] = []
if 'signals' not in st.session_state: st.session_state['signals'] = []
if 'market_data' not in st.session_state: st.session_state['market_data'] = {}
if 'last_update' not in st.session_state: st.session_state['last_update'] = None

# --- 5. القائمة الجانبية (Sidebar) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3310/3310636.png", width=80)
    st.title("لوحة التحكم")
    st.markdown("---")
    
    st.write("🔧 **إعدادات الفحص:**")
    st.caption(f"RSI Period: {RSI_PERIOD}")
    st.caption(f"EMA Period: {EMA_PERIOD}")
    
    scan_btn = st.button("🚀 تشغيل المسح الذكي", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.info("""
    **استراتيجية القناص:**
    1. اختراق RSI لمستوى 30 صعوداً.
    2. اختراق السعر لمتوسط EMA 8.
    *يجب حدوث الشرطين خلال آخر 3 أيام.*
    """)
    
    if st.session_state['last_update']:
        st.success(f"آخر تحديث:\n{st.session_state['last_update']}")

# --- 6. المنطق الرئيسي (Engine) ---
if scan_btn:
    # تصفير البيانات
    st.session_state['summary'] = []
    st.session_state['signals'] = []
    st.session_state['market_data'] = {}
    
    # واجهة التحميل
    progress_placeholder = st.empty()
    bar = st.progress(0)
    
    tickers_list = list(TICKERS.keys())
    total = len(tickers_list)
    chunk_size = 50
    
    for i in range(0, total, chunk_size):
        chunk = tickers_list[i:i + chunk_size]
        progress_placeholder.info(f"⏳ جاري تحليل الدفعة {i//chunk_size + 1} ({min(i+chunk_size, total)}/{total})...")
        
        try:
            data = yf.download(chunk, period="6mo", interval="1d", group_by='ticker', auto_adjust=False, threads=True, progress=False)
            
            if not data.empty:
                for symbol in chunk:
                    try:
                        name = TICKERS.get(symbol, symbol)
                        try: df = data[symbol].copy()
                        except: continue

                        col = 'Close' if 'Close' in df.columns else 'Adj Close'
                        if col in df.columns:
                            series = df[col].dropna()
                            if len(series) > 50:
                                # الحسابات
                                df['RSI'] = calculate_rsi_rma(series, RSI_PERIOD)
                                df['EMA8'] = calculate_ema(series, EMA_PERIOD)
                                df['Close_Clean'] = series
                                
                                st.session_state['market_data'][name] = df
                                
                                # الملخص
                                last_price = series.iloc[-1]
                                last_rsi = df['RSI'].iloc[-1]
                                
                                if not np.isnan(last_rsi):
                                    st.session_state['summary'].append({
                                        "الاسم": name, "الرمز": symbol, "السعر": last_price, f"RSI": last_rsi
                                    })
                                
                                # منطق الإشارات (آخر 3 أيام)
                                tail = df.tail(4)
                                if len(tail) == 4:
                                    rsi_break = False
                                    ema_break = False
                                    for idx in range(1, 4):
                                        if tail['RSI'].iloc[idx-1] <= 30 and tail['RSI'].iloc[idx] > 30: rsi_break = True
                                        if tail['Close_Clean'].iloc[idx-1] <= tail['EMA8'].iloc[idx-1] and tail['Close_Clean'].iloc[idx] > tail['EMA8'].iloc[idx]: ema_break = True
                                    
                                    if rsi_break and ema_break:
                                        st.session_state['signals'].append({
                                            "الاسم": name, "السعر": last_price, "RSI": last_rsi, "الحالة": "✅ اختراق مزدوج"
                                        })
                    except: continue
        except: pass
        
        bar.progress(min((i + chunk_size) / total, 1.0))
        time.sleep(0.1)
    
    bar.empty()
    progress_placeholder.empty()
    st.session_state['last_update'] = time.strftime("%H:%M:%S")

# --- 7. لوحة العرض (Dashboard Layout) ---

# العنوان الرئيسي
st.title("📊 محلل السوق السعودي الاحترافي")

# عدادات المعلومات
col1, col2, col3 = st.columns(3)
col1.metric("عدد الشركات المحللة", len(st.session_state['summary']))
col2.metric("الفرص الذهبية (Sniper)", len(st.session_state['signals']))
market_trend = "غير محدد"
if st.session_state['summary']:
    avg_rsi = np.mean([d['RSI'] for d in st.session_state['summary']])
    col3.metric("متوسط RSI للسوق", f"{avg_rsi:.2f}", delta="منطقة تشبع" if avg_rsi > 70 else "طبيعي")

st.markdown("---")

# التبويبات (Tabs) للتنظيم
tab_signals, tab_market, tab_details = st.tabs(["🎯 الفرص الذهبية (Signals)", "📋 شامل السوق", "🔍 المحلل الفني"])

# --- TAB 1: الفرص الذهبية ---
with tab_signals:
    if st.session_state['signals']:
        st.success(f"تم العثور على {len(st.session_state['signals'])} شركة حققت شروط الاختراق في آخر 3 أيام!")
        df_sig = pd.DataFrame(st.session_state['signals'])
        
        # تنسيق خاص
        st.dataframe(
            df_sig.style.format({"السعر": "{:.2f}", "RSI": "{:.2f}"})
            .set_properties(**{'background-color': '#e6fffa', 'color': 'black', 'border-color': 'white'}),
            use_container_width=True
        )
    else:
        if st.session_state['summary']:
            st.warning("لا توجد أسهم حققت شروط الاختراق المزدوج (RSI 30 + EMA 8) حالياً.")
        else:
            st.info("اضغط زر التشغيل في القائمة الجانبية للبدء.")

# --- TAB 2: شامل السوق ---
with tab_market:
    if st.session_state['summary']:
        st.write("ترتيب السوق حسب التشبع:")
        df_all = pd.DataFrame(st.session_state['summary']).sort_values(by="RSI", ascending=False)
        
        def color_rsi_grad(val):
            if val >= 70: return 'background-color: #ffcccc; color: red; font-weight: bold'
            elif val <= 30: return 'background-color: #ccffcc; color: green; font-weight: bold'
            return ''

        st.dataframe(
            df_all.style.map(color_rsi_grad, subset=['RSI'])
            .format({"السعر": "{:.2f}", "RSI": "{:.2f}"}),
            use_container_width=True,
            height=600
        )

# --- TAB 3: التفاصيل التفاعلية ---
with tab_details:
    st.subheader("فحص الشارت الرقمي لسهم محدد")
    
    if st.session_state['summary']:
        names = sorted([d['الاسم'] for d in st.session_state['summary']])
        selected = st.selectbox("ابحث عن شركة:", names)
        
        if selected:
            df_chart = st.session_state['market_data'][selected].tail(14).sort_index(ascending=False)
            
            # تجهيز العرض
            display = df_chart[['Close_Clean', 'EMA8', 'RSI']].rename(columns={'Close_Clean': 'Close'})
            
            # دالة تلوين متقدمة توضح التقاطعات
            def style_chart(row):
                styles = [''] * 3
                # Close vs EMA
                if row['Close'] > row['EMA8']: styles[0] = 'color: green; font-weight: bold' # Close
                else: styles[0] = 'color: red'
                
                # RSI
                if row['RSI'] <= 30: styles[2] = 'background-color: #ccffcc; color: green; font-weight: bold'
                elif row['RSI'] >= 70: styles[2] = 'background-color: #ffcccc; color: red; font-weight: bold'
                return styles

            st.write(f"سجل بيانات **{selected}** (آخر 14 يوم):")
            st.dataframe(
                display.style.apply(style_chart, axis=1).format("{:.2f}"),
                use_container_width=True
            )
    else:
        st.info("يرجى تشغيل المسح أولاً.")

