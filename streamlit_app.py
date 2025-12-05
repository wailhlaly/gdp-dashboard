import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time

# --- 1. إعداد الصفحة وتصميم الوضع الليلي (Dark Mode) ---
st.set_page_config(page_title="TASI Dark Pro", layout="wide", initial_sidebar_state="collapsed")

# حقن CSS لتغيير الألوان بالكامل إلى نمط TradingView
st.markdown("""
<style>
    /* الخلفية العامة */
    .stApp {
        background-color: #131722;
        color: #d1d4dc;
    }
    
    /* الجداول */
    .stDataFrame {
        border: 1px solid #2a2e39;
    }
    div[data-testid="stDataFrame"] div[class*="css"] {
        background-color: #1e222d;
        color: white;
    }
    
    /* الأزرار */
    div.stButton > button {
        background-color: #2962ff;
        color: white;
        border: none;
        border-radius: 4px;
        font-weight: bold;
        padding: 0.5rem 1rem;
    }
    div.stButton > button:hover {
        background-color: #1e53e5;
        border: none;
        color: white;
    }
    
    /* البطاقات (Metrics) */
    div[data-testid="stMetric"] {
        background-color: #1e222d;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #2a2e39;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetricLabel"] {
        color: #787b86;
    }
    div[data-testid="stMetricValue"] {
        color: #d1d4dc;
    }
    
    /* العناوين */
    h1, h2, h3 {
        color: #d1d4dc !important;
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* التبويبات */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #131722;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e222d;
        border-radius: 4px;
        color: #d1d4dc;
        border: 1px solid #2a2e39;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2962ff !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. القائمة الكاملة (تضمين جميع الشركات لضمان المسح الشامل) ---
TICKERS = {
    # الطاقة والمواد الأساسية
    "2222.SR": "أرامكو", "2010.SR": "سابك", "1211.SR": "معادن", "2020.SR": "سابك للمغذيات", "2310.SR": "سبكيم",
    "2290.SR": "ينساب", "2060.SR": "التصنيع", "2330.SR": "المتقدمة", "2350.SR": "كيان", "2001.SR": "كيمانول",
    "2170.SR": "اللجين", "2380.SR": "رابغ", "2381.SR": "الحفر العربية", "2382.SR": "أديس", "4030.SR": "البحري",
    "4200.SR": "الدريس", "5110.SR": "الكهرباء", "2030.SR": "المصافي", "2080.SR": "الغاز", "2150.SR": "زجاج",
    "2180.SR": "فيبكو", "2200.SR": "أنابيب", "2210.SR": "نما", "2230.SR": "الكيميائية", "2240.SR": "الزامل",
    "2250.SR": "المجموعة", "2300.SR": "صناعة الورق", "2320.SR": "البابطين", "2340.SR": "العبداللطيف", "2360.SR": "الفخارية",
    "2370.SR": "مسك", "1301.SR": "أسلاك", "1304.SR": "اليمامة للحديد", "1320.SR": "أنابيب الشرق", "1321.SR": "أنابيب السعودية",
    
    # الأسمنتات
    "3010.SR": "أسمنت العربية", "3020.SR": "أسمنت اليمامة", "3030.SR": "أسمنت السعودية", "3040.SR": "أسمنت القصيم",
    "3050.SR": "أسمنت الجنوب", "3060.SR": "أسمنت ينبع", "3080.SR": "أسمنت الشرقية", "3090.SR": "أسمنت تبوك",
    "3091.SR": "أسمنت الجوف", "3001.SR": "أسمنت حائل", "3002.SR": "أسمنت نجران", "3003.SR": "أسمنت المدينة",
    "3004.SR": "أسمنت الشمالية", "3005.SR": "أسمنت أم القرى", "3007.SR": "زهرة الواحة", "3008.SR": "الكثيري",

    # البنوك والاستثمار
    "1120.SR": "الراجحي", "1180.SR": "الأهلي", "1010.SR": "الرياض", "1150.SR": "الإنماء", "1060.SR": "الأول",
    "1140.SR": "البلاد", "1030.SR": "الاستثمار", "1020.SR": "الجزيرة", "1080.SR": "العربي", "1050.SR": "الفرنسي",
    "1111.SR": "تداول", "1182.SR": "أملاك", "1183.SR": "سهل", "4081.SR": "النايفات", "4280.SR": "المملكة",

    # الاتصالات والتقنية
    "7010.SR": "STC", "7020.SR": "موبايلي", "7030.SR": "زين", "7040.SR": "عذيب", "7200.SR": "سلوشنز",
    "7201.SR": "بحر العرب", "7202.SR": "علم", "7203.SR": "توبي",

    # التجزئة، الأغذية، الخدمات
    "4190.SR": "جرير", "4001.SR": "العثيم", "4003.SR": "إكسترا", "4164.SR": "النهدي", "2280.SR": "المراعي",
    "2270.SR": "سدافكو", "6002.SR": "هرفي", "6004.SR": "التموين", "6010.SR": "نادك", "6020.SR": "جاكو",
    "6040.SR": "تبوك الزراعية", "6050.SR": "الأسماك", "6060.SR": "الشرقية الزراعية", "6070.SR": "الجوف",
    "6090.SR": "جازادكو", "1810.SR": "سيرا", "1820.SR": "الحكير", "1830.SR": "وقت اللياقة", "4260.SR": "بدجت",
    "4261.SR": "ذيب", "4262.SR": "لومي", "4031.SR": "الخدمات الأرضية", "4263.SR": "سال",

    # الصحة والتأمين
    "4002.SR": "المواساة", "4004.SR": "دلة", "4007.SR": "الحمادي", "4009.SR": "الألماني", "4013.SR": "سليمان الحبيب",
    "8010.SR": "التعاونية", "8210.SR": "بوبا", "8230.SR": "الراجحي تكافل", "8012.SR": "جزيرة تكافل", "8020.SR": "ملاذ",
    "8030.SR": "ميدغلف", "8040.SR": "أليانز", "8050.SR": "سلامة", "8060.SR": "ولاء", "8070.SR": "الدرع",
    "8100.SR": "سايكو", "8120.SR": "اتحاد الخليج", "8150.SR": "أسيج", "8160.SR": "التأمين العربية", "8170.SR": "الاتحاد",
    "8200.SR": "إعادة", "8250.SR": "جي جي", "8270.SR": "بروج", "8310.SR": "أمانة", "8311.SR": "عناية",

    # العقار والتطوير
    "4300.SR": "دار الأركان", "4250.SR": "جبل عمر", "4220.SR": "إعمار", "4321.SR": "المراكز", "4230.SR": "البحر الأحمر",
    "4090.SR": "طيبة", "4100.SR": "مكة", "4150.SR": "التعمير", "4310.SR": "مدينة المعرفة", "4320.SR": "الأندلس",
    "4322.SR": "رتال", "4323.SR": "سمو",
    
    # المؤشر
    "^TASI.SR": "المؤشر العام"
}

# --- 3. الإعدادات والدوال ---
RSI_PERIOD = 24
EMA_PERIOD = 8

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

# --- 4. واجهة التطبيق ---

# Header
c1, c2 = st.columns([1, 5])
with c1:
    st.image("https://cdn-icons-png.flaticon.com/512/3310/3310636.png", width=70)
with c2:
    st.title("الماسح الاحترافي (Dark Pro)")
    st.caption(f"عدد الشركات في قاعدة البيانات: {len(TICKERS)} | الإستراتيجية: RSI 30 Breakout + EMA 8")

# زر التشغيل الكبير
if st.button("تشغيل المسح الكامل للسوق (Scan All)", use_container_width=True):
    
    # تهيئة المتغيرات
    summary_data = []
    signals_data = []
    
    # واجهة التقدم
    progress_text = st.empty()
    bar = st.progress(0)
    
    # تحضير القائمة والتقسيم للدفعات
    tickers_list = list(TICKERS.keys())
    total_stocks = len(tickers_list)
    chunk_size = 50 # 50 سهم في كل دفعة
    
    for i in range(0, total_stocks, chunk_size):
        chunk = tickers_list[i:i + chunk_size]
        
        # تحديث النص
        progress_text.markdown(f"**⏳ جاري معالجة الدفعة {i//chunk_size + 1} (الأسهم من {i} إلى {min(i+chunk_size, total_stocks)})...**")
        
        try:
            # تحميل الدفعة
            data = yf.download(chunk, period="6mo", interval="1d", group_by='ticker', auto_adjust=False, threads=True, progress=False)
            
            if not data.empty:
                for symbol in chunk:
                    try:
                        # التأكد من وجود البيانات
                        try:
                            df = data[symbol].copy()
                        except KeyError:
                            continue

                        # توحيد اسم العمود
                        col_name = 'Close' if 'Close' in df.columns else 'Adj Close'
                        if col_name not in df.columns: continue
                        
                        series = df[col_name].dropna()
                        
                        if len(series) > 60:
                            # الحسابات الفنية
                            rsi = calculate_rsi_rma(series, RSI_PERIOD)
                            ema = calculate_ema(series, EMA_PERIOD)
                            
                            last_price = series.iloc[-1]
                            last_rsi = rsi.iloc[-1]
                            
                            # إضافة للملخص العام
                            if not np.isnan(last_rsi):
                                summary_data.append({
                                    "الاسم": TICKERS.get(symbol, symbol),
                                    "الرمز": symbol,
                                    "السعر": last_price,
                                    "RSI": last_rsi
                                })
                            
                            # استراتيجية القناص (آخر 3 أيام)
                            # ننشئ داتا فريم صغير للفحص
                            check_df = pd.DataFrame({
                                'RSI': rsi.tail(4),
                                'Price': series.tail(4),
                                'EMA': ema.tail(4)
                            })
                            
                            if len(check_df) == 4:
                                rsi_cross = False
                                ema_cross = False
                                
                                # فحص الأيام الثلاثة الأخيرة
                                for idx in range(1, 4):
                                    # اختراق RSI 30
                                    if check_df['RSI'].iloc[idx-1] <= 30 and check_df['RSI'].iloc[idx] > 30:
                                        rsi_cross = True
                                    # اختراق EMA 8
                                    if check_df['Price'].iloc[idx-1] <= check_df['EMA'].iloc[idx-1] and check_df['Price'].iloc[idx] > check_df['EMA'].iloc[idx]:
                                        ema_cross = True
                                
                                if rsi_cross and ema_cross:
                                    signals_data.append({
                                        "الاسم": TICKERS.get(symbol, symbol),
                                        "السعر": last_price,
                                        "RSI": last_rsi,
                                        "الحالة": "BUY SIGNAL 🚀"
                                    })
                    except Exception:
                        continue
        except Exception:
            pass
        
        # تحديث الشريط
        bar.progress(min((i + chunk_size) / total_stocks, 1.0))
        time.sleep(0.2) # استراحة قصيرة جداً

    bar.empty()
    progress_text.success("✅ اكتمل المسح!")
    
    # حفظ النتائج
    st.session_state['summary'] = summary_data
    st.session_state['signals'] = signals_data

# --- 5. عرض النتائج (Dashboard) ---

if 'summary' in st.session_state and st.session_state['summary']:
    
    # عدادات المعلومات (Stats)
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("الشركات التي تم فحصها", len(st.session_state['summary']))
    kpi2.metric("الفرص الذهبية (Sniper)", len(st.session_state['signals']), delta_color="normal")
    
    avg_rsi = np.mean([d['RSI'] for d in st.session_state['summary']])
    kpi3.metric("متوسط RSI للسوق", f"{avg_rsi:.2f}")

    st.markdown("---")

    # التبويبات
    tab1, tab2 = st.tabs(["💎 الفرص الذهبية", "📊 السوق بالكامل"])
    
    # --- TAB 1: الفرص ---
    with tab1:
        if st.session_state['signals']:
            st.markdown("### أسهم حققت شروط الدخول (RSI Breakout + EMA Cross)")
            df_signals = pd.DataFrame(st.session_state['signals'])
            
            # تنسيق الجدول الاحترافي
            st.dataframe(
                df_signals.style.format({"السعر": "{:.2f}", "RSI": "{:.2f}"})
                .set_properties(**{
                    'background-color': '#1e222d',
                    'color': '#00ff00', # أخضر فسفوري
                    'font-weight': 'bold',
                    'border': '1px solid #333'
                }),
                use_container_width=True
            )
        else:
            st.info("لم يتم العثور على فرص تطابق الشروط بدقة في آخر 3 أيام.")

    # --- TAB 2: السوق الكامل ---
    with tab2:
        st.markdown("### نظرة شاملة (مرتبة حسب RSI)")
        df_all = pd.DataFrame(st.session_state['summary'])
        df_all = df_all.sort_values(by="RSI", ascending=False)
        
        # دالة تلوين متقدمة (Dark Theme Style)
        def style_dark_table(val):
            color = '#d1d4dc' # رمادي فاتح افتراضي
            weight = 'normal'
            
            if val >= 70:
                color = '#ff5252' # أحمر فاتح
                weight = 'bold'
            elif val <= 30:
                color = '#4caf50' # أخضر فاتح
                weight = 'bold'
            
            return f'color: {color}; font-weight: {weight};'

        st.dataframe(
            df_all.style.map(style_dark_table, subset=['RSI'])
            .format({"السعر": "{:.2f}", "RSI": "{:.2f}"}),
            use_container_width=True,
            height=600
        )

else:
    st.info("اضغط زر 'تشغيل المسح' للبدء.")
