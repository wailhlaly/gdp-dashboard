import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# --- إعداد الصفحة ---
st.set_page_config(page_title="TASI Pro Screener", layout="wide")
st.title("📊 ماسح السوق السعودي الشامل (TASI All-In-One)")

# --- الإعدادات ---
RSI_PERIOD = 24

# --- قائمة الأسهم الشاملة (الأكثر نشاطاً وقيادية) ---
TICKERS = {
    # --- المؤشرات ---
    "^TASI.SR": "المؤشر العام",
    
    # --- الطاقة والمرافق ---
    "2222.SR": "أرامكو",
    "2030.SR": "المصافي",
    "4200.SR": "الدريس",
    "5110.SR": "الكهرباء",
    "2080.SR": "الغاز",
    "4030.SR": "البحري",
    
    # --- المواد الأساسية (بتروكيماويات ومعادن) ---
    "2010.SR": "سابك",
    "1211.SR": "معادن",
    "2020.SR": "سابك للمغذيات",
    "2310.SR": "سبكيم",
    "2060.SR": "التصنيع",
    "2290.SR": "ينساب",
    "2001.SR": "كيمانول",
    "2170.SR": "اللجين",
    "2330.SR": "المتقدمة",
    "2350.SR": "كيان",
    "2380.SR": "رابغ",
    
    # --- البنوك والخدمات المالية ---
    "1120.SR": "الراجحي",
    "1180.SR": "الأهلي",
    "1010.SR": "الرياض",
    "1150.SR": "الإنماء",
    "1060.SR": "الأول (ساب)",
    "1140.SR": "البلاد",
    "1030.SR": "الاستثمار",
    "1020.SR": "الجزيرة",
    "1080.SR": "العربي",
    "1050.SR": "الفرنسي",
    "1183.SR": "سهل", # أملاك سابقاً أو شركات التمويل
    "1111.SR": "تداول",
    
    # --- الاتصالات ---
    "7010.SR": "STC",
    "7020.SR": "موبايلي",
    "7030.SR": "زين",
    "7200.SR": "سلوشنز",
    "7040.SR": "عذيب",
    
    # --- الأسمنت ---
    "3030.SR": "أسمنت السعودية",
    "3040.SR": "أسمنت القصيم",
    "3050.SR": "أسمنت الجنوب",
    "3060.SR": "أسمنت ينبع",
    "3010.SR": "أسمنت العربية",
    "3020.SR": "أسمنت اليمامة",
    "3080.SR": "أسمنت الشرقية",
    
    # --- التجزئة والأغذية ---
    "4190.SR": "جرير",
    "4001.SR": "العثيم",
    "4164.SR": "النهدي",
    "2280.SR": "المراعي",
    "2270.SR": "سدافكو",
    "6002.SR": "هرفي",
    "4160.SR": "تموين (التموين)",
    "6010.SR": "نادك",
    "6020.SR": "جاكو",
    "6040.SR": "تبوك الزراعية",
    
    # --- الصحة والتأمين ---
    "4002.SR": "المواساة",
    "4004.SR": "دلة",
    "4007.SR": "الحمادي",
    "4009.SR": "الألماني",
    "4013.SR": "سليمان الحبيب",
    "8010.SR": "التعاونية",
    "8210.SR": "بوبا",
    "8230.SR": "الراجحي تكافل",
    "8012.SR": "جزيرة تكافل",
    
    # --- التطوير العقاري والريت ---
    "4300.SR": "دار الأركان",
    "4250.SR": "جبل عمر",
    "4220.SR": "إعمار",
    "4321.SR": "المراكز",
    "4230.SR": "البحر الأحمر",
    "4090.SR": "طيبة",
    "4100.SR": "مكة",
    "4330.SR": "الرياض ريت",
    "4340.SR": "الراجحي ريت",
    
    # --- السياحة والخدمات الأخرى ---
    "1810.SR": "سيرا",
    "1830.SR": "وقت اللياقة",
    "4070.SR": "تهامة",
    "4210.SR": "الأبحاث",
    "4080.SR": "سناد القابضة"
}

# --- دالة RMA (Pine Script Logic) ---
def calculate_rsi_rma(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- تهيئة الجلسة ---
if 'market_data' not in st.session_state:
    st.session_state['market_data'] = {}
if 'summary' not in st.session_state:
    st.session_state['summary'] = []

# --- الواجهة ---
col_btn, col_count = st.columns([1, 4])
with col_btn:
    start_btn = st.button('🚀 فحص شامل للسوق')

with col_count:
    st.caption(f"عدد الشركات المدرجة للفحص: {len(TICKERS)}")

if start_btn:
    st.session_state['market_data'] = {}
    st.session_state['summary'] = []
    
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    # تحميل جماعي ذكي (دفعة واحدة لتسريع العملية)
    try:
        status_text.text("⏳ جاري الاتصال بقاعدة البيانات وسحب سجلات سنتين...")
        
        # التغيير هنا: التحميل الجماعي أسرع بكثير من حلقة التكرار
        # threads=True يفعل التحميل المتوازي
        raw_data = yf.download(list(TICKERS.keys()), period="2y", interval="1d", group_by='ticker', auto_adjust=False, threads=True, progress=False)
        
        if not raw_data.empty:
            processed_count = 0
            
            for symbol, name in TICKERS.items():
                try:
                    # محاولة استخراج بيانات الشركة
                    try:
                        df = raw_data[symbol].copy()
                    except KeyError:
                        continue # الشركة قد لا يكون لها بيانات (موقوفة مثلاً)

                    # توحيد اسم العمود
                    target_col = None
                    if 'Close' in df.columns: target_col = 'Close'
                    elif 'Adj Close' in df.columns: target_col = 'Adj Close'
                    
                    if target_col:
                        # تنظيف البيانات
                        series = df[target_col].dropna()
                        
                        # نحتاج بيانات كافية للحساب
                        if len(series) > RSI_PERIOD + 20:
                            # حساب RSI
                            rsi_values = calculate_rsi_rma(series, RSI_PERIOD)
                            
                            # تخزين البيانات في الداتا فريم
                            df['RSI'] = rsi_values
                            df['Close_Clean'] = series
                            
                            last_rsi = rsi_values.iloc[-1]
                            last_price = series.iloc[-1]
                            
                            # الحفظ في الذاكرة
                            st.session_state['market_data'][name] = df
                            
                            if not np.isnan(last_rsi):
                                st.session_state['summary'].append({
                                    "الاسم": name,
                                    "الرمز": symbol,
                                    "آخر سعر": last_price,
                                    f"RSI ({RSI_PERIOD})": last_rsi
                                })
                    
                    processed_count += 1
                    progress_bar.progress(processed_count / len(TICKERS))
                    
                except Exception as e:
                    pass
            
            progress_bar.empty()
            status_text.success("✅ تم الانتهاء من فحص السوق!")
            
        else:
            status_text.error("فشل التحميل الجماعي. قد يكون هناك ضغط على المصدر.")
            
    except Exception as e:
        status_text.error(f"حدث خطأ غير متوقع: {e}")

# --- العرض ---
if st.session_state['summary']:
    
    # 1. الجدول الرئيسي
    st.subheader("📋 نتائج الفحص الشامل")
    
    df_sum = pd.DataFrame(st.session_state['summary'])
    df_sum = df_sum.sort_values(by=f"RSI ({RSI_PERIOD})", ascending=False)
    
    # دالة التلوين
    def highlight_rsi(val):
        bg = ''
        color = '#d1d1d1' # رمادي فاتح للنصوص العادية
        weight = 'normal'
        
        if val >= 70:
            bg = '#8B0000' # أحمر غامق
            color = 'white'
            weight = 'bold'
        elif val <= 30:
            bg = '#006400' # أخضر غامق
            color = 'white'
            weight = 'bold'
        elif 30 < val < 40:
             color = '#90EE90' # أخضر فاتح
             weight = 'bold'
        elif 60 < val < 70:
             color = '#FF7F7F' # أحمر فاتح
             weight = 'bold'
             
        style = f'color: {color}; font-weight: {weight};'
        if bg: style += f' background-color: {bg}; border-radius: 4px;'
        return style

    st.dataframe(
        df_sum.style.map(highlight_rsi, subset=[f"RSI ({RSI_PERIOD})"])
                  .format({"آخر سعر": "{:.2f}", f"RSI ({RSI_PERIOD})": "{:.2f}"}),
        use_container_width=True,
        height=500
    )
    
    st.markdown("---")
    
    # 2. التفاصيل التفاعلية
    st.subheader("🔍 تحليل عميق لشركة محددة")
    
    company_list = [d['الاسم'] for d in st.session_state['summary']]
    selected_comp = st.selectbox("اختر الشركة لعرض سجل 24 يوم:", company_list)
    
    if selected_comp:
        df_details = st.session_state['market_data'][selected_comp]
        
        # تجهيز آخر 24 يوم
        last_24 = df_details.tail(24).sort_index(ascending=False)
        
        # اختيار الأعمدة
        cols = ['Close_Clean', 'RSI']
        if 'Open' in last_24.columns: cols.insert(0, 'Open')
        if 'High' in last_24.columns: cols.insert(1, 'High')
        if 'Low' in last_24.columns: cols.insert(2, 'Low')
        
        # إعادة تسمية للعرض
        last_24 = last_24[cols].rename(columns={'Close_Clean': 'Close'})
        
        st.write(f"سجل **{selected_comp}**:")
        st.dataframe(
            last_24.style.map(highlight_rsi, subset=['RSI'])
                     .format("{:.2f}"),
            use_container_width=True
        )

else:
    if not start_btn:
        st.info("اضغط الزر أعلاه لبدء تحميل وتحليل بيانات السوق.")

