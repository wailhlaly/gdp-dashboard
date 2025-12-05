import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# --- إعداد الصفحة ---
st.set_page_config(page_title="RSI Pro Interactive", layout="wide")
st.title("📊 ماسح RSI التفاعلي (مع سجل 24 يوم)")

# --- الإعدادات ---
RSI_PERIOD = 24

# قائمة الأسهم
TICKERS = {
    "1180.SR": "الأهلي",
    "1120.SR": "الراجحي",
    "2222.SR": "أرامكو",
    "2010.SR": "سابك",
    "7010.SR": "STC",
    "1150.SR": "الإنماء",
    "1211.SR": "معادن",
    "4030.SR": "البحري",
    "4200.SR": "الدريس",
    "4190.SR": "جرير",
    "^TASI.SR": "المؤشر العام"
}

# --- دالة RMA (المطابقة لـ Pine Script) ---
def calculate_rsi_rma(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- تهيئة الجلسة لحفظ البيانات (Caching) ---
if 'market_data' not in st.session_state:
    st.session_state['market_data'] = {}

# --- زر التحديث ---
col_btn, col_info = st.columns([1, 4])
with col_btn:
    if st.button('🔄 تحديث ومسح السوق'):
        st.session_state['market_data'] = {} # تصفير البيانات القديمة
        
        with st.spinner("جاري سحب بيانات سنتين لضمان الدقة..."):
            try:
                # سحب البيانات
                raw_data = yf.download(list(TICKERS.keys()), period="2y", interval="1d", group_by='ticker', auto_adjust=False, progress=False)
                
                if not raw_data.empty:
                    processed_data = {}
                    summary_list = []
                    
                    # معالجة كل سهم
                    for symbol, name in TICKERS.items():
                        try:
                            # استخراج البيانات الخاصة بالسهم
                            try:
                                df = raw_data[symbol].copy()
                            except KeyError:
                                continue

                            # تحديد عمود الإغلاق
                            if 'Close' in df.columns:
                                df = df.rename(columns={'Close': 'Close_Price'}) # إعادة تسمية لتجنب التعارض
                                series = df['Close_Price']
                            elif 'Adj Close' in df.columns:
                                df = df.rename(columns={'Adj Close': 'Close_Price'})
                                series = df['Close_Price']
                            else:
                                continue
                            
                            df = df.dropna()

                            if len(series) > RSI_PERIOD + 20:
                                # حساب RSI وإضافته كعمود في الداتا فريم
                                df['RSI'] = calculate_rsi_rma(series, RSI_PERIOD)
                                
                                last_rsi = df['RSI'].iloc[-1]
                                last_price = series.iloc[-1]
                                
                                # حفظ البيانات الكاملة في الذاكرة (للاستدعاء عند الضغط)
                                processed_data[name] = df 
                                
                                if not np.isnan(last_rsi):
                                    summary_list.append({
                                        "الاسم": name,
                                        "الرمز": symbol,
                                        "السعر الحالي": last_price,
                                        f"RSI ({RSI_PERIOD})": last_rsi
                                    })
                        except Exception as e:
                            pass
                    
                    # حفظ النتائج في الجلسة
                    st.session_state['market_data'] = processed_data
                    st.session_state['summary'] = summary_list
                    st.success("تم التحديث بنجاح!")
                else:
                    st.error("لم يتم العثور على بيانات.")
            except Exception as e:
                st.error(f"حدث خطأ: {e}")

# --- عرض النتائج ---
if 'summary' in st.session_state and st.session_state['summary']:
    
    # 1. جدول الملخص
    st.subheader("📋 ملخص السوق (مرتب حسب التشبع)")
    
    df_summary = pd.DataFrame(st.session_state['summary'])
    df_summary = df_summary.sort_values(by=f"RSI ({RSI_PERIOD})", ascending=False)
    
    # تنسيق الألوان المطور
    def highlight_rsi_advanced(val):
        color = '#ffffff' # لون الخط الافتراضي (أبيض)
        bg_color = ''     # لون الخلفية
        weight = 'normal'
        
        if val >= 70:
            bg_color = '#8B0000' # أحمر غامق (خلفية)
            color = 'white'
            weight = 'bold'
        elif val <= 30:
            bg_color = '#006400' # أخضر غامق (خلفية)
            color = 'white'
            weight = 'bold'
        elif 30 < val < 40:
             color = '#90EE90' # أخضر فاتح (نص فقط)
        elif 60 < val < 70:
             color = '#FF7F7F' # أحمر فاتح (نص فقط)
             
        style = f'color: {color}; font-weight: {weight};'
        if bg_color:
            style += f' background-color: {bg_color}; border-radius: 5px;'
        return style

    st.dataframe(
        df_summary.style.map(highlight_rsi_advanced, subset=[f"RSI ({RSI_PERIOD})"])
                  .format({"السعر الحالي": "{:.2f}", f"RSI ({RSI_PERIOD})": "{:.2f}"}),
        use_container_width=True
    )
    
    st.divider()

    # 2. ميزة استدعاء التفاصيل (التفاعل)
    st.subheader("🔎 تفاصيل الأسعار (آخر 24 يوم)")
    
    # قائمة منسدلة لاختيار الشركة
    selected_company = st.selectbox(
        "اختر الشركة لعرض سجل الأسعار والـ RSI:",
        options=[item['الاسم'] for item in st.session_state['summary']],
        index=0
    )
    
    if selected_company:
        # استرجاع الداتا فريم المحفوظة لهذه الشركة
        stock_df = st.session_state['market_data'][selected_company]
        
        # استخراج آخر 24 يوم فقط
        last_24_days = stock_df.tail(24).copy()
        
        # ترتيب الأعمدة للعرض
        # نحاول العثور على الأعمدة المتاحة (Open, High, Low, Close_Price, RSI)
        cols_to_show = ['Close_Price', 'RSI']
        if 'Open' in last_24_days.columns: cols_to_show.insert(0, 'Open')
        if 'High' in last_24_days.columns: cols_to_show.insert(1, 'High')
        if 'Low' in last_24_days.columns: cols_to_show.insert(2, 'Low')
        
        display_df = last_24_days[cols_to_show].sort_index(ascending=False) # الأحدث في الأعلى
        
        # عرض البيانات مع التلوين
        st.write(f"سجل بيانات **{selected_company}**:")
        st.dataframe(
            display_df.style.map(highlight_rsi_advanced, subset=['RSI'])
                      .format("{:.2f}"),
            use_container_width=True,
            height=400 # ارتفاع مناسب لعرض 24 صف
        )

else:
    st.info("اضغط على زر 'تحديث ومسح السوق' للبدء.")
