import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import os

# --- إعداد الصفحة ---
st.set_page_config(page_title="TASI Cached Screener", layout="wide")
st.title("📊 الماسح الشامل للسوق السعودي (مع ميزة الحفظ التلقائي)")

# --- الإعدادات ---
RSI_PERIOD = 24
CACHE_FILE = "tasi_market_results.csv"

# --- القائمة الشاملة ---
TICKERS = {
    # === الطاقة ===
    "2222.SR": "أرامكو", "2030.SR": "المصافي", "4200.SR": "الدريس", "5110.SR": "الكهرباء", "2080.SR": "الغاز", "4030.SR": "البحري", "2380.SR": "رابغ", "2381.SR": "الحفر العربية", "2382.SR": "أديس",
    
    # === المواد الأساسية ===
    "2010.SR": "سابك", "1211.SR": "معادن", "2020.SR": "سابك للمغذيات", "2310.SR": "سبكيم", "2060.SR": "التصنيع", "2290.SR": "ينساب", "2001.SR": "كيمانول", "2170.SR": "اللجين", "2330.SR": "المتقدمة", "2350.SR": "كيان", "2090.SR": "جبسكو", "2150.SR": "زجاج", "2180.SR": "فيبكو", "2200.SR": "أنابيب", "2210.SR": "نما", "2230.SR": "الكيميائية", "2240.SR": "الزامل", "2250.SR": "المجموعة", "2300.SR": "صناعة الورق", "2320.SR": "البابطين", "2340.SR": "العبداللطيف", "2360.SR": "الفخارية", "2370.SR": "مسك", "3001.SR": "أسمنت حائل", "3002.SR": "أسمنت نجران", "3003.SR": "أسمنت المدينة", "3004.SR": "أسمنت الشمالية", "3005.SR": "أسمنت أم القرى", "3007.SR": "زهرة الواحة", "3008.SR": "الكثيري", "3010.SR": "أسمنت العربية", "3020.SR": "أسمنت اليمامة", "3030.SR": "أسمنت السعودية", "3040.SR": "أسمنت القصيم", "3050.SR": "أسمنت الجنوب", "3060.SR": "أسمنت ينبع", "3080.SR": "أسمنت الشرقية", "3090.SR": "أسمنت تبوك", "3091.SR": "أسمنت الجوف", "1301.SR": "أسلاك", "1304.SR": "اليمامة للحديد", "1320.SR": "أنابيب الشرق", "1321.SR": "أنابيب السعودية", "1322.SR": "المطاحن الأولى",
    
    # === البنوك والتمويل ===
    "1120.SR": "الراجحي", "1180.SR": "الأهلي", "1010.SR": "الرياض", "1150.SR": "الإنماء", "1060.SR": "الأول", "1140.SR": "البلاد", "1030.SR": "الاستثمار", "1020.SR": "الجزيرة", "1080.SR": "العربي", "1050.SR": "الفرنسي", "1111.SR": "تداول", "1182.SR": "أملاك", "1183.SR": "سهل", "4081.SR": "النايفات", "4280.SR": "المملكة",
    
    # === السلع والخدمات ===
    "1201.SR": "تكوين", "1202.SR": "مبكو", "1210.SR": "بي سي آي", "1212.SR": "أسترا", "1214.SR": "شاكر", "1302.SR": "بوان", "1303.SR": "الصناعات الكهربائية", "1831.SR": "مهارة", "1832.SR": "صدر", "2040.SR": "الخزف", "2110.SR": "الكابلات", "2140.SR": "الأحساء", "2390.SR": "أسيج", "4020.SR": "العقارية", "4040.SR": "الجماعي", "4050.SR": "ساسكو", "4070.SR": "تهامة", "4110.SR": "باتك", "4140.SR": "الصادرات", "4141.SR": "العمران", "4142.SR": "الرياض للحديد",
    
    # === الخدمات التجارية ===
    "1810.SR": "سيرا", "1820.SR": "مجموعة الحكير", "1830.SR": "وقت اللياقة", "1833.SR": "الموارد", "4260.SR": "بدجت", "4261.SR": "ذيب", "4262.SR": "لومي", "4080.SR": "سناد", "6004.SR": "التموين", "6012.SR": "ريدان", 
    
    # === النقل ===
    "4031.SR": "الخدمات الأرضية", "4263.SR": "سال",
    
    # === الاستهلاكية والتجزئة ===
    "4011.SR": "لازوردي", "4012.SR": "أصيل", "4014.SR": "تنمية", "1834.SR": "مرافق", "2190.SR": "سيسكو", "4003.SR": "إكسترا", "4008.SR": "ساكو", "4161.SR": "بن داود", "4162.SR": "المنجم", "4163.SR": "الدواء", "4164.SR": "النهدي", "4190.SR": "جرير", "4191.SR": "السيف غاليري", "4001.SR": "العثيم", "4006.SR": "المزرعة", "4061.SR": "أنعام", "4100.SR": "مكة", "4170.SR": "شمس", "4180.SR": "فتيحي", "4290.SR": "الخليج للتدريب", "4291.SR": "الوطنية للتعليم", "4292.SR": "عطاء", "6001.SR": "حلواني", "6002.SR": "هرفي", "2270.SR": "سدافكو", "2280.SR": "المراعي", "6010.SR": "نادك", "6020.SR": "جاكو", "6040.SR": "تبوك الزراعية", "6050.SR": "الأسماك", "6060.SR": "الشرقية الزراعية", "6070.SR": "الجوف", "6090.SR": "جازادكو",
    
    # === الصحة والتأمين ===
    "4002.SR": "المواساة", "4004.SR": "دلة", "4005.SR": "رعاية", "4007.SR": "الحمادي", "4009.SR": "الألماني", "4013.SR": "سليمان الحبيب", "4015.SR": "جمجوم فارما", "8010.SR": "التعاونية", "8012.SR": "جزيرة تكافل", "8020.SR": "ملاذ", "8030.SR": "ميدغلف", "8040.SR": "أليانز", "8050.SR": "سلامة", "8060.SR": "ولاّء", "8070.SR": "الدرع العربي", "8100.SR": "سايكو", "8120.SR": "اتحاد الخليج", "8150.SR": "أسيج", "8160.SR": "التأمين العربية", "8170.SR": "الاتحاد", "8180.SR": "الصقر", "8190.SR": "المتحدة", "8200.SR": "إعادة", "8210.SR": "بوبا", "8230.SR": "الراجحي تكافل", "8240.SR": "تشب", "8250.SR": "جي جي", "8260.SR": "الخليجية", "8270.SR": "بروج", "8280.SR": "العالمية", "8300.SR": "الوطنية", "8310.SR": "أمانة", "8311.SR": "عناية", "8312.SR": "الإنماء طوكيو",
    
    # === الاتصالات والعقار ===
    "7010.SR": "STC", "7020.SR": "موبايلي", "7030.SR": "زين", "7040.SR": "عذيب", "7200.SR": "سلوشنز", "7201.SR": "بحر العرب", "7202.SR": "علم", "7203.SR": "توبي", "4090.SR": "طيبة", "4150.SR": "التعمير", "4220.SR": "إعمار", "4230.SR": "البحر الأحمر", "4240.SR": "الحكير", "4250.SR": "جبل عمر", "4300.SR": "دار الأركان", "4310.SR": "مدينة المعرفة", "4320.SR": "الأندلس", "4321.SR": "المراكز", "4322.SR": "رتال", "4323.SR": "سمو",
    "4330.SR": "الرياض ريت", "4331.SR": "الجزيرة ريت", "4332.SR": "جدوى ريت الحرمين", "4333.SR": "تعليم ريت", "4334.SR": "المعذر ريت", "4335.SR": "مشاركة ريت", "4336.SR": "ملكيات ريت", "4337.SR": "سدكو كابيتال ريت", "4338.SR": "الأهلي ريت 1", "4339.SR": "دراية ريت", "4340.SR": "الراجحي ريت", "4342.SR": "جدوى ريت السعودية", "4344.SR": "سدكو كابيتال", "4345.SR": "الإنماء ريت", "4346.SR": "ميفك ريت", "4347.SR": "بنيان ريت", "4348.SR": "الخبير ريت", 
    
    # === المؤشر ===
    "^TASI.SR": "المؤشر العام"
}

# --- دالة RMA (حساب RSI) ---
def calculate_rsi_rma(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- دالة التلوين ---
def highlight_rsi(val):
    bg = ''
    color = '#333333'
    weight = 'normal'
    if val >= 70:
        bg = '#8B0000' # أحمر
        color = 'white'
        weight = 'bold'
    elif val <= 30:
        bg = '#006400' # أخضر
        color = 'white'
        weight = 'bold'
    
    style = f'color: {color}; font-weight: {weight};'
    if bg: style += f' background-color: {bg}; border-radius: 4px;'
    return style

# --- تهيئة الجلسة وتحميل الملف ---
if 'summary' not in st.session_state:
    if os.path.exists(CACHE_FILE):
        try:
            # تحميل الملف المحفوظ
            loaded_df = pd.read_csv(CACHE_FILE)
            st.session_state['summary'] = loaded_df.to_dict('records')
            st.toast("📂 تم تحميل النتائج من الملف المحفوظ بنجاح.")
        except:
            st.session_state['summary'] = []
    else:
        st.session_state['summary'] = []

# --- الواجهة ---
col_btn, col_info = st.columns([1, 4])

with col_btn:
    update_btn = st.button('🚀 تحديث بيانات السوق')

with col_info:
    if st.session_state['summary']:
        st.info(f"يتم عرض بيانات محفوظة لـ {len(st.session_state['summary'])} شركة. اضغط 'تحديث' لجلب بيانات جديدة.")
    else:
        st.warning("لا توجد بيانات محفوظة. يرجى الضغط على 'تحديث' لأول مرة.")

# --- منطق التحديث (Batch Processing) ---
if update_btn:
    st.write("بدء عملية المسح الشامل... (يرجى الانتظار)")
    
    tickers_list = list(TICKERS.keys())
    total_tickers = len(tickers_list)
    chunk_size = 50 
    
    new_summary = []
    status_box = st.empty()
    progress_bar = st.progress(0)
    
    for i in range(0, total_tickers, chunk_size):
        chunk = tickers_list[i:i + chunk_size]
        status_box.text(f"⏳ جاري تحميل الدفعة {i//chunk_size + 1} (الأسهم {i} إلى {min(i+chunk_size, total_tickers)})...")
        
        try:
            # تحميل الدفعة
            data_chunk = yf.download(chunk, period="2y", interval="1d", group_by='ticker', auto_adjust=False, threads=True, progress=False)
            
            if not data_chunk.empty:
                for symbol in chunk:
                    try:
                        name = TICKERS[symbol]
                        
                        # استخراج البيانات
                        try:
                            df = data_chunk[symbol].copy()
                        except KeyError:
                            continue

                        # توحيد العمود
                        target_col = None
                        if 'Close' in df.columns: target_col = 'Close'
                        elif 'Adj Close' in df.columns: target_col = 'Adj Close'
                        
                        if target_col:
                            series = df[target_col].dropna()
                            
                            if len(series) > RSI_PERIOD + 20:
                                rsi_vals = calculate_rsi_rma(series, RSI_PERIOD)
                                last_rsi = rsi_vals.iloc[-1]
                                last_price = series.iloc[-1]
                                
                                if not np.isnan(last_rsi):
                                    new_summary.append({
                                        "الاسم": name,
                                        "الرمز": symbol,
                                        "السعر": last_price,
                                        f"RSI ({RSI_PERIOD})": last_rsi
                                    })
                    except:
                        continue
        except Exception as e:
            st.warning(f"مشكلة في الدفعة: {e}")
        
        progress_bar.progress(min((i + chunk_size) / total_tickers, 1.0))
        time.sleep(0.5)

    progress_bar.empty()
    
    if new_summary:
        # تحديث الجلسة
        st.session_state['summary'] = new_summary
        
        # حفظ النتائج في ملف CSV
        df_to_save = pd.DataFrame(new_summary)
        df_to_save.to_csv(CACHE_FILE, index=False)
        
        status_box.success(f"✅ تم التحديث وحفظ {len(new_summary)} شركة في الملف '{CACHE_FILE}'")
    else:
        status_box.error("فشل التحديث.")

# --- العرض ---
if st.session_state['summary']:
    
    # 1. الجدول الرئيسي
    st.subheader("📋 نتائج السوق (من الذاكرة)")
    
    df_sum = pd.DataFrame(st.session_state['summary'])
    df_sum = df_sum.sort_values(by=f"RSI ({RSI_PERIOD})", ascending=False)
    
    st.dataframe(
        df_sum.style.map(highlight_rsi, subset=[f"RSI ({RSI_PERIOD})"])
                  .format({"السعر": "{:.2f}", f"RSI ({RSI_PERIOD})": "{:.2f}"}),
        use_container_width=True,
        height=500
    )
    
    st.markdown("---")
    
    # 2. التفاصيل التفاعلية (تحميل لحظي)
    col_sel, col_chart = st.columns([1, 2])
    
    with col_sel:
        st.subheader("🔎 فحص تفصيلي لشركة")
        comp_list = sorted([d['الاسم'] for d in st.session_state['summary']])
        selected = st.selectbox("اختر الشركة:", comp_list)
        
    if selected:
        # البحث عن الرمز
        selected_row = next((item for item in st.session_state['summary'] if item["الاسم"] == selected), None)
        
        if selected_row:
            symbol = selected_row['الرمز']
            
            with st.spinner(f"جاري جلب سجل {selected} لحظياً..."):
                # تحميل بيانات سهم واحد فقط (سريع جداً)
                df_single = yf.download(symbol, period="6mo", interval="1d", auto_adjust=False, progress=False)
                
                if not df_single.empty:
                    # معالجة RSI للسهم الواحد
                    # التعامل مع الـ MultiIndex إذا وجد
                    try:
                         if isinstance(df_single.columns, pd.MultiIndex):
                            series = df_single.xs('Close', level=0, axis=1)[symbol]
                         else:
                            series = df_single['Close']
                    except:
                        series = df_single['Close']

                    series = series.dropna()
                    rsi_series = calculate_rsi_rma(series, RSI_PERIOD)
                    
                    # دمج البيانات للعرض
                    display_df = pd.DataFrame({
                        'Open': df_single['Open'] if 'Open' in df_single.columns else df_single.xs('Open', level=0, axis=1)[symbol],
                        'High': df_single['High'] if 'High' in df_single.columns else df_single.xs('High', level=0, axis=1)[symbol],
                        'Low': df_single['Low'] if 'Low' in df_single.columns else df_single.xs('Low', level=0, axis=1)[symbol],
                        'Close': series,
                        'RSI': rsi_series
                    })
                    
                    # عرض آخر 24 يوم
                    last_24 = display_df.tail(24).sort_index(ascending=False)
                    
                    st.write(f"سجل **{selected}** (لحظي):")
                    st.dataframe(
                        last_24.style.map(highlight_rsi, subset=['RSI']).format("{:.2f}"),
                        use_container_width=True
                    )
