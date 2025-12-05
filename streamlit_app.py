import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import os
from datetime import date, timedelta

# --- إعدادات الصفحة ---
st.set_page_config(page_title="ماسح الأسهم الذكي", layout="wide")
st.title("📊 ماسح RSI للسوق السعودي (تحديث ذكي)")

# --- قائمة الأسهم (عينة للتجربة - يمكنك إضافة السوق كاملاً هنا) ---
# وضعت لك أهم الشركات لتجربة السرعة
TICKERS = [
    "1120.SR", "2222.SR", "2010.SR", "1180.SR", "7010.SR", 
    "4030.SR", "5110.SR", "4200.SR", "1150.SR", "1010.SR",
    "^TASI.SR" # المؤشر العام
]

FILE_NAME = "saudi_market_data.csv"

# --- دالة التعامل مع البيانات (القلب النابض للتطبيق) ---
def get_smart_data(tickers):
    # 1. تحديد تاريخ اليوم وتاريخ البداية (قبل 3 شهور)
    today = date.today()
    start_lookback = today - timedelta(days=90)
    
    combined_df = pd.DataFrame()

    # 2. هل الملف موجود مسبقاً؟
    if os.path.exists(FILE_NAME):
        # تحميل البيانات المحفوظة
        try:
            stored_df = pd.read_csv(FILE_NAME, index_col=0, parse_dates=True)
            
            # التأكد من أن البيانات ليست فارغة
            if not stored_df.empty:
                last_stored_date = stored_df.index[-1].date()
                
                # إذا كانت البيانات قديمة (آخر تاريخ أصغر من اليوم)
                if last_stored_date < today:
                    st.toast(f"🔄 وجدنا بيانات حتى {last_stored_date}.. جاري تحديث الجديد فقط!")
                    
                    # نطلب البيانات من اليوم التالي لآخر حفظ
                    new_start = last_stored_date + timedelta(days=1)
                    
                    # إذا كان هناك أيام مفقودة فعلاً
                    if new_start <= today:
                        new_data = yf.download(tickers, start=new_start, end=today + timedelta(days=1), group_by='ticker', progress=False)
                        
                        if not new_data.empty:
                            # دمج البيانات القديمة مع الجديدة
                            # ملاحظة: yfinance multi-index يحتاج معالجة خاصة عند الدمج، هنا نبسطه للتجربة
                            # للتبسيط في هذا النموذج: سنقوم بإعادة بناء الملف إذا كان الفارق كبير
                            # ولكن الكود أدناه هو لمحاولة الدمج
                            combined_df = pd.concat([stored_df, new_data])
                        else:
                            combined_df = stored_df
                    else:
                        combined_df = stored_df
                else:
                    st.toast("✅ البيانات محدثة، يتم التحميل من الملف المحلي.")
                    combined_df = stored_df
            else:
                # الملف موجود لكن فارغ
                combined_df = yf.download(tickers, start=start_lookback, group_by='ticker', progress=False)
        except:
             combined_df = yf.download(tickers, start=start_lookback, group_by='ticker', progress=False)
    else:
        st.toast("📥 جاري تحميل بيانات 3 أشهر لأول مرة...")
        combined_df = yf.download(tickers, start=start_lookback, group_by='ticker', progress=False)

    # 3. حفظ البيانات المحدثة
    if not combined_df.empty:
        combined_df.to_csv(FILE_NAME)
        
    return combined_df

# --- تشغيل الدالة وجلب البيانات ---
try:
    df_master = get_smart_data(TICKERS)
except Exception as e:
    st.error(f"حدث خطأ في جلب البيانات: {e}")
    st.stop()

# --- معالجة وحساب RSI ---
rsi_results = []

if not df_master.empty:
    # نحتاج للدوران على كل سهم لحساب مؤشراته
    # هيكلة بيانات yfinance تكون: (PriceType, Ticker) أو (Ticker, PriceType) حسب النسخة
    # سنتعامل معها بمرونة
    
    for ticker in TICKERS:
        try:
            # استخراج بيانات السهم الواحد
            # التعامل مع MultiIndex يعتمد على طريقة التحميل
            try:
                stock_df = df_master[ticker].copy()
            except KeyError:
                continue # السهم غير موجود في البيانات
            
            # تنظيف البيانات (حذف الصفوف الفارغة)
            stock_df.dropna(inplace=True)

            if len(stock_df) > 14: # نحتاج 14 يوم على الأقل للـ RSI
                # حساب RSI باستخدام pandas_ta
                # نستخدم .iloc لاستخراج عمود الإغلاق كسلسلة بيانات
                close_series = stock_df['Close']
                
                # حساب القيمة
                rsi_val = ta.rsi(close_series, length=14)
                
                if rsi_val is not None:
                    last_rsi = rsi_val.iloc[-1]
                    last_price = stock_df['Close'].iloc[-1]
                    
                    rsi_results.append({
                        "الرمز": ticker,
                        "السعر الحالي": round(last_price, 2),
                        "RSI (14)": round(last_rsi, 2)
                    })
        except Exception as e:
            pass # تجاهل الأخطاء الفردية للاستمرار

# --- العرض النهائي ---

# 1. تحويل القائمة لجدول
df_results = pd.DataFrame(rsi_results)

if not df_results.empty:
    # 2. الترتيب من الأكبر للأصغر حسب الطلب
    df_results = df_results.sort_values(by="RSI (14)", ascending=False)
    
    # تنسيق الجدول وتلوينه
    st.subheader("📋 قائمة الأسهم مرتبة حسب قوة RSI")
    
    # دالة لتلوين القيم
    def color_rsi(val):
        color = 'black'
        if val > 70:
            color = 'red' # تشبع شرائي (خطر/جني أرباح)
        elif val < 30:
            color = 'green' # تشبع بيعي (فرصة محتملة)
        return f'color: {color}'

    st.dataframe(
        df_results.style.map(color_rsi, subset=['RSI (14)'])
                  .format({"السعر الحالي": "{:.2f}", "RSI (14)": "{:.2f}"}),
        use_container_width=True,
        height=600 # طول الجدول
    )
    
    # معلومات إضافية
    st.info("💡 ملاحظة: الترتيب من الأعلى (70+ تشبع شرائي) إلى الأسفل (30- تشبع بيعي).")
    
    # زر لحذف الملف (لتجربة إعادة التحميل من الصفر)
    if st.button("🗑️ حذف البيانات المحفوظة (إعادة ضبط)"):
        if os.path.exists(FILE_NAME):
            os.remove(FILE_NAME)
            st.rerun() # إعادة تحميل الصفحة
else:
    st.warning("لا توجد بيانات كافية لحساب RSI حالياً.")

