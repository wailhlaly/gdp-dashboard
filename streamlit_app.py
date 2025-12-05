import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# --- إعداد الصفحة ---
st.set_page_config(page_title="محلل السوق السعودي", layout="wide")
st.title("📊 تحليل RSI للسوق السعودي")

# --- دالة حساب RSI (نسخة مُحسنة) ---
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    # استخدام المتوسط المتحرك الأسي (Exponential Moving Average)
    avg_gain = gain.ewm(span=window, adjust=False).mean()
    avg_loss = loss.ewm(span=window, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- قائمة الأسهم ---
# يمكنك إضافة المزيد هنا
TICKERS = {
    "1120.SR": "الراجحي",
    "2222.SR": "أرامكو",
    "2010.SR": "سابك",
    "1180.SR": "الأهلي",
    "7010.SR": "STC",
    "4030.SR": "البحري",
    "5110.SR": "كهرباء السعودية",
    "4200.SR": "الدريس",
    "^TASI.SR": "المؤشر العام"
}

# --- زر التشغيل ---
if st.button('🔄 تحديث البيانات وحساب المؤشرات'):
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # حلقة لجلب بيانات كل سهم على حدة (أكثر استقراراً)
    for i, (symbol, name) in enumerate(TICKERS.items()):
        status_text.text(f"جاري تحليل: {name}...")
        
        try:
            # جلب بيانات سنة كاملة لضمان دقة الحساب
            stock_data = yf.download(symbol, period="1y", interval="1d", progress=False)
            
            # التأكد من أن البيانات ليست فارغة
            if not stock_data.empty and len(stock_data) > 20:
                
                # التعامل مع مشاكل تسمية الأعمدة
                if 'Close' in stock_data.columns:
                    close_prices = stock_data['Close']
                elif 'Adj Close' in stock_data.columns:
                    close_prices = stock_data['Adj Close']
                else:
                    # محاولة أخيرة لاستخراج العمود الأول كأنه الإغلاق
                    close_prices = stock_data.iloc[:, 0]

                # --- حساب RSI ---
                # نقوم بتحويل البيانات إلى سلسلة رقمية بحتة لتجنب الأخطاء
                close_series = pd.Series(close_prices.values.flatten(), index=stock_data.index)
                
                rsi_series = calculate_rsi(close_series)
                
                # استخراج آخر قيمة
                last_rsi = rsi_series.iloc[-1]
                last_price = close_series.iloc[-1]
                
                # التحقق أن القيمة ليست NaN
                if not np.isnan(last_rsi):
                    results.append({
                        "الرمز": symbol,
                        "الاسم": name,
                        "السعر": round(float(last_price), 2),
                        "RSI": round(float(last_rsi), 2)
                    })
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            continue
        
        # تحديث شريط التقدم
        progress_bar.progress((i + 1) / len(TICKERS))

    status_text.text("✅ تم الانتهاء!")
    progress_bar.empty()

    # --- عرض النتائج ---
    if results:
        df_final = pd.DataFrame(results)
        
        # ترتيب حسب RSI من الأكبر للأصغر
        df_final = df_final.sort_values(by="RSI", ascending=False)
        
        # دالة التلوين
        def color_rsi(val):
            color = 'white'
            if val >= 70:
                color = '#ff4b4b' # أحمر (تشبع شرائي)
            elif val <= 30:
                color = '#09ab3b' # أخضر (تشبع بيعي)
            return f'color: {color}; font-weight: bold;'

        st.subheader("📋 ملخص السوق (الأعلى RSI في الأعلى)")
        
        st.dataframe(
            df_final.style.map(color_rsi, subset=['RSI'])
                    .format({"السعر": "{:.2f}", "RSI": "{:.2f}"}),
            use_container_width=True,
            hide_index=True # إخفاء عمود الترقيم الجانبي لشكل أنظف
        )
    else:
        st.error("لم نتمكن من حساب البيانات. قد يكون السوق مغلقاً أو هناك مشكلة في المصدر.")

else:
    st.info("اضغط على الزر أعلاه لبدء التحليل.")

