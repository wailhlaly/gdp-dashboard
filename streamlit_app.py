import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, date

st.set_page_config(page_title="RSI Debugger", layout="wide")
st.title("🕵️‍♂️ فحص تطابق البيانات (RSI 24)")

# --- الإعدادات ---
RSI_PERIOD = 24
FILE_NAME = "debug_data.csv"

# القائمة
TICKERS = {
    "1180.SR": "الأهلي",
    "1120.SR": "الراجحي",
    "^TASI.SR": "المؤشر العام"
}

# --- معادلة TradingView (Wilder's) ---
def calculate_rsi_wilder(series, period):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # هذه المعادلة هي الأدق لمطابقة TradingView
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- الزر والتشغيل ---
if st.button('🔍 افحص البيانات الآن'):
    st.write("جاري جلب البيانات المباشرة بدون حفظ (لضمان التحديث)...")
    
    # نجبر البرنامج على حذف الملف القديم لضمان بيانات جديدة
    if os.path.exists(FILE_NAME):
        os.remove(FILE_NAME)
        
    # تحميل البيانات (بدون تعديل التوزيعات auto_adjust=False)
    data = yf.download(list(TICKERS.keys()), period="2y", interval="1d", group_by='ticker', auto_adjust=False, progress=False)
    
    results = []
    
    for symbol, name in TICKERS.items():
        try:
            # استخراج البيانات
            try:
                df = data[symbol].copy()
            except KeyError:
                continue

            # تحديد العمود
            if 'Close' in df.columns:
                target_col = 'Close'
            elif 'Adj Close' in df.columns:
                target_col = 'Adj Close'
            else:
                continue
            
            # تنظيف
            df = df.dropna(subset=[target_col])
            
            # --- نقطة الفحص المهمة ---
            # نأخذ آخر سعر وآخر تاريخ موجود في البيانات
            last_date = df.index[-1].strftime('%Y-%m-%d')
            last_price = df[target_col].iloc[-1]
            
            # حساب RSI
            rsi_series = calculate_rsi_wilder(df[target_col], RSI_PERIOD)
            last_rsi = rsi_series.iloc[-1]

            results.append({
                "الاسم": name,
                "آخر تاريخ تم جلبه": last_date,     # هذا سيكشف المشكلة
                "آخر سعر إغلاق عند الكود": round(last_price, 2), # قارن هذا بسعرك
                "RSI (24)": round(last_rsi, 2)
            })
            
        except Exception as e:
            st.error(f"خطأ في {name}: {e}")

    # عرض الجدول
    if results:
        df_res = pd.DataFrame(results)
        st.table(df_res) # نستخدم table لعرض البيانات كاملة بوضوح
        
        st.info("""
        **كيف تقرأ النتيجة؟**
        1. انظر لعمود **"آخر سعر إغلاق عند الكود"**: هل يطابق السعر في TradingView (37.22)؟
           - إذا كان الرقم مختلفاً (مثلاً 36.80)، فالمشكلة أن Yahoo Finance لم يحدث السعر بعد.
        2. انظر لعمود **"آخر تاريخ"**: هل هو تاريخ اليوم؟
        """)
    else:
        st.error("لم يتم جلب بيانات.")

