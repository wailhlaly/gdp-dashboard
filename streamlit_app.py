import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, date

st.set_page_config(page_title="RSI 24 Exact", layout="wide")
st.title("📊 ماسح RSI 24 (المطابق رياضياً لـ TradingView)")

# --- الإعدادات ---
RSI_PERIOD = 24
FILE_NAME = "tasi_rsi_exact.csv"

# القائمة
TICKERS = {
    "1180.SR": "الأهلي",
    "1120.SR": "الراجحي",
    "2222.SR": "أرامكو",
    "2010.SR": "سابك",
    "7010.SR": "STC",
    "^TASI.SR": "المؤشر العام"
}

# --- دالة RSI اليدوية (Simulating Pine Script RMA) ---
def calculate_rsi_exact(series, period):
    # تحويل البيانات إلى قائمة للسرعة في المعالجة
    prices = series.values
    
    # حساب التغيرات
    deltas = np.diff(prices)
    
    # مصفوفات الأرباح والخسائر
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gains = np.zeros_like(prices)
    avg_losses = np.zeros_like(prices)
    
    # --- الخطوة 1: التهيئة (SMA) ---
    # TradingView يبدأ بحساب متوسط بسيط لأول 24 يوم
    if len(prices) > period:
        avg_gains[period] = np.mean(gains[:period])
        avg_losses[period] = np.mean(losses[:period])
        
        # --- الخطوة 2: التنعيم الأسي (RMA/Wilder's) ---
        # المعادلة: (Previous * (n-1) + Current) / n
        for i in range(period + 1, len(prices)):
            avg_gains[i] = (avg_gains[i-1] * (period - 1) + gains[i-1]) / period
            avg_losses[i] = (avg_losses[i-1] * (period - 1) + losses[i-1]) / period
            
    # حساب RS و RSI
    # نتجنب القسمة على صفر
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
    
    # استبدال القيم اللانهائية (في حال كان الهبوط صفر)
    rsi[np.isinf(rsi)] = 100
    
    return pd.Series(rsi, index=series.index)

# --- التشغيل ---
if st.button('🚀 تحديث ومطابقة البيانات'):
    
    st.write("جاري المعالجة...")
    
    # نطلب فترة '5y' لضمان استقرار المعادلة (لن يؤثر على سرعة العرض)
    # المعادلة تحتاج لتاريخ طويل لتصل للدقة العشرية المطلوبة
    data = yf.download(list(TICKERS.keys()), period="5y", interval="1d", group_by='ticker', auto_adjust=False, progress=False)
    
    if not data.empty:
        results = []
        
        for symbol, name in TICKERS.items():
            try:
                try:
                    df = data[symbol].copy()
                except KeyError:
                    continue

                if 'Close' in df.columns:
                    series = df['Close']
                elif 'Adj Close' in df.columns:
                    series = df['Adj Close']
                else:
                    continue
                
                # تنظيف
                series = series.dropna()

                # نحتاج بيانات كافية
                if len(series) > RSI_PERIOD + 1:
                    
                    # استخدام الدالة اليدوية الجديدة
                    rsi_series = calculate_rsi_exact(series, RSI_PERIOD)
                    
                    last_rsi = rsi_series.iloc[-1]
                    last_price = series.iloc[-1]
                    
                    # التحقق من عدم وجود NaN
                    if not np.isnan(last_rsi) and last_rsi != 0:
                        results.append({
                            "الرمز": symbol,
                            "الاسم": name,
                            "السعر": round(last_price, 2),
                            f"RSI ({RSI_PERIOD})": round(last_rsi, 2)
                        })
            except Exception as e:
                pass

        if results:
            df_final = pd.DataFrame(results)
            col_rsi = f"RSI ({RSI_PERIOD})"
            df_final = df_final.sort_values(by=col_rsi, ascending=False)
            
            # تلوين
            def color_rsi(val):
                color = 'black'
                if val >= 70: color = '#d32f2f'
                elif val <= 30: color = '#388e3c'
                return f'color: {color}; font-weight: bold'

            st.dataframe(
                df_final.style.map(color_rsi, subset=[col_rsi])
                        .format({"السعر": "{:.2f}", col_rsi: "{:.2f}"}),
                use_container_width=True
            )
        else:
            st.error("لا توجد بيانات كافية للحساب.")
    else:
        st.error("فشل الاتصال بالمصدر.")
