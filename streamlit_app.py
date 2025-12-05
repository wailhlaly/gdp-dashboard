import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="RSI Simple 24", layout="wide")
st.title("📊 ماسح RSI (الحساب المباشر لآخر 24 شمعة)")

# --- الإعدادات ---
RSI_PERIOD = 24
TARGET_STOCK = "1180.SR" # البنك الأهلي

# --- دالة الحساب المباشر (Cutler's / Simple RSI) ---
def calculate_simple_rsi_on_window(series, period):
    # 1. نحتاج آخر (Period + 1) إغلاق لحساب (Period) تغيير
    if len(series) < period + 1:
        return None
        
    # نأخذ النافذة الزمنية المطلوبة بالضبط (آخر 25 يوم للحصول على 24 تغيير)
    window_series = series.iloc[-(period + 1):]
    
    # حساب الفرق
    delta = window_series.diff().dropna()
    
    # فصل الربح والخسارة
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # --- التنفيذ الحرفي لطلبك ---
    # حساب المتوسط البسيط (Simple Mean) لهذه الفترة فقط
    avg_gain = gain.mean()
    avg_loss = loss.mean()
    
    if avg_loss == 0:
        return 100
        
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

if st.button(f"احسب بناءً على آخر {RSI_PERIOD} يوم فقط"):
    
    # نجلب بيانات شهرين لنضمن وجود 24 يوم تداول
    df = yf.download(TARGET_STOCK, period="3mo", interval="1d", auto_adjust=False, progress=False)
    
    if not df.empty:
        # استخراج عمود الإغلاق
        try:
            if isinstance(df.columns, pd.MultiIndex):
                close_series = df.xs('Close', level=0, axis=1)[TARGET_STOCK]
            else:
                close_series = df['Close']
        except:
             close_series = df.iloc[:, 0]

        close_series = close_series.dropna()
        
        # --- الحساب ---
        rsi_val = calculate_simple_rsi_on_window(close_series, RSI_PERIOD)
        
        last_price = close_series.iloc[-1]
        
        st.subheader("النتيجة (Strict 24-Day Calculation):")
        
        if rsi_val is not None:
            col1, col2 = st.columns(2)
            col1.metric("آخر سعر إغلاق", f"{last_price:.2f}")
            col2.metric(f"RSI ({RSI_PERIOD})", f"{rsi_val:.2f}")
            
            st.info(f"""
            **طريقة الحساب المستخدمة هنا:**
            1. تم عزل آخر {RSI_PERIOD} تغيير في السعر بالضبط.
            2. تم حساب مجموع الأرباح ÷ {RSI_PERIOD}.
            3. تم حساب مجموع الخسائر ÷ {RSI_PERIOD}.
            4. تم استخراج المؤشر (بدون أي اعتماد على بيانات أقدم من 24 يوم).
            """)
            
            # عرض البيانات المستخدمة للمصداقية
            with st.expander("عرض الـ 24 يوم المستخدمة في الحساب"):
                window_data = close_series.iloc[-(RSI_PERIOD+1):]
                st.dataframe(window_data)
        else:
            st.error("البيانات غير كافية (نحتاج 25 يوم تداول على الأقل).")
            
    else:
        st.error("فشل الاتصال بالمصدر.")
