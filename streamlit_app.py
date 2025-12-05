import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import os
from datetime import date, timedelta

# --- إعداد الصفحة (يجب أن يكون في البداية) ---
st.set_page_config(page_title="ماسح السوق", layout="wide")

# --- شبكة الأمان لكشف الأخطاء ---
try:
    st.title("📊 فحص حالة التطبيق")

    # قائمة أسهم للتجربة (نقلل العدد للتجربة السريعة)
    TICKERS = ["1120.SR", "2222.SR", "^TASI.SR"]
    
    st.write("1. جاري التحقق من المكتبات... ✅")
    
    # --- دالة جلب البيانات ---
    def get_data():
        st.write("2. محاولة الاتصال بـ Yahoo Finance... ⏳")
        start_date = date.today() - timedelta(days=60)
        # نستخدم download بسيط جداً لتجنب مشاكل الهيكلة
        data = yf.download(TICKERS, start=start_date, group_by='ticker', progress=False)
        
        if data.empty:
            st.warning("⚠️ لم يتم جلب أي بيانات! قد يكون هناك حظر مؤقت من المصدر.")
            return None
        st.write("3. تم جلب البيانات بنجاح ✅")
        return data

    df_master = get_data()

    if df_master is not None:
        rsi_data = []
        
        st.write("4. جاري حساب المؤشرات... ⏳")
        for ticker in TICKERS:
            try:
                # محاولة استخراج السهم (معالجة الأخطاء المحتملة في الهيكلة)
                try:
                    df_stock = df_master[ticker].copy()
                except KeyError:
                    # في حال عادت البيانات بتركيبة مختلفة (بدون MultiIndex)
                    if ticker in df_master.columns:
                        df_stock = df_master  # حالة سهم واحد
                    else:
                        continue

                # تنظيف البيانات
                df_stock = df_stock.dropna()

                if len(df_stock) > 14:
                    # حساب RSI
                    # استخدام 'Close' أو 'Adj Close'
                    close_col = 'Close' if 'Close' in df_stock.columns else 'Adj Close'
                    rsi_val = ta.rsi(df_stock[close_col], length=14)
                    
                    if rsi_val is not None:
                        last_rsi = rsi_val.iloc[-1]
                        last_price = df_stock[close_col].iloc[-1]
                        
                        rsi_data.append({
                            "الرمز": ticker,
                            "السعر": round(last_price, 2),
                            "RSI": round(last_rsi, 2)
                        })
            except Exception as e_inner:
                st.write(f"⚠️ خطأ بسيط في السهم {ticker}: {e_inner}")
                continue

        # عرض الجدول
        if rsi_data:
            df_res = pd.DataFrame(rsi_data)
            st.subheader("النتيجة النهائية:")
            
            # --- التصحيح المحتمل: استخدام applymap بدلاً من map القديمة ---
            # (هذا غالباً هو سبب المشكلة السابقة)
            def color_rsi(val):
                color = 'black'
                if val > 70: color = 'red'
                elif val < 30: color = 'green'
                return f'color: {color}'
            
            # استخدام applymap المتوافقة مع جميع النسخ
            st.dataframe(df_res.style.applymap(color_rsi, subset=['RSI']))
            st.success("✅ التطبيق يعمل بنجاح!")
        else:
            st.warning("لم يتم حساب RSI لأي سهم.")

except Exception as e:
    # هذا الجزء هو الأهم: سيعرض لك الخطأ الحقيقي
    st.error("🚨 حدث خطأ أثناء التشغيل:")
    st.code(e)
    st.write("صور هذه الشاشة وأرسلها لي لنحل المشكلة فوراً.")
