import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# --- إعداد الصفحة ---
st.set_page_config(page_title="ماسح السوق السعودي", layout="wide")
st.title("✅ ماسح RSI (بدون مكتبات خارجية)")

# --- دالة حساب RSI يدوياً (لتجنب مشاكل المكتبات) ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- القائمة والبيانات ---
TICKERS = [
    "1120.SR", "2222.SR", "2010.SR", "1180.SR", "7010.SR", 
    "4030.SR", "5110.SR", "4200.SR", "1150.SR", "1010.SR",
    "^TASI.SR"
]

try:
    st.write("جاري جلب البيانات من المصدر...")
    
    # جلب البيانات لجميع الأسهم مرة واحدة
    data = yf.download(TICKERS, period="3mo", group_by='ticker', progress=False)
    
    if data.empty:
        st.error("فشل الاتصال بالمصدر (Yahoo Finance). حاول لاحقاً.")
    else:
        results = []
        
        for ticker in TICKERS:
            try:
                # استخراج بيانات السهم
                df_stock = data[ticker].copy() if ticker in data.columns.levels[0] else pd.DataFrame()
                
                if df_stock.empty:
                    # محاولة أخرى في حال لم يكن MultiIndex
                    if ticker in data.columns: 
                        df_stock = data # حالة سهم واحد
                    else:
                        continue

                # التأكد من وجود عمود الإغلاق
                close_col = 'Close' if 'Close' in df_stock.columns else 'Adj Close'
                
                # تنظيف البيانات
                df_stock = df_stock.dropna(subset=[close_col])

                if len(df_stock) > 20:
                    # تطبيق معادلة RSI اليدوية
                    df_stock['RSI'] = calculate_rsi(df_stock[close_col])
                    
                    last_rsi = df_stock['RSI'].iloc[-1]
                    last_price = df_stock[close_col].iloc[-1]
                    
                    results.append({
                        "الرمز": ticker,
                        "السعر": last_price,
                        "RSI": last_rsi
                    })
            except Exception as e:
                continue # تخطي السهم الذي فيه مشكلة

        # --- عرض الجدول النهائي ---
        if results:
            df_final = pd.DataFrame(results)
            df_final = df_final.sort_values(by="RSI", ascending=False)
            
            # دالة التلوين البسيطة
            def highlight_rsi(val):
                color = 'black'
                if val > 70: color = 'red'
                elif val < 30: color = 'green'
                return f'color: {color}'

            st.subheader("النتائج:")
            st.dataframe(
                df_final.style.map(highlight_rsi, subset=['RSI'])
                        .format({"السعر": "{:.2f}", "RSI": "{:.2f}"}),
                use_container_width=True
            )
        else:
            st.warning("لم يتم العثور على بيانات كافية.")

except Exception as e:
    st.error(f"حدث خطأ غير متوقع: {e}")

