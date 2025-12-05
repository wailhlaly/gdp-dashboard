import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="RSI Debugger", layout="wide")
st.title("🕵️‍♂️ كاشف الأخطاء: لماذا يختلف الرقم؟")

# --- الإعدادات ---
RSI_PERIOD = 24
TARGET_STOCK = "1180.SR"  # البنك الأهلي

# --- 1. معادلة TradingView الدقيقة (مع الذاكرة) ---
def rsi_tradingview_logic(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # محاكاة دالة RMA في TradingView
    # تبدأ بمتوسط بسيط SMA ثم تكمل بالمتوسط الأسي
    avg_gain = np.zeros_like(series)
    avg_loss = np.zeros_like(series)
    
    # البداية: متوسط بسيط
    avg_gain[period] = gain[1:period+1].mean()
    avg_loss[period] = loss[1:period+1].mean()
    
    # التكملة: متوسط أسي
    for i in range(period + 1, len(series)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain.iloc[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss.iloc[i]) / period
        
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return pd.Series(rsi, index=series.index)

# --- التشغيل ---
if st.button(f"افحص بيانات {TARGET_STOCK}"):
    st.info("جاري سحب البيانات...")
    
    # نسحب بيانات كافية (سنة) لكي تعمل المعادلة بشكل صحيح
    df = yf.download(TARGET_STOCK, period="1y", interval="1d", auto_adjust=False, progress=False)
    
    if not df.empty:
        # تجهيز البيانات
        try:
            close_series = df.xs('Close', level=0, axis=1)[TARGET_STOCK]
        except:
            close_series = df['Close']
            
        close_series = close_series.dropna()
        
        # حساب RSI
        rsi_series = rsi_tradingview_logic(close_series, RSI_PERIOD)
        
        # --- عرض جدول "الحقيقة" ---
        st.subheader("🧐 دقق في هذا الجدول:")
        st.write("قارن آخر صف في الجدول مع شاشة TradingView:")

        # نأخذ آخر 5 أيام
        last_5 = pd.DataFrame({
            'التاريخ': close_series.index[-5:].strftime('%Y-%m-%d'),
            'سعر الإغلاق (Yahoo)': close_series.iloc[-5:].values.round(2),
            f'قيمة RSI ({RSI_PERIOD})': rsi_series.iloc[-5:].values.round(2)
        })
        
        st.table(last_5)
        
        # استنتاج تلقائي
        last_date_code = last_5.iloc[-1]['التاريخ']
        last_price_code = last_5.iloc[-1]['سعر الإغلاق (Yahoo)']
        
        st.warning(f"""
        **التشخيص:**
        1. **التاريخ:** الكود يقرأ آخر شمعة بتاريخ: **{last_date_code}**. هل هذا هو تاريخ اليوم؟
           - (إذا كان تاريخ أمس، فهذا هو سبب اختلاف الرقم، Yahoo متأخر).
        2. **السعر:** الكود يرى السعر: **{last_price_code}**. هل يطابق السعر في شاشتك؟
           - (إذا كان السعر مختلفاً، فالرقم الناتج سيكون مختلفاً حتماً).
        """)
        
    else:
        st.error("لم يتم جلب بيانات.")
