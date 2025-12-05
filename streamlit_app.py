import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="RSI Compare", layout="wide")
st.title("⚖️ كشف الحقيقة: أيهما يطابق TradingView؟")

# الإعدادات
RSI_PERIOD = 24
TARGET_STOCK = "1180.SR" # البنك الأهلي (سهم المشكلة)

# --- دالة حساب RSI باستخدام EWM (الأدق والأسرع) ---
def calculate_rsi_vectorized(series, period):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # محاكاة Wilder's Smoothing باستخدام alpha=1/N
    # هذه الطريقة تتطابق مع TradingView عند وجود بيانات تاريخية طويلة
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- زر التشغيل ---
if st.button(f"تحليل سهم {TARGET_STOCK} بكافة الطرق"):
    
    st.write("1. جاري جلب جميع البيانات التاريخية (Max History)...")
    # نجلب البيانات الخام والمعدلة معاً
    df = yf.download(TARGET_STOCK, period="max", interval="1d", auto_adjust=False, progress=False)
    
    if not df.empty:
        # التعامل مع هيكلة البيانات المعقدة
        try:
            # محاولة فك MultiIndex إذا وجد
            if isinstance(df.columns, pd.MultiIndex):
                close_raw = df.xs('Close', level=0, axis=1)[TARGET_STOCK]
                close_adj = df.xs('Adj Close', level=0, axis=1)[TARGET_STOCK]
            else:
                close_raw = df['Close']
                close_adj = df['Adj Close']
        except:
             # طريقة بديلة في حال فشل التحديد المباشر
             close_raw = df['Close']
             close_adj = df['Adj Close']

        # حذف القيم المفقودة
        close_raw = close_raw.dropna()
        close_adj = close_adj.dropna()

        # --- الحساب الأول: على السعر الخام (Close) ---
        rsi_raw_series = calculate_rsi_vectorized(close_raw, RSI_PERIOD)
        last_rsi_raw = rsi_raw_series.iloc[-1]
        last_price_raw = close_raw.iloc[-1]

        # --- الحساب الثاني: على السعر المعدل (Adj Close) ---
        rsi_adj_series = calculate_rsi_vectorized(close_adj, RSI_PERIOD)
        last_rsi_adj = rsi_adj_series.iloc[-1]
        last_price_adj = close_adj.iloc[-1]

        # --- عرض النتائج للمقارنة ---
        st.subheader("النتيجة النهائية:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("الخيار 1: السعر الخام (Raw Close)")
            st.metric("السعر", f"{last_price_raw:.2f}")
            st.metric(f"RSI ({RSI_PERIOD})", f"{last_rsi_raw:.2f}")
            st.caption("يستخدم سعر الشاشة كما هو، بدون خصم توزيعات سابقة.")

        with col2:
            st.warning("الخيار 2: السعر المعدل (Adj Close)")
            st.metric("السعر (قد يختلف)", f"{last_price_adj:.2f}")
            st.metric(f"RSI ({RSI_PERIOD})", f"{last_rsi_adj:.2f}")
            st.caption("يخصم الأرباح والمنح تاريخياً (غالباً هذا ما يستخدمه التحليل الفني).")

        st.divider()
        st.write("👆 **قارن الرقمين أعلاه مع شاشة TradingView وأخبرني أيهما طابق الـ 54.17؟**")
        
    else:
        st.error("فشل جلب البيانات.")

