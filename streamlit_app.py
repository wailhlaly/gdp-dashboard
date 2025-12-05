import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, date

st.set_page_config(page_title="RSI 24 Precise", layout="wide")
st.title("📊 ماسح RSI 24 (سريع ودقيق)")

# --- الإعدادات ---
RSI_PERIOD = 24
FILE_NAME = "tasi_optimized.csv"

# القائمة
TICKERS = {
    "1180.SR": "الأهلي",
    "1120.SR": "الراجحي",
    "2222.SR": "أرامكو",
    "2010.SR": "سابك",
    "7010.SR": "STC",
    "^TASI.SR": "المؤشر العام"
}

# --- معادلة TradingView (Wilder's Smoothing) ---
def calculate_rsi_wilder(series, period):
    # حساب الفرق
    delta = series.diff()
    
    # فصل الربح والخسارة
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # تحضير المصفوفات
    avg_gain = np.full_like(series, np.nan)
    avg_loss = np.full_like(series, np.nan)
    
    g_values = gain.values
    l_values = loss.values
    
    # الخطوة 1: أول قيمة تكون متوسط بسيط (SMA)
    # نحتاج للتأكد من توفر بيانات كافية
    if len(series) > period:
        avg_gain[period] = g_values[1:period+1].mean()
        avg_loss[period] = l_values[1:period+1].mean()
        
        # الخطوة 2: باقي القيم تكون متوسط أسي (Smoothing)
        for i in range(period + 1, len(series)):
            avg_gain[i] = (g_values[i] + (period - 1) * avg_gain[i-1]) / period
            avg_loss[i] = (l_values[i] + (period - 1) * avg_loss[i-1]) / period
            
    rs = avg_gain / avg_loss
    
    # معادلة RSI النهائية
    np.seterr(divide='ignore', invalid='ignore')
    rsi = 100 - (100 / (1 + rs))
    
    return pd.Series(rsi, index=series.index)

# --- التشغيل ---
if st.button('🚀 تحديث (بيانات سنتين)'):
    
    st.write("جاري سحب بيانات سنتين فقط (كافية للدقة وسريعة)...")
    
    # قمنا بتقليل المدة إلى سنتين "2y" بدلاً من "max"
    # هذا هو الحد الأدنى للحصول على رقم مطابق لـ TradingView
    data = yf.download(list(TICKERS.keys()), period="2y", interval="1d", group_by='ticker', auto_adjust=False, progress=True)
    
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
                
                series = series.dropna()

                # نحتاج بيانات أكثر من 24 يوم لكي تعمل المعادلة
                if len(series) > RSI_PERIOD + 10:
                    
                    rsi_series = calculate_rsi_wilder(series, RSI_PERIOD)
                    
                    last_rsi = rsi_series.iloc[-1]
                    last_price = series.iloc[-1]
                    
                    if not np.isnan(last_rsi):
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
            
            # التلوين
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
            st.error("لا توجد نتائج.")
    else:
        st.error("فشل الاتصال.")

