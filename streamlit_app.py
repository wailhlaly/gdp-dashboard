import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, date

# --- إعداد الصفحة ---
st.set_page_config(page_title="RSI Pro Checker", layout="wide")
st.title("📊 ماسح RSI الاحترافي (مطابق لـ TradingView)")

# --- الإعدادات ---
# تأكد أن هذا الرقم يطابق الرقم الذي وضعته في TradingView للمقارنة
RSI_PERIOD = 24  
FILE_NAME = "tasi_data_tv_match.csv"

# قائمة الأسهم
TICKERS = {
    "1180.SR": "الأهلي", # السهم الذي في صورتك
    "1120.SR": "الراجحي", "2222.SR": "أرامكو", "2010.SR": "سابك",
    "7010.SR": "STC", "1150.SR": "الإنماء", "1211.SR": "معادن",
    "2020.SR": "سابك للمغذيات", "4030.SR": "البحري", "4190.SR": "جرير",
    "4200.SR": "الدريس", "2380.SR": "رابغ", "1010.SR": "الرياض",
    "5110.SR": "الكهرباء", "^TASI.SR": "المؤشر العام"
}

# --- دالة حساب RSI (Wilder's Smoothing) ---
# هذه هي المعادلة السرية التي تستخدمها TradingView
def calculate_rsi_wilder(series, period):
    delta = series.diff()
    
    # فصل الأرباح والخسائر
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # استخدام معادلة Wilder بدلاً من EMA العادية
    # alpha = 1 / period هي المفتاح للتطابق مع TradingView
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- إدارة البيانات ---
def get_data():
    if os.path.exists(FILE_NAME):
        try:
            file_time = os.path.getmtime(FILE_NAME)
            if datetime.fromtimestamp(file_time).date() == date.today():
                st.toast("📂 تحميل بيانات محفوظة...")
                return pd.read_csv(FILE_NAME, index_col=0, header=[0, 1], parse_dates=True)
        except:
            pass

    st.write("⏳ جاري سحب البيانات (نسخة Close الخام)...")
    # auto_adjust=False يضمن الحصول على السعر الخام بدون تعديل التوزيعات
    df = yf.download(list(TICKERS.keys()), period="2y", interval="1d", group_by='ticker', auto_adjust=False, progress=True)
    
    if not df.empty:
        df.to_csv(FILE_NAME)
    return df

# --- التشغيل ---
if st.button('🚀 احسب RSI'):
    
    data = get_data()
    
    if data is not None and not data.empty:
        results = []
        
        for symbol, name in TICKERS.items():
            try:
                # استخراج البيانات
                try:
                    df_stock = data[symbol].copy()
                except KeyError:
                    continue

                # الخطوة الأهم: تحديد عمود الإغلاق الصحيح
                # TradingView يستخدم 'Close' وليس 'Adj Close'
                if 'Close' in df_stock.columns:
                    series = df_stock['Close']
                elif 'Adj Close' in df_stock.columns:
                    series = df_stock['Adj Close'] # بديل اضطراري
                else:
                    continue
                
                series = series.dropna()

                if len(series) > RSI_PERIOD:
                    # الحساب بالمعادلة الجديدة
                    rsi_series = calculate_rsi_wilder(series, period=RSI_PERIOD)
                    
                    last_rsi = rsi_series.iloc[-1]
                    last_price = series.iloc[-1]
                    
                    if not np.isnan(last_rsi):
                        results.append({
                            "الرمز": symbol,
                            "الاسم": name,
                            "السعر": last_price,
                            f"RSI ({RSI_PERIOD})": last_rsi
                        })
            except Exception as e:
                pass
        
        # عرض النتائج
        if results:
            df_final = pd.DataFrame(results)
            col_rsi = f"RSI ({RSI_PERIOD})"
            df_final = df_final.sort_values(by=col_rsi, ascending=False)
            
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
            st.error("لا توجد بيانات.")

