import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, date

# --- إعداد الصفحة ---
st.set_page_config(page_title="ماسح RSI 24", layout="wide")
st.title("📊 ماسح الأسهم السعودية (RSI 24) - الفريم اليومي")

# --- الإعدادات الجديدة ---
RSI_PERIOD = 24  # تم التعديل إلى 24 حسب طلبك
TIMEFRAME = "1d" # فريم يومي
FILE_NAME = "tasi_data_rsi24.csv" # غيرنا اسم الملف لكي لا يتعارض مع القديم

# قائمة الأسهم (عينة)
TICKERS = {
    "1120.SR": "الراجحي", "2222.SR": "أرامكو", "2010.SR": "سابك",
    "1180.SR": "الأهلي", "7010.SR": "STC", "1150.SR": "الإنماء",
    "1211.SR": "معادن", "2020.SR": "سابك للمغذيات", "4030.SR": "البحري",
    "4190.SR": "جرير", "4200.SR": "الدريس", "2380.SR": "رابغ",
    "1010.SR": "الرياض", "5110.SR": "الكهرباء", "^TASI.SR": "المؤشر العام"
}

# --- دالة حساب RSI ---
def calculate_rsi(series, period):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    # استخدام EMA
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- دالة إدارة البيانات ---
def get_market_data(tickers_dict):
    # 1. القراءة من الملف المحلي
    if os.path.exists(FILE_NAME):
        try:
            file_time = os.path.getmtime(FILE_NAME)
            file_date = datetime.fromtimestamp(file_time).date()
            
            if file_date == date.today():
                st.toast("📂 تحميل بيانات اليومي المحفوظة...")
                df = pd.read_csv(FILE_NAME, index_col=0, header=[0, 1], parse_dates=True)
                return df
            else:
                st.toast("⚠️ تحديث البيانات لليوم الجديد...")
        except:
            pass

    # 2. التحميل من الإنترنت
    tickers_list = list(tickers_dict.keys())
    st.write(f"⏳ جاري سحب بيانات يومية (Interval: {TIMEFRAME}) لحساب RSI {RSI_PERIOD}...")
    
    # نحدد interval="1d" صراحةً للفريم اليومي
    # period="2y" لضمان وجود شمعات كافية لمعادلة 24 يوم
    df = yf.download(tickers_list, period="2y", interval=TIMEFRAME, group_by='ticker', progress=True)
    
    if not df.empty:
        df.to_csv(FILE_NAME)
        st.success("✅ تم التحديث والحفظ")
    
    return df

# --- تشغيل البرنامج ---
if st.button('🚀 تشغيل فحص RSI 24'):
    
    data_master = get_market_data(TICKERS)
    
    if data_master is not None and not data_master.empty:
        results = []
        progress_bar = st.progress(0)
        
        for i, (symbol, name) in enumerate(TICKERS.items()):
            try:
                # استخراج البيانات
                try:
                    df_stock = data_master[symbol].copy()
                except KeyError:
                    continue

                if 'Close' in df_stock.columns:
                    series = df_stock['Close']
                elif 'Adj Close' in df_stock.columns:
                    series = df_stock['Adj Close']
                else:
                    continue
                
                series = series.dropna()

                # شرط: نحتاج بيانات أكثر من فترة الـ RSI
                if len(series) > RSI_PERIOD:
                    
                    # الحساب باستخدام الفترة 24
                    rsi_series = calculate_rsi(series, period=RSI_PERIOD)
                    last_rsi = rsi_series.iloc[-1]
                    last_price = series.iloc[-1]
                    
                    if not np.isnan(last_rsi):
                        results.append({
                            "الرمز": symbol,
                            "الاسم": name,
                            "السعر": last_price,
                            f"RSI ({RSI_PERIOD})": last_rsi
                        })
            except:
                pass
            
            progress_bar.progress((i + 1) / len(TICKERS))
        
        progress_bar.empty()

        # --- العرض ---
        if results:
            df_final = pd.DataFrame(results)
            col_rsi_name = f"RSI ({RSI_PERIOD})"
            
            # ترتيب
            df_final = df_final.sort_values(by=col_rsi_name, ascending=False)
            
            # تلوين (يمكنك تعديل أرقام التشبع هنا إذا أردت)
            # عادة مع طول 24 تصبح الحركة أبطأ، لذا مستويات 70/30 قوية جداً
            def color_rsi(val):
                color = 'black'
                weight = 'normal'
                if val >= 70: # تشبع شرائي قوي
                    color = '#d32f2f'
                    weight = 'bold'
                elif val <= 30: # تشبع بيعي قوي
                    color = '#388e3c'
                    weight = 'bold'
                return f'color: {color}; font-weight: {weight}'

            st.dataframe(
                df_final.style.map(color_rsi, subset=[col_rsi_name])
                        .format({"السعر": "{:.2f}", col_rsi_name: "{:.2f}"}),
                use_container_width=True,
                height=600
            )
            
        else:
            st.error("لا توجد نتائج.")
    else:
        st.error("فشل المصدر.")
else:
    st.info(f"اضغط للبدء (الإعدادات: RSI {RSI_PERIOD} - يومي).")
