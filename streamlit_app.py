import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, date

# --- إعداد الصفحة ---
st.set_page_config(page_title="ماسح RSI 14", layout="wide")
st.title("📊 ماسح الأسهم السعودية (RSI 14) مع الحفظ الذكي")

# --- إعدادات ---
RSI_PERIOD = 14
FILE_NAME = "tasi_data.csv"

# قائمة ببعض الأسهم القياسية (يمكنك زيادتها لتشمل السوق كاملاً)
TICKERS = {
    "1120.SR": "الراجحي", "2222.SR": "أرامكو", "2010.SR": "سابك",
    "1180.SR": "الأهلي", "7010.SR": "STC", "1150.SR": "الإنماء",
    "1211.SR": "معادن", "2020.SR": "سابك للمغذيات", "4030.SR": "البحري",
    "4190.SR": "جرير", "4200.SR": "الدريس", "2380.SR": "رابغ",
    "1010.SR": "الرياض", "5110.SR": "الكهرباء", "^TASI.SR": "المؤشر العام"
}

# --- دالة حساب RSI يدوياً (DQM) ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    # استخدام معادلة EMA (الأكثر دقة للتحليل الفني)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- دالة إدارة البيانات (الحفظ والاسترجاع) ---
def get_market_data(tickers_dict):
    today_str = date.today().strftime("%Y-%m-%d")
    
    # 1. محاولة القراءة من الملف المحلي
    if os.path.exists(FILE_NAME):
        try:
            # قراءة الملف لمعرفة تاريخ آخر تحديث
            # سنعتمد على "وقت تعديل الملف" في النظام
            file_time = os.path.getmtime(FILE_NAME)
            file_date = datetime.fromtimestamp(file_time).date()
            
            if file_date == date.today():
                st.toast("📂 يتم تحميل البيانات من الملف المحفوظ (سريع)...")
                df = pd.read_csv(FILE_NAME, index_col=0, header=[0, 1], parse_dates=True)
                return df
            else:
                st.toast("⚠️ البيانات قديمة.. جاري التحديث من المصدر...")
        except Exception as e:
            st.warning("حدث خطأ في قراءة الملف، سيتم إعادة التحميل.")

    # 2. التحميل من الإنترنت (في حال عدم وجود ملف أو البيانات قديمة)
    tickers_list = list(tickers_dict.keys())
    st.write("⏳ جاري سحب بيانات سنة كاملة لضمان دقة RSI...")
    
    # تحميل جماعي سريع
    df = yf.download(tickers_list, period="1y", group_by='ticker', progress=True)
    
    # حفظ البيانات للمرات القادمة
    if not df.empty:
        df.to_csv(FILE_NAME)
        st.success(f"✅ تم تحديث البيانات وحفظها في {FILE_NAME}")
    
    return df

# --- تشغيل البرنامج ---
if st.button('🚀 تشغيل الفحص'):
    
    data_master = get_market_data(TICKERS)
    
    if data_master is not None and not data_master.empty:
        results = []
        
        # شريط تقدم
        progress_bar = st.progress(0)
        total_stocks = len(TICKERS)
        
        for i, (symbol, name) in enumerate(TICKERS.items()):
            try:
                # استخراج بيانات السهم الواحد من الجدول الكبير
                # ملاحظة: yfinance multi-index structure: (Ticker, PriceType) or (PriceType, Ticker)
                # نحاول الوصول للبيانات بمرونة
                try:
                    df_stock = data_master[symbol].copy()
                except KeyError:
                    continue # السهم غير موجود

                # تنظيف
                if 'Close' in df_stock.columns:
                    series = df_stock['Close']
                elif 'Adj Close' in df_stock.columns:
                    series = df_stock['Adj Close']
                else:
                    continue
                
                series = series.dropna()

                if len(series) > RSI_PERIOD:
                    # حساب RSI
                    rsi_series = calculate_rsi(series, period=RSI_PERIOD)
                    last_rsi = rsi_series.iloc[-1]
                    last_price = series.iloc[-1]
                    
                    if not np.isnan(last_rsi):
                        results.append({
                            "الرمز": symbol,
                            "الاسم": name,
                            "السعر": last_price,
                            "RSI (14)": last_rsi
                        })
            except Exception as e:
                pass
            
            # تحديث الشريط
            progress_bar.progress((i + 1) / total_stocks)
        
        progress_bar.empty()

        # --- العرض والترتيب ---
        if results:
            df_final = pd.DataFrame(results)
            # الترتيب: من الأكبر للأصغر
            df_final = df_final.sort_values(by="RSI (14)", ascending=False)
            
            # التلوين
            def color_rsi(val):
                color = 'black'
                weight = 'normal'
                if val >= 70:
                    color = '#d32f2f' # أحمر غامق
                    weight = 'bold'
                elif val <= 30:
                    color = '#388e3c' # أخضر غامق
                    weight = 'bold'
                return f'color: {color}; font-weight: {weight}'

            st.dataframe(
                df_final.style.map(color_rsi, subset=['RSI (14)'])
                        .format({"السعر": "{:.2f}", "RSI (14)": "{:.2f}"}),
                use_container_width=True,
                height=600
            )
            
            # زر لتحميل النتائج
            csv = df_final.to_csv(index=False).encode('utf-8')
            st.download_button("📥 تحميل قائمة النتائج CSV", csv, "rsi_scan_results.csv", "text/csv")
            
        else:
            st.error("لم يتم العثور على نتائج.")
    else:
        st.error("فشل في جلب البيانات.")
else:
    st.info("اضغط الزر للبدء. سيتم حفظ البيانات لتسريع العمليات القادمة.")

