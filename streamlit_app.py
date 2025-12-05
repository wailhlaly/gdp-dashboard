import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# --- إعداد الصفحة ---
st.set_page_config(page_title="ماسح السوق السعودي", layout="wide")
st.title("📊 ماسح RSI 24 (النسخة المستقرة)")

# --- الإعدادات ---
RSI_PERIOD = 24

# قائمة الأسهم (يمكنك إضافة المزيد)
TICKERS = {
    "1180.SR": "الأهلي",
    "1120.SR": "الراجحي",
    "2222.SR": "أرامكو",
    "2010.SR": "سابك",
    "7010.SR": "STC",
    "1150.SR": "الإنماء",
    "1211.SR": "معادن",
    "4030.SR": "البحري",
    "4200.SR": "الدريس",
    "^TASI.SR": "المؤشر العام"
}

# --- دالة RMA (مطابقة لـ Pine Script) ---
def calculate_rsi_rma(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # محاكاة دالة RMA بدقة باستخدام EWM
    # alpha = 1/period هي المعادلة الرياضية لـ RMA
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- زر التشغيل ---
if st.button('🔄 تحديث البيانات'):
    
    st.info("جاري سحب البيانات من Yahoo Finance (سنتين لضمان دقة المعادلة)...")
    
    # سحب البيانات
    try:
        data = yf.download(list(TICKERS.keys()), period="2y", interval="1d", group_by='ticker', auto_adjust=False, progress=True)
    except Exception as e:
        st.error("خطأ في الاتصال بالمصدر.")
        st.stop()

    if not data.empty:
        results = []
        progress_bar = st.progress(0)
        
        for i, (symbol, name) in enumerate(TICKERS.items()):
            try:
                # استخراج البيانات
                try:
                    df = data[symbol].copy()
                except KeyError:
                    continue

                # تحديد عمود الإغلاق
                if 'Close' in df.columns:
                    series = df['Close']
                elif 'Adj Close' in df.columns:
                    series = df['Adj Close']
                else:
                    continue
                
                series = series.dropna()

                # شرط البيانات الكافية
                if len(series) > RSI_PERIOD + 20:
                    
                    # حساب RSI
                    rsi_series = calculate_rsi_rma(series, RSI_PERIOD)
                    
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

        # --- عرض النتائج ---
        if results:
            df_final = pd.DataFrame(results)
            col_rsi = f"RSI ({RSI_PERIOD})"
            
            # ترتيب من الأكبر (تشبع شرائي) للأصغر
            df_final = df_final.sort_values(by=col_rsi, ascending=False)
            
            # تلوين
            def color_rsi(val):
                color = 'black'
                weight = 'normal'
                if val >= 70: 
                    color = '#d32f2f' # أحمر
                    weight = 'bold'
                elif val <= 30: 
                    color = '#388e3c' # أخضر
                    weight = 'bold'
                return f'color: {color}; font-weight: {weight}'

            st.dataframe(
                df_final.style.map(color_rsi, subset=[col_rsi])
                        .format({"السعر": "{:.2f}", col_rsi: "{:.2f}"}),
                use_container_width=True,
                height=600
            )
            
            st.warning("⚠️ تنبيه: القيم تعتمد على بيانات Yahoo Finance المجانية وقد تختلف قليلاً عن TradingView.")
        else:
            st.error("لا توجد نتائج لعرضها.")
    else:
        st.error("فشل جلب البيانات.")
