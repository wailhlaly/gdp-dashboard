import streamlit as st
import pandas as pd
import numpy as np
from tvDatafeed import TvDatafeed, Interval

st.set_page_config(page_title="RSI TV Match", layout="wide")
st.title("📊 ماسح RSI (بيانات TradingView مباشرة)")

# --- الإعدادات ---
RSI_PERIOD = 24
# لاحظ: الرموز في TradingView للسوق السعودي لا تحتاج .SR بل تحتاج تحديد السوق TADAWUL
TICKERS_MAP = {
    "1180": "الأهلي",
    "1120": "الراجحي",
    "2222": "أرامكو",
    "2010": "سابك",
    "7010": "STC"
}

# --- دالة معادلة TradingView (RMA) ---
def calculate_rsi_pine(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = series.ewm(alpha=1/period, min_periods=period, adjust=False).mean() # استخدام تقريب EWM
    # للمطابقة التامة نحتاج RMA يدوية، لكن EWM قريبة جداً مع البيانات الطويلة
    
    # التنفيذ اليدوي الدقيق لـ RMA (كما في Pine Script)
    avg_gain = np.zeros_like(series)
    avg_loss = np.zeros_like(series)
    
    # البداية SMA
    avg_gain[period] = gain[1:period+1].mean()
    avg_loss[period] = loss[1:period+1].mean()
    
    # التكملة RMA
    for i in range(period + 1, len(series)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain.iloc[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss.iloc[i]) / period
        
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return pd.Series(rsi, index=series.index)

# --- التشغيل ---
if st.button('🚀 الاتصال بسيرفرات TradingView'):
    
    st.write("جاري الاتصال بـ TradingView (قد يستغرق وقتاً أطول قليلاً من Yahoo)...")
    
    # تهيئة الاتصال (بدون يوزر نيم وباسورد يدخل كزائر)
    tv = TvDatafeed()
    
    results = []
    
    progress_bar = st.progress(0)
    
    for i, (symbol, name) in enumerate(TICKERS_MAP.items()):
        try:
            # سحب البيانات من TADAWUL
            # نطلب 500 شمعة (حوالي سنتين)
            df = tv.get_hist(symbol=symbol, exchange='TADAWUL', interval=Interval.in_daily, n_bars=500)
            
            if df is not None and not df.empty:
                # البيانات تأتي واسم العمود close (صغير) أو close (كبير) حسب النسخة
                # tvDatafeed عادة تعيد الأعمدة كـ: symbol, open, high, low, close, volume
                
                # توحيد اسم العمود
                df.columns = [c.lower() for c in df.columns]
                
                if 'close' in df.columns:
                    close_series = df['close']
                    
                    # حساب RSI
                    rsi_series = calculate_rsi_pine(close_series, RSI_PERIOD)
                    
                    last_rsi = rsi_series.iloc[-1]
                    last_price = close_series.iloc[-1]
                    
                    results.append({
                        "الرمز": symbol,
                        "الاسم": name,
                        "السعر (TV)": round(last_price, 2),
                        f"RSI ({RSI_PERIOD})": round(last_rsi, 2)
                    })
        except Exception as e:
            st.error(f"خطأ في {name}: {e}")
            
        progress_bar.progress((i + 1) / len(TICKERS_MAP))
        
    progress_bar.empty()

    if results:
        st.subheader("النتائج (المصدر: TradingView):")
        df_final = pd.DataFrame(results)
        df_final = df_final.sort_values(by=f"RSI ({RSI_PERIOD})", ascending=False)
        st.dataframe(df_final, use_container_width=True)
    else:
        st.error("فشل في جلب البيانات من TradingView. قد يكون هناك حظر IP مؤقت.")

