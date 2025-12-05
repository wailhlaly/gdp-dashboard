import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="مفتش البيانات", layout="wide")
st.title("🕵️‍♂️ مفتش البيانات: مقارنة الأسعار مع TradingView")

# السهم الذي فيه المشكلة (الأهلي)
target_symbol = "1180.SR" 

# دالة معادلة RSI الخاصة بـ TradingView
def calculate_rsi_wilder(series, period=24):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

if st.button("جلب بيانات البنك الأهلي (1180.SR)"):
    st.write("جاري سحب البيانات الخام من المصدر...")
    
    # سحب البيانات (محاولة منع التعديل التلقائي)
    df = yf.download(target_symbol, period="3mo", interval="1d", auto_adjust=False, progress=False)
    
    if not df.empty:
        # التعامل مع اختلاف هيكلة البيانات
        if isinstance(df.columns, pd.MultiIndex):
            # إذا كانت البيانات تحتوي على MultiIndex (مثل Price, Ticker)
            try:
                # نحاول الوصول للبيانات بشكل مباشر
                close_col = df['Close']
                # إذا كان العمود لا يزال إطار بيانات (DataFrame)، نحوله لسلسلة (Series)
                if isinstance(close_col, pd.DataFrame):
                    close_col = close_col.iloc[:, 0]
            except:
                 close_col = df.iloc[:, 0] # محاولة احتياطية
        else:
            close_col = df['Close']

        # حساب RSI
        rsi_series = calculate_rsi_wilder(close_col)
        
        # تجهيز جدول المقارنة (آخر 5 أيام)
        st.subheader("🧐 قارن هذه الأسعار مع شمعات TradingView:")
        
        last_5_days = []
        for i in range(5):
            idx = -(i+1) # العد العكسي
            date_val = close_col.index[idx].strftime('%Y-%m-%d')
            price_val = close_col.iloc[idx]
            rsi_val = rsi_series.iloc[idx]
            
            last_5_days.append({
                "التاريخ": date_val,
                "سعر الإغلاق في الكود": round(float(price_val), 2),
                "قيمة RSI المحسوبة": round(float(rsi_val), 2)
            })
            
        # عرض الجدول
        df_display = pd.DataFrame(last_5_days)
        st.table(df_display)
        
        st.warning("""
        **التشخيص:**
        1. انظر لصف **أحدث تاريخ**.
        2. هل **"سعر الإغلاق في الكود"** يطابق إغلاق الشمعة في TradingView؟
        
        - إذا كان السعر **مختلفاً**: فالمشكلة في Yahoo Finance (بيانات غير دقيقة/معدلة).
        - إذا كان السعر **مطابقاً** ولكن RSI مختلف: فالمشكلة في المعادلة (وهذا مستبعد الآن).
        """)
        
    else:
        st.error("لم يتم جلب بيانات.")
