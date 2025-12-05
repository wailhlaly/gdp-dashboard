import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="RSI Pine Match", layout="wide")
st.title("📊 ماسح RSI (بنفس معادلة Pine Script)")

# --- الإعدادات ---
# تأكد من وضع نفس الرقم الموجود في TradingView
RSI_LENGTH = 24  
# السهم للمقارنة
TARGET_STOCK = "1180.SR" 

# --- ترجمة دالة RMA من Pine Script إلى Python ---
def rma(series, length):
    # RMA في Pine Script تعادل Exponential Moving Average مع alpha = 1/length
    return series.ewm(alpha=1/length, min_periods=length, adjust=False).mean()

# --- دالة حساب RSI المطابقة للكود المرسل ---
def calculate_rsi_pine(close_prices, length):
    # 1. حساب التغير (change(src))
    delta = close_prices.diff()
    
    # 2. تحديد الصعود والهبوط
    # max(change(src), 0)
    up_move = delta.clip(lower=0)
    # -min(change(src), 0) -> لاحظ الإشارة السالبة لقلب الرقم
    down_move = -delta.clip(upper=0)
    
    # 3. تطبيق دالة rma كما في الكود المرسل
    # up = rma(max(change(src), 0), len)
    up_avg = rma(up_move, length)
    # down = rma(-min(change(src), 0), len)
    down_avg = rma(down_move, length)
    
    # 4. حساب RSI
    # rsi = down == 0 ? 100 : up == 0 ? 0 : 100 - (100 / (1 + up / down))
    rs = up_avg / down_avg
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

if st.button(f"احسب RSI ({RSI_LENGTH}) للسهم {TARGET_STOCK}"):
    
    st.write("1. جاري تحميل البيانات التاريخية لضمان دقة دالة RMA...")
    # ملاحظة هامة: يجب تحميل بيانات كافية (سنتين مثلاً) لكي تستقر دالة rma
    # لن يؤثر هذا على السرعة، لكنه ضروري للدقة الرياضية
    df = yf.download(TARGET_STOCK, period="2y", interval="1d", auto_adjust=False, progress=False)
    
    if not df.empty:
        # استخراج عمود الإغلاق
        try:
            if isinstance(df.columns, pd.MultiIndex):
                close_series = df.xs('Close', level=0, axis=1)[TARGET_STOCK]
            else:
                close_series = df['Close']
        except:
             close_series = df['Close'] # محاولة أخيرة

        close_series = close_series.dropna()
        
        # --- الحساب ---
        rsi_series = calculate_rsi_pine(close_series, RSI_LENGTH)
        
        # استخراج آخر قيمة
        last_rsi = rsi_series.iloc[-1]
        last_price = close_series.iloc[-1]
        
        # --- العرض ---
        st.subheader("النتيجة النهائية:")
        col1, col2 = st.columns(2)
        
        col1.metric("سعر الإغلاق", f"{last_price:.2f}")
        
        # تلوين النتيجة
        rsi_color = "normal"
        if last_rsi > 70: rsi_color = "inverse" # أحمر/تحذير
        elif last_rsi < 30: rsi_color = "normal" # أخضر/جيد
        
        col2.metric(f"RSI ({RSI_LENGTH})", f"{last_rsi:.2f}")
        
        st.success(f"""
        **تم استخدام المعادلة التالية (ترجمة حرفية لكودك):**
        1. Source: Close
        2. Up = RMA(change_up, {RSI_LENGTH})
        3. Down = RMA(change_down, {RSI_LENGTH})
        4. RSI = 100 - (100 / (1 + Up/Down))
        """)

    else:
        st.error("فشل الاتصال بالمصدر.")
