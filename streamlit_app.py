import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

# عنوان الصفحة
st.title("تجربة منصة الأسهم السعودية 🇸🇦")

# قائمة بسيطة للاختبار
option = st.selectbox(
    'اختر السهم للعرض:',
    ('^TASI.SR', '1120.SR', '2222.SR', '2010.SR'),
    format_func=lambda x: "المؤشر العام" if x == "^TASI.SR" else (
        "الراجحي" if x == "1120.SR" else ("أرامكو" if x == "2222.SR" else "سابك")
    )
)

# جلب البيانات (آخر 3 شهور)
st.write(f"جاري جلب بيانات {option}...")
df = yf.download(option, period="3mo", interval="1d")

# التأكد من وجود بيانات
if not df.empty:
    # عرض السعر الحالي (آخر إغلاق)
    current_price = df['Close'].iloc[-1].item()
    st.metric(label="آخر سعر إغلاق", value=f"{current_price:.2f} ر.س")

    # رسم الشارت التفاعلي
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'])])
    
    fig.update_layout(title=f'شارت {option}', xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("عذراً، لم نتمكن من جلب البيانات حالياً.")

