import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# --- 1. إعداد الصفحة والوضع الليلي (Dark Mode) ---
st.set_page_config(page_title="Saudi Pro Dark", layout="wide", initial_sidebar_state="expanded")

# CSS متقدم لإجبار الألوان الداكنة وإصلاح المظهر الأبيض
st.markdown("""
<style>
    /* إجبار الخلفية الداكنة على كامل التطبيق */
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    
    /* إصلاح ألوان الجداول */
    .stDataFrame {
        border: 1px solid #30333d;
    }
    div[data-testid="stDataFrame"] div[class*="css"] {
        background-color: #161b24;
        color: white;
    }
    
    /* إصلاح ألوان البطاقات (Metrics) لتكون داكنة */
    div[data-testid="stMetric"] {
        background-color: #1d212b !important;
        border: 1px solid #30333d;
        padding: 15px;
        border-radius: 8px;
        color: white !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #b0b3b8 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    
    /* الأزرار */
    div.stButton > button {
        background-color: #2962ff;
        color: white;
        border: none;
        width: 100%;
        font-weight: bold;
    }
    div.stButton > button:hover {
        background-color: #1e53e5;
    }
    
    /* التبويبات */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1d212b;
        color: #e0e0e0;
        border-radius: 4px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2962ff !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. الإعدادات والقائمة الجانبية ---
with st.sidebar:
    st.title("⚙️ الإعدادات")
    RSI_PERIOD = st.number_input("فترة RSI", value=24)
    EMA_PERIOD = st.number_input("فترة EMA", value=8)
    st.info("اضغط زر التشغيل لبدء التحليل.")

# --- 3. القائمة الكاملة (تم إدراج العينة، يمكنك لصق قائمتك الكاملة هنا) ---
# سأضع أهم الأسهم هنا لتضمن عمل الكود، أضف باقي الـ 200 شركة هنا
TICKERS = {
    # بنوك
    "1180.SR": "الأهلي", "1120.SR": "الراجحي", "1010.SR": "الرياض", "1150.SR": "الإنماء", 
    # طاقة ومواد
    "2222.SR": "أرامكو", "2010.SR": "سابك", "1211.SR": "معادن", "2020.SR": "سابك للمغذيات", "4030.SR": "البحري",
    # اتصالات وتقنية
    "7010.SR": "STC", "7020.SR": "موبايلي", "7202.SR": "علم",
    # تجزئة وخدمات
    "4190.SR": "جرير", "4200.SR": "الدريس", "4002.SR": "المواساة", "2280.SR": "المراعي",
    # مؤشر
    "^TASI.SR": "المؤشر العام"
}

# --- 4. الدوال الحسابية ---
def calculate_indicators(df):
    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # EMA
    df['EMA'] = df['Close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # نسبة التغير (Change) - نستخدم مفتاح إنجليزي لضمان عدم حدوث KeyError
    df['Change'] = df['Close'].pct_change() * 100
    
    return df

# --- 5. التشغيل والواجهة ---
st.title("📊 محلل السوق السعودي (النسخة المستقرة)")

# إدارة الذاكرة
if 'data' not in st.session_state: st.session_state['data'] = []
if 'signals' not in st.session_state: st.session_state['signals'] = []
if 'history' not in st.session_state: st.session_state['history'] = {}

if st.button("🚀 تحديث البيانات (Scan Market)"):
    # تصفير البيانات القديمة
    st.session_state['data'] = []
    st.session_state['signals'] = []
    st.session_state['history'] = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    tickers_list = list(TICKERS.keys())
    
    # تحميل البيانات
    try:
        status_text.text("جاري الاتصال بقاعدة البيانات...")
        raw_data = yf.download(tickers_list, period="6mo", interval="1d", group_by='ticker', auto_adjust=False, threads=True, progress=False)
        
        if not raw_data.empty:
            for i, symbol in enumerate(tickers_list):
                try:
                    name = TICKERS[symbol]
                    try: df = raw_data[symbol].copy()
                    except: continue

                    # توحيد اسم العمود
                    col = 'Close' if 'Close' in df.columns else 'Adj Close'
                    if col in df.columns:
                        df = df.rename(columns={col: 'Close'})
                        df = df.dropna()
                        
                        if len(df) > 50:
                            df = calculate_indicators(df)
                            
                            last_row = df.iloc[-1]
                            
                            # حفظ البيانات للشارت
                            st.session_state['history'][name] = df
                            
                            # تخزين الملخص (استخدام مفاتيح إنجليزية حصراً هنا لمنع KeyError)
                            st.session_state['data'].append({
                                "Name": name,
                                "Symbol": symbol,
                                "Price": last_row['Close'],
                                "Change": last_row['Change'], # تصحيح الخطأ السابق
                                "RSI": last_row['RSI'],
                                "MACD": last_row['MACD'],
                                "Signal_Line": last_row['Signal_Line']
                            })
                            
                            # منطق الإشارات
                            tail = df.tail(4)
                            if len(tail) == 4:
                                rsi_break = False
                                ema_break = False
                                for idx in range(1, 4):
                                    if tail['RSI'].iloc[idx-1] <= 30 and tail['RSI'].iloc[idx] > 30: rsi_break = True
                                    if tail['Close'].iloc[idx-1] <= tail['EMA'].iloc[idx-1] and tail['Close'].iloc[idx] > tail['EMA'].iloc[idx]: ema_break = True
                                
                                if rsi_break and ema_break:
                                    macd_status = "✅" if last_row['MACD'] > last_row['Signal_Line'] else "⚠️"
                                    st.session_state['signals'].append({
                                        "الاسم": name, "السعر": last_row['Close'], "RSI": last_row['RSI'], "MACD": macd_status
                                    })
                                    
                except: continue
                progress_bar.progress((i + 1) / len(tickers_list))
                
    except Exception as e: st.error(f"خطأ في التحميل: {e}")
    
    progress_bar.empty()
    status_text.empty()

# --- 6. عرض النتائج ---
if st.session_state['data']:
    df_all = pd.DataFrame(st.session_state['data'])
    
    # الإحصائيات العلوية
    k1, k2, k3 = st.columns(3)
    k1.metric("عدد الشركات", len(df_all))
    k2.metric("فرص القناص", len(st.session_state['signals']))
    
    # هنا تم إصلاح الـ KeyError باستخدام العمود الصحيح 'Change'
    bullish_count = len(df_all[df_all['Change'] > 0])
    k3.metric("شركات خضراء 🟢", bullish_count)
    
    st.markdown("---")
    
    t1, t2, t3 = st.tabs(["🎯 الفرص", "📋 السوق", "📈 شارت"])
    
    with t1:
        if st.session_state['signals']:
            st.dataframe(pd.DataFrame(st.session_state['signals']), use_container_width=True)
        else:
            st.info("لا توجد إشارات حالياً.")
            
    with t2:
        # تجهيز الجدول للعرض (تعريب الأسماء هنا فقط وليس في المنطق)
        display_df = df_all.copy()
        display_df = display_df.rename(columns={
            "Name": "الاسم", "Price": "السعر", "Change": "التغير %", 
            "RSI": f"RSI ({RSI_PERIOD})", "MACD": "MACD"
        })
        
        # اختيار الأعمدة
        cols_to_show = ["الاسم", "السعر", "التغير %", f"RSI ({RSI_PERIOD})", "MACD"]
        
        # التنسيق الشرطي (بدون تعقيدات Jinja2 المفرطة)
        st.dataframe(
            display_df[cols_to_show].style.format({"السعر": "{:.2f}", "التغير %": "{:.2f}%", f"RSI ({RSI_PERIOD})": "{:.2f}"})
            .background_gradient(cmap='RdYlGn', subset=['التغير %']),
            use_container_width=True, height=500
        )
        
    with t3:
        sel = st.selectbox("اختر سهم:", df_all['Name'].unique())
        if sel:
            df_chart = st.session_state['history'][sel]
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_width=[0.3, 0.7])
            fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name='Price'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA'], line=dict(color='orange'), name='EMA'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['RSI'], line=dict(color='purple'), name='RSI'), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False, paper_bgcolor='#161b24', plot_bgcolor='#161b24')
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("اضغط زر التحديث للبدء.")

