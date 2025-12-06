import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import base64

# --- 1. إعداد الصفحة والوضع الليلي الاحترافي ---
st.set_page_config(page_title="Saudi Market Pro", layout="wide", initial_sidebar_state="expanded")

# CSS متقدم لتحسين الواجهة
st.markdown("""
<style>
    /* تحسين الخلفية والخطوط */
    .stApp { background-color: #0e1117; color: #FAFAFA; }
    
    /* تنسيق الجداول لتشبه منصات التداول */
    .stDataFrame { border: 1px solid #30333d; border-radius: 5px; }
    div[data-testid="stDataFrame"] div[class*="css"] { background-color: #161b24; color: white; }
    
    /* البطاقات العلوية Metrics */
    div[data-testid="stMetric"] {
        background-color: #1d212b;
        border: 1px solid #30333d;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.5);
    }
    div[data-testid="stMetricValue"] { color: #4CAF50; font-weight: bold; }
    
    /* الأزرار */
    .stButton > button {
        background: linear-gradient(45deg, #2962ff, #1e88e5);
        color: white;
        border: none;
        font-weight: bold;
    }
    .stButton > button:hover { opacity: 0.9; }
    
    /* التبويبات */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: #0e1117; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1d212b; border-radius: 4px; color: #b0b3b8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2962ff !important; color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. الإعدادات والقائمة الجانبية ---
with st.sidebar:
    st.title("⚙️ إعدادات الماسح")
    st.caption("تحكم في معايير الاستراتيجية")
    
    RSI_PERIOD = st.number_input("فترة RSI", min_value=7, max_value=30, value=24)
    EMA_PERIOD = st.number_input("فترة EMA", min_value=5, max_value=50, value=8)
    RSI_BUY_LEVEL = st.slider("مستوى دخول RSI", 20, 50, 30)
    
    st.divider()
    st.markdown("### ℹ️ عن الاستراتيجية")
    st.info("""
    **استراتيجية القناص المطور:**
    1. اختراق RSI لمستوى التشبع.
    2. السعر يخترق متوسط EMA.
    3. (جديد) تقاطع MACD إيجابي.
    """)

# --- 3. قائمة الأسهم (مختصرة للسرعة، أضف قائمتك الكاملة هنا) ---
# ملاحظة: يمكنك لصق قائمتك الطويلة هنا بدلاً من هذه العينة
TICKERS = {
    # الطاقة والمواد
    "2222.SR": "أرامكو", "2010.SR": "سابك", "1211.SR": "معادن", "2020.SR": "سابك للمغذيات",
    "4030.SR": "البحري", "4200.SR": "الدريس", "5110.SR": "الكهرباء", "2310.SR": "سبكيم",
    # البنوك
    "1120.SR": "الراجحي", "1180.SR": "الأهلي", "1010.SR": "الرياض", "1150.SR": "الإنماء",
    "1060.SR": "الأول", "1140.SR": "البلاد",
    # الاتصالات والتقنية
    "7010.SR": "STC", "7020.SR": "موبايلي", "7030.SR": "زين", "7202.SR": "علم",
    "7200.SR": "سلوشنز",
    # التجزئة والصحة
    "4190.SR": "جرير", "4002.SR": "المواساة", "4013.SR": "سليمان الحبيب", "2280.SR": "المراعي",
    "8010.SR": "التعاونية", "8210.SR": "بوبا",
    # العقار
    "4300.SR": "دار الأركان", "4250.SR": "جبل عمر", "4090.SR": "طيبة", "4321.SR": "المراكز",
    # المؤشر
    "^TASI.SR": "المؤشر العام"
}

# --- 4. الدوال الحسابية المتقدمة ---
def calculate_indicators(df):
    # RSI (RMA Method)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/RSI_PERIOD, min_periods=RSI_PERIOD, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # EMA
    df['EMA'] = df['Close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    
    # MACD (12, 26, 9)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # نسبة التغير اليومي
    df['Change%'] = df['Close'].pct_change() * 100
    
    return df

# --- 5. المنطق الرئيسي (Engine) ---
st.title("📊 منصة التحليل الذكي (Saudi Market Pro)")

col_btn, col_kpi = st.columns([1, 3])
with col_btn:
    start_scan = st.button("🚀 بدء المسح الشامل", use_container_width=True)

if start_scan:
    st.session_state['data'] = []
    st.session_state['signals'] = []
    st.session_state['market_history'] = {} # لحفظ البيانات للرسم
    
    tickers_list = list(TICKERS.keys())
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # تحميل البيانات (دفعة واحدة للسرعة)
    status_text.text("⏳ جاري الاتصال بالسوق وسحب البيانات...")
    try:
        raw_data = yf.download(tickers_list, period="6mo", interval="1d", group_by='ticker', auto_adjust=False, threads=True, progress=False)
        
        if not raw_data.empty:
            for i, symbol in enumerate(tickers_list):
                try:
                    name = TICKERS[symbol]
                    try: df = raw_data[symbol].copy()
                    except: continue

                    col = 'Close' if 'Close' in df.columns else 'Adj Close'
                    if col in df.columns:
                        df = df.rename(columns={col: 'Close'})
                        df = df.dropna()
                        
                        if len(df) > 50:
                            # حساب المؤشرات
                            df = calculate_indicators(df)
                            
                            last_row = df.iloc[-1]
                            prev_row = df.iloc[-2]
                            
                            # حفظ البيانات
                            st.session_state['market_history'][name] = df
                            
                            # تسجيل الملخص
                            st.session_state['data'].append({
                                "الرمز": symbol, "الاسم": name,
                                "السعر": last_row['Close'],
                                "التغير": last_row['Change%'],
                                "RSI": last_row['RSI'],
                                "MACD": last_row['MACD'],
                                "Signal": last_row['Signal_Line']
                            })
                            
                            # --- منطق القناص (Signals) ---
                            # فحص آخر 3 أيام
                            tail = df.tail(4)
                            if len(tail) == 4:
                                rsi_break = False
                                ema_break = False
                                
                                for idx in range(1, 4):
                                    # RSI Crossing Up
                                    if tail['RSI'].iloc[idx-1] <= RSI_BUY_LEVEL and tail['RSI'].iloc[idx] > RSI_BUY_LEVEL:
                                        rsi_break = True
                                    # Price Crossing EMA
                                    if tail['Close'].iloc[idx-1] <= tail['EMA'].iloc[idx-1] and tail['Close'].iloc[idx] > tail['EMA'].iloc[idx]:
                                        ema_break = True
                                
                                if rsi_break and ema_break:
                                    # إضافة شرط MACD إيجابي لزيادة الذكاء
                                    macd_conf = "✅ إيجابي" if last_row['MACD'] > last_row['Signal_Line'] else "⚠️ سلبي"
                                    st.session_state['signals'].append({
                                        "الاسم": name,
                                        "السعر": last_row['Close'],
                                        "RSI": last_row['RSI'],
                                        "MACD": macd_conf,
                                        "الوقت": "آخر 3 أيام"
                                    })
                                    
                except Exception as e: continue
                progress_bar.progress((i + 1) / len(tickers_list))
                
    except Exception as e: st.error(f"حدث خطأ: {e}")
    
    progress_bar.empty()
    status_text.success("تم التحديث!")

# --- 6. لوحة القيادة (Dashboard) ---

if 'data' in st.session_state and st.session_state['data']:
    df_all = pd.DataFrame(st.session_state['data'])
    
    # إحصائيات علوية
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("عدد الشركات", len(df_all))
    kpi2.metric("فرص القناص", len(st.session_state['signals']))
    
    bullish_count = len(df_all[df_all['Change'] > 0])
    kpi3.metric("الشركات الصاعدة 🟢", bullish_count)
    
    bearish_count = len(df_all[df_all['Change'] < 0])
    kpi4.metric("الشركات الهابطة 🔴", bearish_count)
    
    st.markdown("---")
    
    # التبويبات
    tab1, tab2, tab3 = st.tabs(["🎯 إشارات الدخول", "📊 ماسح السوق الشامل", "📈 الشارت التفاعلي"])
    
    # --- TAB 1: القناص ---
    with tab1:
        if st.session_state['signals']:
            st.markdown("#### أسهم حققت شروط الاختراق (RSI + EMA)")
            df_sig = pd.DataFrame(st.session_state['signals'])
            st.dataframe(
                df_sig.style.format({"السعر": "{:.2f}", "RSI": "{:.2f}"})
                .background_gradient(cmap='Greens', subset=['RSI']),
                use_container_width=True
            )
        else:
            st.info("لا توجد إشارات اختراق مؤكدة حالياً.")
            
    # --- TAB 2: الماسح الشامل ---
    with tab2:
        # فلاتر ذكية
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            sort_by = st.selectbox("ترتيب حسب:", ["RSI (الأعلى)", "RSI (الأقل)", "الأكثر ارتفاعاً %", "الأكثر انخفاضاً %"])
        with col_f2:
            search_txt = st.text_input("بحث عن شركة (اكتب الاسم):")
            
        # تطبيق الفلاتر
        df_view = df_all.copy()
        if search_txt:
            df_view = df_view[df_view['الاسم'].str.contains(search_txt)]
            
        if sort_by == "RSI (الأعلى)": df_view = df_view.sort_values('RSI', ascending=False)
        elif sort_by == "RSI (الأقل)": df_view = df_view.sort_values('RSI', ascending=True)
        elif sort_by == "الأكثر ارتفاعاً %": df_view = df_view.sort_values('التغير', ascending=False)
        else: df_view = df_view.sort_values('التغير', ascending=True)
        
        # تلوين الجدول
        def color_change(val):
            color = '#00ff00' if val > 0 else '#ff0000'
            return f'color: {color}'
        
        st.dataframe(
            df_view.style.format({"السعر": "{:.2f}", "التغير": "{:.2f}%", "RSI": "{:.2f}", "MACD": "{:.3f}"})
            .map(color_change, subset=['التغير'])
            .background_gradient(cmap='RdYlGn', subset=['RSI']),
            use_container_width=True,
            height=500
        )
        
        # زر تحميل البيانات
        csv = df_view.to_csv(index=False).encode('utf-8')
        st.download_button("📥 تحميل التقرير (Excel/CSV)", csv, "market_report.csv", "text/csv")

    # --- TAB 3: الشارت التفاعلي ---
    with tab3:
        st.markdown("#### التحليل الفني المتقدم")
        selected_stock = st.selectbox("اختر الشركة لعرض الشارت:", df_all['الاسم'].tolist())
        
        if selected_stock:
            df_chart = st.session_state['market_history'][selected_stock]
            
            # رسم الشارت باستخدام Plotly (شموع يابانية)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.03, subplot_titles=(f'{selected_stock} Price & EMA', 'RSI'), 
                                row_width=[0.2, 0.7])

            # 1. الشموع
            fig.add_trace(go.Candlestick(x=df_chart.index,
                            open=df_chart['Open'], high=df_chart['High'],
                            low=df_chart['Low'], close=df_chart['Close'], name='Price'), row=1, col=1)
            
            # 2. خط EMA
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA'], 
                                     line=dict(color='orange', width=1), name=f'EMA {EMA_PERIOD}'), row=1, col=1)

            # 3. مؤشر RSI
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['RSI'], 
                                     line=dict(color='purple', width=2), name='RSI'), row=2, col=1)
            
            # خطوط 30 و 70 للـ RSI
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

            # تنسيق الخلفية (Dark Mode)
            fig.update_layout(
                template="plotly_dark",
                height=600,
                xaxis_rangeslider_visible=False,
                paper_bgcolor='#161b24',
                plot_bgcolor='#161b24'
            )
            
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 اضغط على زر 'بدء المسح الشامل' لتحميل البيانات.")
