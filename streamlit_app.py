import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import os
import joblib
import time

# --- مكتبات الذكاء الاصطناعي ---
try:
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

# --- استيراد البيانات ---
try:
    from data.saudi_tickers import STOCKS_DB
except ImportError:
    st.error("🚨 ملف البيانات مفقود.")
    st.stop()

TICKERS = {item['symbol']: item['name'] for item in STOCKS_DB}
# تجميع الأسهم حسب القطاع للمحاكاة
SECTORS_DICT = {}
for item in STOCKS_DB:
    sec = item['sector']
    if sec not in SECTORS_DICT: SECTORS_DICT[sec] = []
    SECTORS_DICT[sec].append(item['symbol'])

# --- 1. إعداد الصفحة ---
st.set_page_config(page_title="TASI AI Replay", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif; }
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    
    /* تنسيق البطاقات */
    div[data-testid="stMetric"] {
        background-color: #1d212b; border: 1px solid #333; padding: 15px; border-radius: 12px;
    }
    [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 1.4rem; }
    
    /* الأزرار */
    div.stButton > button {
        background: linear-gradient(90deg, #6200ea, #3700b3); color: white; border: none;
        padding: 12px; width: 100%; border-radius: 8px; font-weight: bold;
    }
    
    /* القوائم */
    .stSelectbox > div > div { background-color: #1e222d; color: white; }
</style>
""", unsafe_allow_html=True)

# --- 2. القائمة العلوية ---
selected_tab = option_menu(
    menu_title=None,
    options=["الرئيسية", "🧪 مختبر المحاكاة (AI Replay)", "الشارت الفني"],
    icons=["house", "fast-forward-circle", "graph-up"],
    default_index=1,
    orientation="horizontal",
    styles={"container": {"background-color": "transparent"}, "nav-link-selected": {"background-color": "#6200ea"}}
)

# --- 3. الدوال المساعدة للذكاء ---
if not os.path.exists('models'): os.makedirs('models')

def prepare_data(df, lookback=60):
    # إضافة المؤشرات (Features)
    df['RSI'] = 100 - (100 / (1 + df['Close'].diff().clip(lower=0).ewm(alpha=1/14).mean() / df['Close'].diff().clip(upper=0).abs().ewm(alpha=1/14).mean()))
    df['EMA'] = df['Close'].ewm(span=20).mean()
    df['Box_High'] = df['High'].rolling(20).max() # محاكاة بسيطة للصندوق
    df.dropna(inplace=True)
    
    if len(df) < lookback + 10: return None, None, None, None
    
    # البيانات المستخدمة في التدريب: إغلاق، RSI، EMA
    dataset = df[['Close', 'RSI', 'EMA']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    x, y = [], []
    for i in range(lookback, len(scaled_data)):
        x.append(scaled_data[i-lookback:i, :])
        y.append(scaled_data[i, 0]) # الهدف: السعر
        
    return np.array(x), np.array(y), scaler, df

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# --- 4. المحرك الرئيسي (Simulation Engine) ---

if selected_tab == "🧪 مختبر المحاكاة (AI Replay)":
    st.title("🧪 مختبر إعادة التشغيل (AI Replay Strategy)")
    st.markdown("هنا نقوم بتدريب الذكاء على 'سهم معلم' ثم نختبره على 'سهم طالب' في نفس القطاع لنرى دقة التوقع.")
    
    if not AI_AVAILABLE:
        st.error("المكتبات غير متوفرة. يرجى تحديث requirements.txt")
        st.stop()

    # واجهة التحكم
    c1, c2, c3 = st.columns(3)
    with c1:
        chosen_sector = st.selectbox("1. اختر القطاع", list(SECTORS_DICT.keys()))
    
    # تصفية الأسهم حسب القطاع
    sector_stocks = SECTORS_DICT[chosen_sector]
    stock_options = {s: TICKERS.get(s, s) for s in sector_stocks}
    
    with c2:
        teacher_sym = st.selectbox("2. سهم التدريب (المعلم)", options=list(stock_options.keys()), format_func=lambda x: stock_options[x])
    with c3:
        student_sym = st.selectbox("3. سهم الاختبار (المحاكاة)", options=list(stock_options.keys()), format_func=lambda x: stock_options[x], index=1 if len(stock_options)>1 else 0)

    # إعدادات المحاكاة
    replay_days = st.slider("فترة المحاكاة (Replay Days)", 30, 90, 60, help="عدد الأيام التي سنخفيها عن الذكاء ونطلب منه توقعها")
    
    if st.button("🚀 تشغيل المحاكاة (Start Replay)"):
        status = st.empty()
        prog = st.progress(0)
        
        try:
            # 1. تدريب المعلم
            status.info(f"جاري تدريب النموذج على بيانات {stock_options[teacher_sym]} لخمس سنوات...")
            df_teacher = yf.download(teacher_sym, period="5y", interval="1d", progress=False)
            
            # تنظيف البيانات
            if isinstance(df_teacher.columns, pd.MultiIndex): df_teacher.columns = df_teacher.columns.get_level_values(0)
            
            x_train, y_train, scaler, _ = prepare_data(df_teacher)
            
            if x_train is None:
                st.error("بيانات المعلم غير كافية.")
                st.stop()
                
            model = build_model((x_train.shape[1], x_train.shape[2]))
            model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=0)
            prog.progress(50)
            
            # 2. اختبار الطالب (Replay)
            status.info(f"جاري تشغيل المحاكاة على {stock_options[student_sym]}...")
            df_student = yf.download(student_sym, period="2y", interval="1d", progress=False)
            if isinstance(df_student.columns, pd.MultiIndex): df_student.columns = df_student.columns.get_level_values(0)
            
            # نأخذ البيانات حتى ما قبل فترة المحاكاة + فترة المحاكاة
            # نحتاج تجهيز البيانات كاملة أولاً للحصول على المؤشرات الصحيحة
            _, _, _, df_student_proc = prepare_data(df_student)
            
            # الآن نقسم البيانات:
            # Real Data: البيانات الحقيقية كاملة
            # Replay Data: آخر (replay_days) يوم
            
            real_prices = df_student_proc['Close'].values[-replay_days:]
            dates = df_student_proc.index[-replay_days:]
            
            predicted_prices = []
            
            # حلقة المحاكاة (يوم بيوم)
            # لكل يوم في فترة المحاكاة، نستخدم الـ 60 يوم التي قبله للتوقع
            full_scaled_data = scaler.transform(df_student_proc[['Close', 'RSI', 'EMA']].values)
            
            for i in range(replay_days):
                # تحديد الـ Window السابقة لهذا اليوم
                # الإندكس الحالي في البيانات الكاملة هو: length - replay_days + i
                curr_idx = len(full_scaled_data) - replay_days + i
                
                # نأخذ الـ 60 يوم السابقة
                input_seq = full_scaled_data[curr_idx-60 : curr_idx]
                input_seq = input_seq.reshape(1, 60, 3) # (1, 60, 3 features)
                
                # التوقع
                pred_scaled = model.predict(input_seq, verbose=0)
                
                # عكس التحجيم
                dummy = np.zeros((1, 3))
                dummy[0, 0] = pred_scaled[0, 0]
                pred_val = scaler.inverse_transform(dummy)[0, 0]
                predicted_prices.append(pred_val)
                
                # تحديث الشريط
                prog.progress(50 + int((i/replay_days)*50))
            
            prog.empty()
            status.success("✅ اكتملت المحاكاة!")
            
            # 3. عرض النتائج
            st.divider()
            
            # حساب الدقة (MAE - Mean Absolute Error)
            mae = np.mean(np.abs(np.array(predicted_prices) - real_prices))
            accuracy = 100 - (mae / np.mean(real_prices) * 100)
            
            # البطاقات
            k1, k2, k3 = st.columns(3)
            k1.metric("دقة المحاكاة", f"{accuracy:.1f}%")
            k2.metric("متوسط الخطأ (ريال)", f"{mae:.2f}")
            trend_match = "✅ متطابق" if (predicted_prices[-1] > predicted_prices[0]) == (real_prices[-1] > real_prices[0]) else "❌ معاكس"
            k3.metric("تطابق الاتجاه العام", trend_match)
            
            # الرسم البياني (Replay Chart)
            fig = go.Figure()
            
            # السعر الحقيقي
            fig.add_trace(go.Scatter(
                x=dates, y=real_prices,
                mode='lines', name='السعر الحقيقي (Real)',
                line=dict(color='#00e676', width=3)
            ))
            
            # توقع الذكاء
            fig.add_trace(go.Scatter(
                x=dates, y=predicted_prices,
                mode='lines', name='توقع الذكاء (AI)',
                line=dict(color='#ff2950', width=2, dash='dot')
            ))
            
            fig.update_layout(
                title=f"نتيجة اختبار المحاكاة على {stock_options[student_sym]}",
                template="plotly_dark", height=500,
                xaxis_title="التاريخ", yaxis_title="السعر",
                legend=dict(orientation="h", y=1.1)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # التحليل النصي
            with st.expander("📝 تقرير المحاكاة التفصيلي"):
                st.write(f"""
                - **المعلم:** {stock_options[teacher_sym]} (تم تدريب النموذج عليه).
                - **الطالب:** {stock_options[student_sym]} (تم اختباره عليه).
                - **النتيجة:** الذكاء الاصطناعي استطاع محاكاة حركة السعر بدقة **{accuracy:.1f}%**.
                - **التفسير:** - إذا كان الخط الأحمر قريباً من الأخضر، فهذا يعني أن سلوك السهمين متشابه وأن استراتيجية الصناديق تعمل بكفاءة في هذا القطاع.
                    - إذا كان هناك تباعد كبير، فهذا يعني أن سهم "{stock_options[student_sym]}" له سلوك شاذ ولا يتبع نمط القطاع العام.
                """)

        except Exception as e:
            st.error(f"حدث خطأ أثناء المحاكاة: {e}")

elif selected_tab == "الرئيسية":
    st.info("انتقل لتبويب 'مختبر المحاكاة' لتجربة ميزة الـ Replay.")
