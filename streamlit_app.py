import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from scipy.signal import argrelextrema
import os
import joblib
import time

# --- محاولة استيراد مكتبات الذكاء (مع حماية) ---
try:
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

# --- استيراد البيانات ---
try:
    from data.saudi_tickers import STOCKS_DB
except ImportError:
    try:
        from saudi_tickers import STOCKS_DB
    except ImportError:
        st.error("🚨 ملف البيانات مفقود.")
        st.stop()

TICKERS = {item['symbol']: item['name'] for item in STOCKS_DB}
SECTORS = {item['name']: item['sector'] for item in STOCKS_DB}

# --- إعداد المجلدات للعقل الإلكتروني ---
if not os.path.exists('ai_mind'):
    os.makedirs('ai_mind')

# --- 1. إعداد الصفحة ---
st.set_page_config(page_title="TASI AI Mind", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif; }
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    div.stButton > button {
        background: linear-gradient(90deg, #6200ea, #3700b3); color: white; border: none;
        padding: 12px; width: 100%; border-radius: 8px; font-weight: bold;
    }
    div[data-testid="stMetric"] { background-color: #1d212b; border-radius: 10px; border: 1px solid #333; }
</style>
""", unsafe_allow_html=True)

# --- 2. القائمة العلوية ---
selected_tab = option_menu(
    menu_title=None,
    options=["الرئيسية", "🧠 العقل الإلكتروني (AI)", "الشارت الفني"],
    icons=["house", "cpu", "graph-up"],
    default_index=1,
    orientation="horizontal",
    styles={"container": {"background-color": "transparent"}, "nav-link-selected": {"background-color": "#6200ea"}}
)

# --- 3. الإعدادات ---
with st.sidebar:
    st.header("⚙️ إعدادات الاستراتيجية")
    RSI_PERIOD = st.number_input("RSI Period", 14, 30, 24)
    EMA_PERIOD = st.number_input("EMA Trend", 10, 200, 20)
    ATR_MULT = st.number_input("ATR Multiplier", 1.0, 3.0, 1.5)
    
    st.divider()
    st.header("🧠 إعدادات الذكاء")
    EPOCHS = st.slider("دقة التدريب (Epochs)", 5, 50, 15)
    LOOKBACK = st.slider("ذاكرة الذكاء (أيام)", 30, 90, 60)

# --- 4. الدوال الفنية وتجهيز البيانات ---
def calculate_atr(df):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    return ranges.max(axis=1).ewm(alpha=1/14, min_periods=14, adjust=False).mean()

def prepare_data_for_ai(df):
    """
    تجهيز البيانات بحيث يتعلم الذكاء من:
    1. السعر (Close)
    2. المؤشرات (RSI, EMA)
    3. حدود الصندوق (Box Levels) - أهم ميزة
    """
    df = df.copy()
    # المؤشرات الفنية
    df['RSI'] = 100 - (100 / (1 + df['Close'].diff().clip(lower=0).ewm(alpha=1/14).mean() / df['Close'].diff().clip(upper=0).abs().ewm(alpha=1/14).mean()))
    df['EMA'] = df['Close'].ewm(span=20).mean()
    df['ATR'] = calculate_atr(df)
    
    # محاكاة بسيطة لحدود الصندوق ليفهمها الذكاء كأرقام
    # (الذكاء لا يرى الرسم، بل يرى الأرقام، لذا نعطيه أعلى وأدنى سعر لآخر 20 يوم كدلالة على الصندوق)
    df['Box_High'] = df['High'].rolling(window=20).max()
    df['Box_Low'] = df['Low'].rolling(window=20).min()
    
    df.dropna(inplace=True)
    return df

# --- 5. محرك الذكاء الاصطناعي (The Brain) ---

def build_brain_model(input_shape):
    """بناء شبكة عصبية LSTM متقدمة"""
    model = Sequential()
    # الطبقة الأولى: استيعاب الأنماط المعقدة
    model.add(LSTM(units=100, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2)) # منع الحفظ الصم
    
    # الطبقة الثانية: ربط الأنماط ببعضها
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    
    # طبقة التفكير (Dense)
    model.add(Dense(units=25))
    model.add(Dense(units=1)) # المخرج: السعر المتوقع
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_mind(symbol):
    """تدريب العقل وحفظه"""
    status = st.empty()
    status.info(f"جاري جلب 5 سنوات من البيانات لتدريب العقل على {symbol}...")
    
    try:
        df = yf.download(symbol, period="5y", interval="1d", progress=False)
        if len(df) < 500:
            st.error("البيانات التاريخية غير كافية للتدريب.")
            return None

        # تنظيف وتجهيز
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df_processed = prepare_data_for_ai(df)
        
        # الميزات التي سيتعلم منها (Features)
        # السعر + RSI + EMA + حدود الصندوق
        features = ['Close', 'RSI', 'EMA', 'Box_High', 'Box_Low']
        data_values = df_processed[features].values
        
        # التوحيد القياسي (Scaling) بين 0 و 1 (مهم جداً للشبكات العصبية)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data_values)
        
        # تكوين سلاسل زمنية (X, y)
        X_train, y_train = [], []
        for i in range(LOOKBACK, len(scaled_data)):
            X_train.append(scaled_data[i-LOOKBACK:i, :]) # المدخلات: آخر 60 يوم بكل ميزاتها
            y_train.append(scaled_data[i, 0]) # الهدف: سعر الإغلاق لليوم التالي
            
        X_train, y_train = np.array(X_train), np.array(y_train)
        
        # بناء وتدريب النموذج
        status.info(f"🧠 العقل يتدرب الآن... (Epochs: {EPOCHS})")
        model = build_brain_model((X_train.shape[1], X_train.shape[2]))
        
        # Early Stopping: يوقف التدريب إذا لم يتحسن النموذج لتقليل الوقت
        early_stop = EarlyStopping(monitor='loss', patience=3)
        
        model.fit(X_train, y_train, batch_size=32, epochs=EPOCHS, callbacks=[early_stop], verbose=0)
        
        # الحفظ في ملف العقل
        safe_sym = symbol.replace(".SR", "")
        model.save(f'ai_mind/{safe_sym}_model.keras')
        joblib.dump(scaler, f'ai_mind/{safe_sym}_scaler.pkl')
        
        status.success("✅ تم التدريب وحفظ الخبرة في ملف العقل!")
        return df_processed
        
    except Exception as e:
        st.error(f"حدث خطأ أثناء التدريب: {e}")
        return None

def consult_mind(symbol):
    """استشارة العقل للتوقع"""
    safe_sym = symbol.replace(".SR", "")
    model_path = f'ai_mind/{safe_sym}_model.keras'
    scaler_path = f'ai_mind/{safe_sym}_scaler.pkl'
    
    if not os.path.exists(model_path):
        return None, "لا يوجد عقل مدرب لهذا السهم. ابدأ التدريب أولاً."
    
    try:
        # تحميل العقل
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        
        # جلب بيانات الحاضر
        df = yf.download(symbol, period="6mo", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        df_processed = prepare_data_for_ai(df)
        
        # أخذ آخر فترة (الحاضر)
        last_days = df_processed[['Close', 'RSI', 'EMA', 'Box_High', 'Box_Low']].values[-LOOKBACK:]
        
        if len(last_days) < LOOKBACK: return None, "البيانات الحالية غير كافية."
        
        # تجهيز وتوقع
        last_days_scaled = scaler.transform(last_days)
        X_test = np.array([last_days_scaled]) # تحويل لـ 3D array
        
        predicted_scaled = model.predict(X_test, verbose=0)
        
        # عكس التحجيم للحصول على السعر
        # ننشئ مصفوفة وهمية بنفس أبعاد الـ scaler لعكس القيمة الأولى فقط
        dummy = np.zeros((1, 5)) 
        dummy[0, 0] = predicted_scaled[0, 0]
        predicted_price = scaler.inverse_transform(dummy)[0, 0]
        
        return predicted_price, df_processed['Close'].iloc[-1]
        
    except Exception as e:
        return None, str(e)

# --- 6. الواجهة والتشغيل ---

if selected_tab == "🧠 العقل الإلكتروني (AI)":
    st.title("🧠 العقل الإلكتروني (Deep Learning LSTM)")
    st.markdown("""
    هذا النظام يستخدم **التعلم العميق** لفهم سلوك السهم بناءً على استراتيجية الصناديق.
    يقوم بحفظ ما تعلمه في مجلد `ai_mind` ليعود إليه لاحقاً.
    """)
    
    if not AI_AVAILABLE:
        st.error("⚠️ يرجى تثبيت مكتبات الذكاء الاصطناعي (tensorflow, scikit-learn) في ملف requirements.txt")
    else:
        col_sel, col_act = st.columns([2, 1])
        with col_sel:
            target_stock = st.selectbox("اختر السهم", list(TICKERS.keys()), format_func=lambda x: f"{TICKERS[x]} ({x})")
        
        with col_act:
            st.write("") # Spacer
            st.write("")
            train_btn = st.button("🔴 تدريب العقل (Train)")
            predict_btn = st.button("🔮 استشارة العقل (Predict)")
            
        if train_btn:
            with st.spinner("جاري بناء الشبكة العصبية..."):
                _ = train_mind(target_stock)
                
        if predict_btn:
            with st.spinner("العقل يفكر..."):
                pred, current = consult_mind(target_stock)
                if pred:
                    change = ((pred - current) / current) * 100
                    color = "green" if change > 0 else "red"
                    direction = "صعود 📈" if change > 0 else "هبوط 📉"
                    
                    st.divider()
                    c1, c2, c3 = st.columns(3)
                    c1.metric("السعر الحالي", f"{current:.2f}")
                    c2.metric("السعر المتوقع (غداً)", f"{pred:.2f}", f"{change:.2f}%")
                    c3.markdown(f"### الاتجاه: :{color}[{direction}]")
                    
                    # نصيحة بناءً على الصندوق والذكاء
                    st.info(f"💡 **تحليل العقل:** بناءً على تاريخ السهم مع الصناديق والمؤشرات في آخر {LOOKBACK} يوم، يتوقع النظام تحركاً بنسبة {change:.2f}%.")
                else:
                    st.warning(f"تنبيه: {current}") # عرض رسالة الخطأ

# --- تبويب الرئيسية (لعرض القائمة بدون أخطاء) ---
elif selected_tab == "الرئيسية":
    st.title("📊 لوحة السوق (Analysis)")
    if st.button("تحديث البيانات"):
        # كود التحديث المبسط الخالي من الأخطاء
        pass # (يمكنك نسخ كود العرض من الردود السابقة هنا إذا أردت)
    st.info("انتقل لتبويب 'العقل الإلكتروني' لتجربة ميزة الذكاء الاصطناعي.")

