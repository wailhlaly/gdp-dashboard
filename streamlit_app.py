import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from scipy.signal import argrelextrema
import os
import joblib

# محاولة استيراد مكتبات الذكاء الاصطناعي (مع معالجة الأخطاء إذا لم تكن مثبتة)
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

# --- 1. إعداد الصفحة ---
st.set_page_config(page_title="TASI AI Deep Learning", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif; }
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    div[data-testid="stMetric"] { background-color: #1d212b; border: 1px solid #333; border-radius: 10px; padding: 10px; }
    div.stButton > button { background: linear-gradient(90deg, #6200ea, #3700b3); color: white; border: none; padding: 10px; border-radius: 5px; width: 100%; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- 2. القائمة العلوية ---
selected_tab = option_menu(
    menu_title=None,
    options=["الرئيسية", "🧠 تدريب الذكاء (AI)", "الماسح الذكي", "الشارت"],
    icons=["house", "robot", "search", "graph-up"],
    default_index=1, # جعلنا تبويب الذكاء هو الافتراضي
    orientation="horizontal",
    styles={"container": {"background-color": "transparent"}, "nav-link-selected": {"background-color": "#6200ea"}}
)

# --- 3. الإعدادات ---
with st.sidebar:
    st.header("⚙️ إعدادات المحلل")
    RSI_PERIOD = st.number_input("RSI Period", 14, 30, 24)
    EMA_PERIOD = st.number_input("EMA Trend", 10, 200, 20)
    ATR_MULT = st.number_input("ATR Multiplier", 1.0, 3.0, 1.5)
    BOX_LOOKBACK = st.slider("Box History", 10, 100, 25)

# --- 4. الدوال الفنية ---
def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    return ranges.max(axis=1).ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def get_boxes_signal(df):
    """
    تحويل منطق الصناديق إلى إشارة رقمية ليفهمها الذكاء الاصطناعي
    1 = داخل صندوق صاعد
    -1 = داخل صندوق هابط
    0 = لا يوجد
    """
    df['ATR'] = calculate_atr(df)
    signals = np.zeros(len(df))
    box_tops = np.zeros(len(df))
    box_bottoms = np.zeros(len(df))
    
    in_series = False; mode = None; start_open = 0.0; end_close = 0.0
    
    # نحتاج للتكرار لضبط المنطق
    prices = df.reset_index()
    atrs = df['ATR'].values
    
    for i in range(len(prices)):
        row = prices.iloc[i]; close = row['Close']; open_p = row['Open']
        is_green = close > open_p; is_red = close < open_p
        current_atr = atrs[i]
        if np.isnan(current_atr): continue
        
        if not in_series:
            if is_green: in_series = True; mode = 'bull'; start_open = open_p
            elif is_red: in_series = True; mode = 'bear'; start_open = open_p
        elif in_series:
            if mode == 'bull' and is_green: end_close = close
            elif mode == 'bear' and is_red: end_close = close
            elif (mode == 'bull' and is_red) or (mode == 'bear' and is_green):
                final_close = end_close if end_close != 0 else start_open
                if abs(final_close - start_open) >= current_atr * ATR_MULT:
                    # تسجيل الإشارة للأيام القادمة (مثلاً لمدة 20 يوم أو حتى يتم كسره)
                    # للتبسيط هنا، نسجل لحظة تكون الصندوق
                    signals[i] = 1 if mode == 'bull' else -1
                    box_tops[i] = max(start_open, final_close)
                    box_bottoms[i] = min(start_open, final_close)
                
                in_series = True; mode = 'bull' if is_green else 'bear'; start_open = open_p; end_close = close
                
    return signals, box_tops, box_bottoms

def prepare_ai_data(df, lookback=60):
    """تجهيز البيانات لشبكة LSTM"""
    df['Box_Signal'], df['Box_Top'], df['Box_Bottom'] = get_boxes_signal(df)
    df['EMA8'] = df['Close'].ewm(span=8).mean()
    df['EMA20'] = df['Close'].ewm(span=20).mean()
    df['RSI'] = 100 - (100 / (1 + df['Close'].diff().clip(lower=0).ewm(alpha=1/14).mean() / df['Close'].diff().clip(upper=0).abs().ewm(alpha=1/14).mean()))
    
    df = df.dropna()
    
    # الميزات التي سيتعلم منها الذكاء (السعر، الصناديق، المتوسطات)
    features = ['Close', 'Box_Signal', 'EMA8', 'EMA20', 'RSI']
    dataset = df[features].values
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    x_train, y_train = [], []
    for i in range(lookback, len(scaled_data)):
        x_train.append(scaled_data[i-lookback:i, :]) # آخر 60 يوم كمدخلات
        y_train.append(scaled_data[i, 0]) # سعر الإغلاق لليوم التالي كهدف
        
    return np.array(x_train), np.array(y_train), scaler, df

# --- 5. منطق الذكاء الاصطناعي (AI Logic) ---
MODEL_FILE = 'my_ai_model.keras'
SCALER_FILE = 'scaler.pkl'

def train_model(symbol, epochs=5):
    status = st.empty()
    status.info(f"جاري جلب بيانات تاريخية لـ {symbol} للتدريب...")
    
    # جلب بيانات طويلة جداً (5 سنوات) للتدريب الجيد
    df = yf.download(symbol, period="5y", interval="1d", progress=False)
    
    if len(df) < 200:
        st.error("البيانات غير كافية للتدريب العميق.")
        return None, None

    status.info("جاري معالجة البيانات وبناء مصفوفات التعلم...")
    x_train, y_train, scaler, processed_df = prepare_ai_data(df)
    
    # بناء الشبكة العصبية (LSTM)
    model = Sequential()
    # الطبقة الأولى: استيعاب السلاسل الزمنية
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(0.2)) # لمنع الحفظ الصم (Overfitting)
    # الطبقة الثانية
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    # طبقات الإخراج
    model.add(Dense(units=25))
    model.add(Dense(units=1)) # التنبؤ بالسعر
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # التدريب
    status.info(f"بدأ تدريب الشبكة العصبية ({epochs} دورات)... قد يستغرق دقيقة.")
    progress_bar = st.progress(0)
    
    # Custom Callback for Streamlit progress
    from tensorflow.keras.callbacks import Callback
    class StreamlitCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            progress_bar.progress((epoch + 1) / epochs)
            
    history = model.fit(x_train, y_train, batch_size=32, epochs=epochs, callbacks=[StreamlitCallback()], verbose=0)
    
    # الحفظ
    model.save(MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    
    status.success("✅ تم التدريب وحفظ النموذج بنجاح!")
    return model, scaler, processed_df

def predict_next_move(model, scaler, df, lookback=60):
    # تجهيز آخر 60 يوم للتنبؤ بالمستقبل
    features = ['Close', 'Box_Signal', 'EMA8', 'EMA20', 'RSI']
    last_60_days = df[features][-lookback:].values
    last_60_days_scaled = scaler.transform(last_60_days)
    
    X_test = []
    X_test.append(last_60_days_scaled)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 5)) # 5 features
    
    # التنبؤ
    pred_price_scaled = model.predict(X_test)
    
    # عكس التحجيم (Inverse Scaling) للحصول على السعر الحقيقي
    # نحتاج لخدعة صغيرة لأن Scaler يتوقع 5 أعمدة
    pred_extended = np.zeros((1, 5))
    pred_extended[0, 0] = pred_price_scaled[0, 0] # نضع السعر المتوقع في مكانه
    pred_price = scaler.inverse_transform(pred_extended)[0, 0]
    
    return pred_price

# --- 6. العرض (UI) ---

if selected_tab == "🧠 تدريب الذكاء (AI)":
    st.header("🧠 مركز تدريب الذكاء الاصطناعي (Deep Learning)")
    
    if not AI_AVAILABLE:
        st.error("⚠️ مكتبات الذكاء الاصطناعي (Tensorflow/Sklearn) غير مثبتة. يرجى إضافتها لملف requirements.txt")
    else:
        c1, c2 = st.columns([2, 1])
        with c1:
            target_stock = st.selectbox("اختر السهم للتدريب", list(TICKERS.keys()), index=list(TICKERS.keys()).index("1120.SR") if "1120.SR" in TICKERS else 0)
        with c2:
            epochs = st.slider("عدد دورات التعلم (Epochs)", 1, 50, 10, help="زيادة العدد تزيد الدقة لكن تبطئ العملية")
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🚀 ابدأ التدريب الآن"):
                model, scaler, df_hist = train_model(target_stock, epochs)
                if model:
                    st.session_state['ai_df'] = df_hist # حفظ البيانات للعرض
        
        with col_btn2:
            if st.button("🔮 تنبؤ بالسعر القادم"):
                if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
                    try:
                        # تحميل النموذج المحفوظ
                        model = load_model(MODEL_FILE)
                        scaler = joblib.load(SCALER_FILE)
                        
                        # جلب بيانات حديثة
                        df_new = yf.download(target_stock, period="1y", interval="1d", progress=False)
                        # إعادة حساب المؤشرات لنفس السهم
                        _, _, _, df_processed = prepare_ai_data(df_new)
                        
                        current_price = df_processed['Close'].iloc[-1]
                        predicted_price = predict_next_move(model, scaler, df_processed)
                        
                        change_pct = ((predicted_price - current_price) / current_price) * 100
                        
                        st.divider()
                        metric_col1, metric_col2 = st.columns(2)
                        metric_col1.metric("السعر الحالي", f"{current_price:.2f}")
                        metric_col2.metric("توقع AI لليوم التالي", f"{predicted_price:.2f}", f"{change_pct:.2f}%")
                        
                        if change_pct > 0:
                            st.success("🤖 توصية الذكاء: الاتجاه صاعد (بناءً على الصناديق والمتوسطات)")
                        else:
                            st.error("🤖 توصية الذكاء: الاتجاه هابط أو تصحيحي")
                            
                    except Exception as e:
                        st.error(f"حدث خطأ أثناء التنبؤ: {e}")
                else:
                    st.warning("لم يتم العثور على نموذج مدرب. يرجى التدريب أولاً.")

# --- بقية التبويبات (نفس الكود السابق للماسح والشارت) ---
elif selected_tab == "الرئيسية":
    st.info("انتقل لتبويب 'تدريب الذكاء' للبدء.")
elif selected_tab == "الماسح الذكي":
    st.write("الماسح هنا...") # (يمكنك نسخ كود الماسح السابق هنا)
elif selected_tab == "الشارت":
    st.write("الشارت هنا...") # (يمكنك نسخ كود الشارت السابق هنا)

