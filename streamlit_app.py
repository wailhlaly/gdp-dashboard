import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import os
import joblib

# --- مكتبات الذكاء ---
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
    st.error("🚨 ملف البيانات مفقود.")
    st.stop()

TICKERS = {item['symbol']: item['name'] for item in STOCKS_DB}

# --- إعداد المجلدات ---
if not os.path.exists('ai_mind'): os.makedirs('ai_mind')

# --- 1. إعداد الصفحة ---
st.set_page_config(page_title="TASI AI Tuner", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif; }
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    div.stButton > button {
        background: linear-gradient(90deg, #d500f9, #651fff); color: white; border: none;
        padding: 12px; width: 100%; border-radius: 8px; font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. القائمة العلوية ---
selected_tab = option_menu(
    menu_title=None,
    options=["الرئيسية", "🧠 ضبط العقل (Bias/Variance)", "الشارت الفني"],
    icons=["house", "sliders", "graph-up"],
    default_index=1,
    orientation="horizontal",
    styles={"container": {"background-color": "transparent"}, "nav-link-selected": {"background-color": "#651fff"}}
)

# --- 3. إعدادات متقدمة (Hyperparameters) ---
with st.sidebar:
    st.header("🎛️ ضبط Bias/Variance")
    
    st.info("💡 **كيف تضبط النموذج؟**\n\n- لتقليل **Bias** (النموذج لا يتعلم): زد عدد الوحدات (Units) والـ Epochs.\n\n- لتقليل **Variance** (النموذج يحفظ فقط): زد نسبة الـ Dropout.")
    
    # تحكم في تعقيد النموذج
    LSTM_UNITS = st.slider("عدد الخلايا العصبية (Complexity)", 20, 200, 50)
    DROPOUT_RATE = st.slider("نسبة النسيان (Dropout)", 0.1, 0.5, 0.2, step=0.05)
    EPOCHS = st.slider("دورات التدريب (Epochs)", 5, 100, 20)
    
    st.divider()
    RSI_PERIOD = st.number_input("RSI Period", 14)
    EMA_PERIOD = st.number_input("EMA Period", 20)

# --- 4. تجهيز البيانات ---
def calculate_atr(df):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    return ranges.max(axis=1).ewm(alpha=1/14, min_periods=14, adjust=False).mean()

def prepare_data_for_ai(df):
    df = df.copy()
    df['RSI'] = 100 - (100 / (1 + df['Close'].diff().clip(lower=0).ewm(alpha=1/14).mean() / df['Close'].diff().clip(upper=0).abs().ewm(alpha=1/14).mean()))
    df['EMA'] = df['Close'].ewm(span=20).mean()
    df['Box_High'] = df['High'].rolling(window=20).max()
    df['Box_Low'] = df['Low'].rolling(window=20).min()
    df.dropna(inplace=True)
    return df

# --- 5. بناء النموذج (Flexible Model) ---
def build_brain_model(input_shape):
    model = Sequential()
    # زيادة الوحدات تقلل Bias، زيادة Dropout تقلل Variance
    model.add(LSTM(units=LSTM_UNITS, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(DROPOUT_RATE)) 
    
    model.add(LSTM(units=LSTM_UNITS, return_sequences=False))
    model.add(Dropout(DROPOUT_RATE))
    
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_mind_with_validation(symbol):
    status = st.empty()
    status.info(f"جاري التدريب والتحقق من الـ Bias/Variance لسهم {symbol}...")
    
    try:
        df = yf.download(symbol, period="5y", interval="1d", progress=False)
        if len(df) < 500: return None, None, None

        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df_processed = prepare_data_for_ai(df)
        
        features = ['Close', 'RSI', 'EMA', 'Box_High', 'Box_Low']
        data_values = df_processed[features].values
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data_values)
        
        X, y = [], []
        lookback = 60
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, :])
            y.append(scaled_data[i, 0])
            
        X, y = np.array(X), np.array(y)
        
        # بناء النموذج بالإعدادات المختارة
        model = build_brain_model((X.shape[1], X.shape[2]))
        
        # التقسيم للتحقق (Validation Split) لكشف الـ Variance
        # validation_split=0.2 يعني أننا نخفي 20% من البيانات عن النموذج لنختبره بها
        history = model.fit(X, y, batch_size=32, epochs=EPOCHS, validation_split=0.2, verbose=0)
        
        # الحفظ
        safe_sym = symbol.replace(".SR", "")
        model.save(f'ai_mind/{safe_sym}_model.keras')
        joblib.dump(scaler, f'ai_mind/{safe_sym}_scaler.pkl')
        
        status.success("✅ تم التدريب! راجع الرسم البياني للأسفل.")
        return history, df_processed, scaler
        
    except Exception as e:
        st.error(f"خطأ: {e}")
        return None, None, None

# --- 6. الواجهة ---

if selected_tab == "🧠 ضبط العقل (Bias/Variance)":
    st.header("🎛️ مختبر ضبط أداء الذكاء الاصطناعي")
    
    if not AI_AVAILABLE:
        st.error("مكتبات AI مفقودة.")
    else:
        c1, c2 = st.columns([3, 1])
        with c1:
            target_stock = st.selectbox("اختر السهم للاختبار", list(TICKERS.keys()))
        with c2:
            st.write("")
            st.write("")
            start_train = st.button("🧪 بدء الاختبار")
            
        if start_train:
            history, df_res, scaler = train_mind_with_validation(target_stock)
            
            if history:
                # --- رسم منحنى التعلم (Learning Curve) ---
                # 
                loss_train = history.history['loss']
                loss_val = history.history['val_loss']
                epochs_range = range(1, len(loss_train) + 1)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=list(epochs_range), y=loss_train, mode='lines', name='Training Loss (خطأ التدريب)', line=dict(color='#00e676')))
                fig.add_trace(go.Scatter(x=list(epochs_range), y=loss_val, mode='lines', name='Validation Loss (خطأ التحقق)', line=dict(color='#ff2950', dash='dot')))
                
                fig.update_layout(
                    title="منحنى التعلم (Learning Curve) - كاشف التحيز والتباين",
                    xaxis_title="الدورات (Epochs)",
                    yaxis_title="متوسط الخطأ (Loss)",
                    template="plotly_dark",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # --- تحليل النتائج آلياً ---
                final_train_loss = loss_train[-1]
                final_val_loss = loss_val[-1]
                gap = final_val_loss - final_train_loss
                
                c_res1, c_res2, c_res3 = st.columns(3)
                c_res1.metric("خطأ التدريب", f"{final_train_loss:.5f}")
                c_res2.metric("خطأ الاختبار (الواقع)", f"{final_val_loss:.5f}")
                
                # التشخيص الآلي
                if final_train_loss > 0.01:
                    status_msg = "🔴 High Bias (Underfitting)"
                    advice = "النموذج 'غبي' قليلاً. الحل: زد عدد الخلايا العصبية (LSTM Units) أو زد الـ Epochs."
                elif gap > 0.005: # فرق كبير بين التدريب والاختبار
                    status_msg = "🟠 High Variance (Overfitting)"
                    advice = "النموذج 'يحفظ' البيانات. الحل: زد نسبة الـ Dropout أو قلل تعقيد الشبكة."
                else:
                    status_msg = "🟢 Balanced Model (ممتاز)"
                    advice = "النموذج متوازن وجاهز للاستخدام!"
                
                c_res3.metric("الحالة", status_msg)
                st.info(f"💡 **التشخيص:** {advice}")

elif selected_tab == "الرئيسية":
    st.info("انتقل لتبويب 'ضبط العقل' للتحكم في دقة النموذج.")
