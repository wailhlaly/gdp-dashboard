import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from streamlit_option_menu import option_menu
from scipy.signal import argrelextrema
import os
import joblib
import time

# محاولة استيراد مكتبات الذكاء (مع حماية)
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

# --- 1. إعداد الصفحة ---
st.set_page_config(page_title="TASI AI Auto-Pilot", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif; }
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    div[data-testid="stMetric"] { background-color: #1d212b; border: 1px solid #333; padding: 10px; border-radius: 8px; }
    [data-testid="stMetricValue"] { color: #ffffff !important; }
    div.stButton > button { background: linear-gradient(90deg, #2962ff, #0039cb); color: white; border: none; padding: 10px; width: 100%; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# --- 2. القائمة العلوية ---
selected_tab = option_menu(
    menu_title=None,
    options=["الرئيسية", "مختبر الذكاء (AI Lab)", "الشارت الفني"],
    icons=["house", "robot", "graph-up"],
    default_index=1,
    orientation="horizontal",
    styles={"container": {"background-color": "transparent"}, "nav-link-selected": {"background-color": "#2962ff"}}
)

# --- 3. إعدادات ومجلدات ---
if not os.path.exists('models'): os.makedirs('models') # مجلد لحفظ ملفات الذكاء

with st.sidebar:
    st.header("⚙️ الإعدادات")
    RSI_PERIOD = st.number_input("RSI Period", 14, 30, 24)
    EMA_PERIOD = st.number_input("EMA Trend", 10, 200, 20)
    ATR_MULT = st.number_input("ATR Mult", 1.0, 3.0, 1.5)
    EPOCHS = st.slider("دورات التدريب (Epochs)", 1, 20, 5)

# --- 4. دوال الذكاء الاصطناعي (AI Engine) ---
def prepare_xy(df, lookback=60):
    # إضافة المؤشرات كـ Features
    df['RSI'] = 100 - (100 / (1 + df['Close'].diff().clip(lower=0).ewm(alpha=1/14).mean() / df['Close'].diff().clip(upper=0).abs().ewm(alpha=1/14).mean()))
    df['EMA'] = df['Close'].ewm(span=20).mean()
    df.dropna(inplace=True)
    
    if len(df) < lookback + 10: return None, None, None, None
    
    # نستخدم السعر و RSI و EMA للتدريب
    dataset = df[['Close', 'RSI', 'EMA']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    x_train, y_train = [], []
    for i in range(lookback, len(scaled_data)):
        x_train.append(scaled_data[i-lookback:i, :]) 
        y_train.append(scaled_data[i, 0]) # نتوقع السعر (العمود 0)
        
    return np.array(x_train), np.array(y_train), scaler, df

def build_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_stock(symbol):
    """دالة تدرب سهم واحد وترجع النتائج"""
    try:
        df = yf.download(symbol, period="5y", interval="1d", progress=False)
        if df.empty: return None
        
        # إصلاح KeyError: التأكد من وجود الأعمدة
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        x_train, y_train, scaler, df_clean = prepare_xy(df)
        
        if x_train is None: return None # بيانات غير كافية
        
        model = build_lstm((x_train.shape[1], x_train.shape[2]))
        
        # التدريب
        history = model.fit(x_train, y_train, batch_size=32, epochs=EPOCHS, verbose=0)
        
        # الحفظ
        safe_sym = symbol.replace(".SR", "")
        model.save(f'models/{safe_sym}_model.keras')
        joblib.dump(scaler, f'models/{safe_sym}_scaler.pkl')
        
        # تقييم سريع (آخر 60 يوم)
        last_x = x_train[-1].reshape(1, x_train.shape[1], x_train.shape[2])
        pred_scaled = model.predict(last_x)
        
        # عكس التحجيم (Trick for 3 features)
        dummy = np.zeros((1, 3))
        dummy[0, 0] = pred_scaled[0,0]
        pred_price = scaler.inverse_transform(dummy)[0, 0]
        
        return {
            "loss": history.history['loss'],
            "last_price": df_clean['Close'].iloc[-1],
            "predicted": pred_price,
            "data_count": len(df_clean)
        }
    except Exception as e:
        print(f"Error training {symbol}: {e}")
        return None

# --- 5. دوال التحليل الفني (الكود السابق المصحح) ---
def process_technical(df):
    # حساب المؤشرات بأمان
    df['Change'] = df['Close'].pct_change() * 100
    
    delta = df['Close'].diff()
    gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/24, min_periods=24, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/24, min_periods=24, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

# --- 6. الواجهة والتشغيل ---

# تهيئة الجلسة
if 'ai_logs' not in st.session_state: st.session_state['ai_logs'] = []
if 'training_active' not in st.session_state: st.session_state['training_active'] = False
if 'market_data' not in st.session_state: st.session_state['market_data'] = []

# === تبويب 1: مختبر الذكاء (AI Lab) ===
if selected_tab == "مختبر الذكاء (AI Lab)":
    st.markdown("### 🧠 الطيار الآلي (Auto-Pilot Training)")
    st.info("سيقوم هذا النظام بالتدريب على الشركات واحدة تلو الأخرى، وحفظ خبرته في ملفات.")
    
    col_btn, col_stat = st.columns([1, 3])
    
    with col_btn:
        if st.button("🔴 بدء التدريب المتسلسل (كل السوق)"):
            st.session_state['training_active'] = True
            st.session_state['ai_logs'] = [] # تصفير السجل
    
    # منطقة العرض الحي
    status_placeholder = st.empty()
    chart_placeholder = st.empty()
    log_placeholder = st.empty()
    
    if st.session_state['training_active']:
        tickers_list = list(TICKERS.keys())
        progress_bar = st.progress(0)
        
        for i, sym in enumerate(tickers_list):
            name = TICKERS[sym]
            status_placeholder.markdown(f"### ⏳ جاري تدريب العقل على: **{name}** ({i+1}/{len(tickers_list)})")
            
            # عملية التدريب
            result = train_stock(sym)
            
            if result:
                # تسجيل النتيجة
                log_entry = {
                    "الشركة": name,
                    "السعر": result['last_price'],
                    "توقع AI": result['predicted'],
                    "الفرق %": ((result['predicted'] - result['last_price']) / result['last_price']) * 100,
                    "الخطأ (Loss)": result['loss'][-1]
                }
                st.session_state['ai_logs'].insert(0, log_entry) # الأحدث في الأعلى
                
                # رسم منحنى التعلم (Loss Curve)
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(y=result['loss'], mode='lines', name='Loss', line=dict(color='#00e676')))
                fig_loss.update_layout(title=f"منحنى تعلم {name} (كلما نزل كان أفضل)", height=300, template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0))
                chart_placeholder.plotly_chart(fig_loss, use_container_width=True)
                
            progress_bar.progress((i + 1) / len(tickers_list))
            
            # تحديث جدول السجل
            if st.session_state['ai_logs']:
                df_log = pd.DataFrame(st.session_state['ai_logs'])
                # تلوين التوقع
                def highlight_pred(val):
                    color = '#00e676' if val > 0 else '#ff5252'
                    return f'color: {color}; font-weight: bold'
                
                log_placeholder.dataframe(
                    df_log.style.format({"السعر": "{:.2f}", "توقع AI": "{:.2f}", "الفرق %": "{:.2f}%", "الخطأ (Loss)": "{:.5f}"})
                    .map(highlight_pred, subset=['الفرق %']),
                    use_container_width=True, height=400
                )
                
        status_placeholder.success("✅ تم الانتهاء من تدريب جميع شركات السوق!")
        st.session_state['training_active'] = False

    # عرض السجل إذا توقف التدريب
    elif st.session_state['ai_logs']:
        st.write("نتائج آخر جلسة تدريب:")
        df_log = pd.DataFrame(st.session_state['ai_logs'])
        st.dataframe(df_log, use_container_width=True)

# === تبويب 2: الرئيسية (التحليل التقليدي) ===
elif selected_tab == "الرئيسية":
    st.markdown("### 📊 لوحة السوق (تحليل فني)")
    
    if st.button("🔄 تحديث البيانات (بدون AI)"):
        st.session_state['market_data'] = []
        tickers = list(TICKERS.keys())
        p_bar = st.progress(0)
        
        # التحميل بنظام الدفعات (Batching) لتفادي الأخطاء
        chunk_size = 50
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i:i + chunk_size]
            try:
                raw = yf.download(chunk, period="1y", interval="1d", group_by='ticker', auto_adjust=False, threads=True, progress=False)
                if not raw.empty:
                    for sym in chunk:
                        try:
                            # إصلاح KeyError هنا: التأكد من وجود البيانات
                            df = raw[sym].copy() if sym in raw.columns.levels[0] else pd.DataFrame()
                            
                            # معالجة الهيكلة MultiIndex
                            if df.empty and sym in raw.columns: df = raw[[sym]] # محاولة أخرى
                            
                            if not df.empty:
                                # توحيد الأعمدة
                                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                                col = 'Close' if 'Close' in df.columns else 'Adj Close'
                                
                                df = df.rename(columns={col: 'Close'})
                                df = df.dropna()
                                
                                if len(df) > 20:
                                    df = process_technical(df)
                                    last = df.iloc[-1]
                                    
                                    st.session_state['market_data'].append({
                                        "الاسم": TICKERS.get(sym, sym),
                                        "السعر": last['Close'],
                                        "التغير %": last['Change'],
                                        "RSI": last['RSI']
                                    })
                        except: continue
            except: pass
            p_bar.progress(min((i + chunk_size) / len(tickers), 1.0))
        
        p_bar.empty()
    
    if st.session_state['market_data']:
        df_m = pd.DataFrame(st.session_state['market_data'])
        
        # الهيت ماب
        if not df_m.empty:
            fig = px.treemap(df_m, path=[px.Constant("السوق"), 'الاسم'], values='السعر', color='التغير %',
                             color_continuous_scale=['#ff5252', '#1e222d', '#00e676'], range_color=[-3, 3])
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(df_m.style.background_gradient(cmap='RdYlGn', subset=['التغير %']), use_container_width=True)

# === تبويب 3: الشارت ===
elif selected_tab == "الشارت الفني":
    st.info("اختر سهماً من القائمة الجانبية (غير مفعل في وضع التدريب)")

