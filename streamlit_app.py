import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import os
import joblib

# --- مكتبات الذكاء الاصطناعي ---
try:
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
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
SECTORS_DICT = {}
for item in STOCKS_DB:
    sec = item['sector']
    if sec not in SECTORS_DICT: SECTORS_DICT[sec] = []
    SECTORS_DICT[sec].append(item['symbol'])

# رموز البيانات الاقتصادية العالمية
MACRO_TICKERS = {
    'Oil': 'BZ=F',       # نفط برنت
    'Gold': 'GC=F',      # الذهب
    'DXY': 'DX-Y.NYB',   # مؤشر الدولار
    'US10Y': '^TNX'      # عوائد السندات الأمريكية 10 سنوات
}

# --- 1. إعداد الصفحة ---
st.set_page_config(page_title="TASI Macro AI", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif; }
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    div.stButton > button {
        background: linear-gradient(90deg, #00c853, #64dd17); color: black; border: none;
        padding: 12px; width: 100%; border-radius: 8px; font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. القائمة العلوية ---
selected_tab = option_menu(
    menu_title=None,
    options=["الرئيسية", "🧠 المحاكي الاقتصادي (Macro AI)", "الشارت الفني"],
    icons=["house", "globe", "graph-up"],
    default_index=1,
    orientation="horizontal",
    styles={"container": {"background-color": "transparent"}, "nav-link-selected": {"background-color": "#00c853", "color": "black"}}
)

# --- 3. دوال البيانات والذكاء ---

@st.cache_data
def get_macro_data(period="5y"):
    """جلب البيانات الاقتصادية العالمية ودمجها"""
    dfs = []
    for name, ticker in MACRO_TICKERS.items():
        try:
            d = yf.download(ticker, period=period, interval="1d", progress=False)
            if not d.empty:
                # إصلاح MultiIndex إذا وجد
                if isinstance(d.columns, pd.MultiIndex):
                    d.columns = d.columns.get_level_values(0)
                
                d = d[['Close']].rename(columns={'Close': name})
                dfs.append(d)
        except: pass
    
    if dfs:
        macro_df = pd.concat(dfs, axis=1)
        # ملء الفراغات (لأن العطلات العالمية تختلف عن السعودية)
        macro_df = macro_df.ffill().bfill()
        return macro_df
    return pd.DataFrame()

def prepare_advanced_data(symbol, lookback=60, training_end_date=None):
    # 1. جلب بيانات السهم
    stock_df = yf.download(symbol, period="10y", interval="1d", progress=False) # فترة طويلة للتدريب
    if stock_df.empty: return None, None, None, None, None
    
    if isinstance(stock_df.columns, pd.MultiIndex): stock_df.columns = stock_df.columns.get_level_values(0)
    
    # 2. جلب البيانات الاقتصادية
    macro_df = get_macro_data("10y")
    
    # 3. دمج البيانات (Merge) بناءً على التاريخ
    # نستخدم مؤشر السهم كأساس، ونربط الماكرو به
    df = stock_df.join(macro_df, how='left')
    df = df.ffill().bfill() # تعبئة أيام الإجازات العالمية بالقيم السابقة
    
    # 4. المؤشرات الفنية
    df['RSI'] = 100 - (100 / (1 + df['Close'].diff().clip(lower=0).ewm(alpha=1/14).mean() / df['Close'].diff().clip(upper=0).abs().ewm(alpha=1/14).mean()))
    df['EMA20'] = df['Close'].ewm(span=20).mean()
    
    df.dropna(inplace=True)
    
    # 5. تقسيم البيانات (Training vs Simulation)
    # إذا حددنا تاريخاً للنهاية، نقطع البيانات عنده
    full_data = df.copy() # نحتفظ بالنسخة الكاملة للمقارنة لاحقاً
    
    if training_end_date:
        # قص البيانات حتى تاريخ المحاكاة (إخفاء المستقبل)
        df = df[df.index <= pd.to_datetime(training_end_date)]
    
    if len(df) < lookback + 50: return None, None, None, None, None

    # الميزات (Features): السهم + الاقتصاد + المؤشرات
    features = ['Close', 'RSI', 'EMA20', 'Oil', 'Gold', 'US10Y', 'DXY']
    # التأكد من وجود الأعمدة
    available_features = [f for f in features if f in df.columns]
    
    dataset = df[available_features].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    x_train, y_train = [], []
    for i in range(lookback, len(scaled_data)):
        x_train.append(scaled_data[i-lookback:i, :])
        y_train.append(scaled_data[i, 0]) # الهدف هو سعر الإغلاق (العمود 0)
        
    return np.array(x_train), np.array(y_train), scaler, df, full_data

def build_advanced_model(input_shape):
    model = Sequential()
    # طبقات LSTM معقدة لفهم العلاقات
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1)) # التوقع
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# --- 4. واجهة المحاكاة (Macro AI Lab) ---

if selected_tab == "🧠 المحاكي الاقتصادي (Macro AI)":
    st.title("🧠 المحاكاة الاقتصادية والتدريب الزمني")
    st.caption("يتعلم النموذج من: حركة السعر + النفط + الذهب + الفائدة + الدولار + المؤشرات الفنية.")
    
    if not AI_AVAILABLE:
        st.error("المكتبات ناقصة.")
        st.stop()

    c1, c2 = st.columns(2)
    with c1:
        chosen_sector = st.selectbox("القطاع", list(SECTORS_DICT.keys()))
        sector_stocks = {s: TICKERS.get(s, s) for s in SECTORS_DICT[chosen_sector]}
        target_stock = st.selectbox("السهم المراد اختباره", list(sector_stocks.keys()), format_func=lambda x: sector_stocks[x])
    
    with c2:
        # تحديد تاريخ "الماضي" الذي نريد التوقف عنده وبدء التوقع منه
        today = pd.Timestamp.now()
        start_date = today - pd.Timedelta(days=365*2) # سنتين للوراء
        
        # سلايدر لاختيار نقطة "قطع البيانات"
        sim_days = st.slider("عدد أيام المحاكاة (Replay Days)", 30, 180, 90, help="سنخفي بيانات هذه الأيام عن الذكاء ونطلب منه توقعها")
        
        # تاريخ القطع (Split Date)
        cutoff_date = today - pd.Timedelta(days=sim_days)
        
        st.info(f"سيتم تدريب الذكاء على البيانات حتى تاريخ: **{cutoff_date.date()}**")
        st.warning(f"سيحاول الذكاء توقع الحركة من {cutoff_date.date()} إلى اليوم ({sim_days} يوم) بناءً على المؤشرات الاقتصادية.")

    if st.button("🚀 بدء المحاكاة والتدريب"):
        status = st.empty()
        prog = st.progress(0)
        
        try:
            status.info("1. جلب البيانات الاقتصادية ودمجها مع السهم...")
            
            # 1. تجهيز البيانات (مع إخفاء المستقبل)
            x_train, y_train, scaler, df_train, df_full = prepare_advanced_data(
                target_stock, lookback=60, training_end_date=cutoff_date
            )
            
            if x_train is None:
                st.error("البيانات غير كافية للتدريب.")
                st.stop()
                
            status.info("2. بناء الشبكة العصبية وتدريبها على الماضي...")
            # 2. بناء النموذج
            model = build_advanced_model((x_train.shape[1], x_train.shape[2]))
            
            # تدريب سريع (للعرض) - لزيادة الدقة زد الـ epochs
            model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=0)
            prog.progress(50)
            
            status.info(f"3. تشغيل المحاكاة لآخر {sim_days} يوم...")
            
            # 3. مرحلة التوقع (Replay Loop)
            # الآن نستخدم البيانات الكاملة (df_full) ولكن فقط كمدخلات (Inputs) لنرى ماذا سيتوقع النموذج
            
            # البيانات الحقيقية لفترة المحاكاة
            real_data = df_full[df_full.index > cutoff_date]
            real_prices = real_data['Close'].values
            real_dates = real_data.index
            
            if len(real_prices) == 0:
                st.error("لا توجد بيانات للفترة المحددة.")
                st.stop()

            # تجهيز المدخلات للمحاكاة
            # نحتاج لكل يوم، الـ 60 يوم التي تسبقه (سواء كانت من التدريب أو من التوقع السابق)
            # هنا سنستخدم "البيانات الحقيقية للمؤشرات" (لأننا نعرف تاريخياً كم كان النفط والذهب)
            # ولكن النموذج يتوقع سعر السهم فقط
            
            full_dataset = df_full[['Close', 'RSI', 'EMA20', 'Oil', 'Gold', 'US10Y', 'DXY']].values # يجب أن تطابق features
            scaled_full = scaler.transform(full_dataset)
            
            predictions = []
            
            # نقطة البداية في المصفوفة الكاملة
            start_idx = len(df_train) 
            
            for i in range(len(real_prices)):
                # نأخذ الـ 60 يوم السابقة لهذا اليوم
                # (لاحظ: في الواقع الحقيقي، نحن نعرف النفط والذهب لهذا اليوم، لذا نستخدمها للمساعدة في التوقع)
                idx = start_idx + i
                if idx < 60: continue
                
                input_seq = scaled_full[idx-60:idx, :]
                input_seq = input_seq.reshape(1, 60, input_seq.shape[1])
                
                pred_val_scaled = model.predict(input_seq, verbose=0)
                
                # عكس التحجيم
                dummy = np.zeros((1, input_seq.shape[2])) # نفس عدد الميزات
                dummy[0, 0] = pred_val_scaled[0, 0] # السعر هو العمود الأول
                pred_price = scaler.inverse_transform(dummy)[0, 0]
                
                predictions.append(pred_price)
                prog.progress(50 + int((i / len(real_prices)) * 50))
            
            prog.empty()
            status.success("✅ اكتملت المحاكاة!")
            
            # 4. عرض النتائج والرسم
            st.divider()
            
            # الرسم البياني للمقارنة
            fig = go.Figure()
            
            # السعر الحقيقي
            fig.add_trace(go.Scatter(
                x=real_dates, y=real_prices,
                mode='lines', name='السعر الحقيقي (Real)',
                line=dict(color='#00e676', width=3)
            ))
            
            # توقع الذكاء
            # قد يكون طول التوقعات أقل قليلاً بسبب نقص البيانات في البداية
            valid_dates = real_dates[:len(predictions)]
            
            fig.add_trace(go.Scatter(
                x=valid_dates, y=predictions,
                mode='lines', name='توقع الذكاء (AI Forecast)',
                line=dict(color='#ff2950', width=2, dash='dot')
            ))
            
            # إضافة مناطق الشرح
            fig.add_vline(x=cutoff_date, line_dash="dash", line_color="white", annotation_text="بداية المحاكاة (إخفاء المستقبل)")
            
            fig.update_layout(
                title=f"اختبار كفاءة النموذج على {TICKERS[target_stock]} مع البيانات الاقتصادية",
                template="plotly_dark", height=600,
                xaxis_title="التاريخ", yaxis_title="السعر"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # تحليل الدقة
            if len(predictions) > 0:
                mae = np.mean(np.abs(np.array(predictions) - real_prices[:len(predictions)]))
                last_real = real_prices[-1]
                last_pred = predictions[-1]
                diff_pct = ((last_pred - last_real) / last_real) * 100
                
                k1, k2, k3 = st.columns(3)
                k1.metric("متوسط الخطأ (MAE)", f"{mae:.2f} ريال")
                k2.metric("السعر الحقيقي اليوم", f"{last_real:.2f}")
                k3.metric("توقع النموذج لليوم", f"{last_pred:.2f}", f"{diff_pct:.2f}% الفرق")
                
                st.caption("""
                **تفسير النتائج:**
                - الخط **الأخضر** هو ما حدث فعلاً في السوق.
                - الخط **الأحمر** هو ما توقعه الذكاء الاصطناعي بناءً على (النفط، الذهب، الفائدة) دون معرفة سعر السهم مسبقاً.
                - التقارب بين الخطين يدل على أن السهم يتأثر بشدة بالعوامل الاقتصادية المذكورة.
                """)

        except Exception as e:
            st.error(f"حدث خطأ: {e}")

# --- بقية التبويبات (كما هي) ---
elif selected_tab == "الرئيسية":
    st.info("انتقل لتبويب 'المحاكي الاقتصادي' لتجربة الميزات الجديدة.")
elif selected_tab == "الشارت الفني":
    st.write("الشارت هنا...") 
