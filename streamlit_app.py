import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime, timedelta

# ---------------------------------------------------------
# 1. إعدادات الصفحة والتهيئة
# ---------------------------------------------------------
st.set_page_config(
    page_title="محلل تاسي الذكي (TASI AI Analyzer)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تخصيص CSS لدعم الاتجاه من اليمين لليسار (RTL)
st.markdown("""
<style>
    .main { direction: rtl; }
    h1, h2, h3, h4, p, div { font-family: 'Tajawal', sans-serif; text-align: right; }
    .stMetric { text-align: right !important; direction: rtl; }
    .stDataFrame { direction: ltr; } 
    div[data-testid="stSidebar"] { text-align: right; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. وظائف مساعدة (Helpers)
# ---------------------------------------------------------
def compute_rsi(series, period=14):
    """حساب مؤشر القوة النسبية يدوياً لضمان الاستقرار"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# ---------------------------------------------------------
# 3. جلب البيانات وتنظيفها (Data Fetching & Cleaning)
# ---------------------------------------------------------

@st.cache_data(ttl=3600)
def get_stock_data(ticker, start_date, end_date):
    """جلب بيانات السهم المحلي مع تنظيف صارم للأعمدة"""
    if not ticker.endswith('.SR'):
        ticker = f"{ticker}.SR"
    
    # جلب البيانات بدون تعديلات تلقائية لتجنب المشاكل
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
    except Exception as e:
        st.error(f"خطأ في الاتصال بالمصدر: {e}")
        return None

    # --- إصلاح مشكلة MultiIndex وتكرار الأعمدة ---
    if isinstance(df.columns, pd.MultiIndex):
        # محاولة تسطيح الأعمدة
        try:
            # إذا كان العمود Ticker موجوداً في المستوى الثاني، نحذفه
            df.columns = df.columns.get_level_values(0)
        except:
            pass
    
    # إزالة أي تكرار في أسماء الأعمدة (الحل الجذري للخطأ)
    df = df.loc[:, ~df.columns.duplicated()]

    # التأكد من وجود البيانات الأساسية
    if df.empty or 'Close' not in df.columns:
        return None
    
    # تحويل البيانات إلى أرقام للتأكد
    cols_to_numeric = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # حساب المؤشرات الفنية
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['RSI'] = compute_rsi(df['Close'], 14)
    
    # Bollinger Bands
    df['BB_High'] = df['Close'].rolling(20).mean() + (df['Close'].rolling(20).std() * 2)
    df['BB_Low'] = df['Close'].rolling(20).mean() - (df['Close'].rolling(20).std() * 2)
    
    return df

@st.cache_data(ttl=3600)
def get_global_indices(start_date, end_date):
    """جلب المؤشرات العالمية"""
    tickers = {
        'S&P 500': '^GSPC',
        'Brent Oil': 'BZ=F',
        'Gold': 'GC=F',
        'USD Index': 'DX-Y.NYB',
        'US 10Y Bond': '^TNX'
    }
    
    global_df = pd.DataFrame()
    
    for name, sym in tickers.items():
        try:
            data = yf.download(sym, start=start_date, end=end_date, progress=False, auto_adjust=False)
            
            # تنظيف MultiIndex
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            data = data.loc[:, ~data.columns.duplicated()]

            if not data.empty and 'Close' in data.columns:
                temp = data[['Close']].rename(columns={'Close': name})
                if global_df.empty:
                    global_df = temp
                else:
                    global_df = global_df.join(temp, how='outer')
        except Exception:
            continue
            
    # تعبئة القيم المفقودة (بسبب العطلات)
    global_df.fillna(method='ffill', inplace=True)
    global_df.fillna(method='bfill', inplace=True)
    
    return global_df

def prepare_dataset(local_df, global_df):
    """دمج البيانات وإعداد الميزات"""
    # دمج البيانات
    combined = local_df.join(global_df, how='inner')
    
    # تنظيف فوري للتكرار بعد الدمج
    combined = combined.loc[:, ~combined.columns.duplicated()]
    
    # هندسة الميزات (Feature Engineering)
    global_cols = global_df.columns
    for col in global_cols:
        if col in combined.columns:
            combined[f'{col}_Pct'] = combined[col].pct_change()
            combined[f'{col}_Lag1'] = combined[f'{col}_Pct'].shift(1)
            combined[f'{col}_Lag3'] = combined[f'{col}_Pct'].shift(3)

    combined.dropna(inplace=True)
    return combined

# ---------------------------------------------------------
# 4. النمذجة (Machine Learning)
# ---------------------------------------------------------

def train_prediction_model(df, target_col='Close', horizon=30):
    """تدريب نموذج XGBoost"""
    # 1. تنظيف وفحص البيانات قبل البدء
    data = df.copy()
    data = data.loc[:, ~data.columns.duplicated()] # خطوة أمان إضافية
    
    if target_col not in data.columns:
        return None, 0, 0, 0, pd.DataFrame(), [], []

    # 2. إنشاء الهدف (Target)
    try:
        horizon = int(horizon)
        data['Target'] = data[target_col].shift(-horizon)
    except Exception as e:
        st.error(f"خطأ في إعداد الهدف: {e}")
        return None, 0, 0, 0, pd.DataFrame(), [], []

    # تحديد الميزات (Features) - استبعاد الأعمدة غير المفيدة للتنبؤ
    drop_cols = ['Target', 'Open', 'High', 'Low', 'Volume', 'Adj Close']
    feature_cols = [c for c in data.columns if c not in drop_cols]
    
    data.dropna(inplace=True)
    
    if len(data) < 50: # لا يكفي للتدريب
        st.warning("البيانات غير كافية لتدريب النموذج.")
        return None, 0, 0, 0, pd.DataFrame(), [], []

    X = data[feature_cols]
    y = data['Target']
    
    # تقسيم البيانات
    split_idx = int(len(X) * 0.85)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # النموذج
    model = XGBRegressor(
        n_estimators=150, 
        learning_rate=0.05, 
        max_depth=4, 
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # التقييم
    preds = model.predict(X_test)
    score = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    
    # التنبؤ المستقبلي (آخر صف في البيانات الأصلية)
    last_row_features = X.iloc[[-1]]
    future_pred = model.predict(last_row_features)[0]
    
    # أهم الميزات
    importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    return model, score, mae, future_pred, importance, preds, y_test

# ---------------------------------------------------------
# 5. واجهة المستخدم (Main Layout)
# ---------------------------------------------------------

st.sidebar.header("📊 إعدادات التحليل")
ticker_input = st.sidebar.text_input("رمز السهم", value="1120", help="أدخل الرمز بدون .SR")
years_back = st.sidebar.slider("البيانات التاريخية (سنوات)", 1, 10, 3)
forecast_days = st.sidebar.selectbox("فترة التوقع (أيام)", [7, 14, 30, 90], index=2)
include_global = st.sidebar.checkbox("تضمين المؤشرات العالمية", value=True)

if st.sidebar.button("تشغيل التحليل 🚀"):
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years_back*365)
    
    with st.spinner('جاري الاتصال بالسوق السعودي وتحليل البيانات...'):
        # 1. جلب البيانات المحلية
        local_df = get_stock_data(ticker_input, start_date, end_date)
        
        if local_df is None:
            st.error(f"لم يتم العثور على بيانات للرمز {ticker_input}. تأكد من صحة الرمز.")
        else:
            # 2. جلب البيانات العالمية ودمجها
            full_df = local_df.copy()
            if include_global:
                global_df = get_global_indices(start_date, end_date)
                full_df = prepare_dataset(local_df, global_df)
            else:
                full_df = local_df.dropna()

            # ---------------------------
            # عرض النتائج
            # ---------------------------
            st.title(f"تحليل سهم: {ticker_input} (TASI)")
            
            # KPIs
            last_close = local_df['Close'].iloc[-1]
            prev_close = local_df['Close'].iloc[-2]
            chg = last_close - prev_close
            chg_pct = (chg / prev_close) * 100
            
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("آخر سعر", f"{last_close:.2f}", f"{chg_pct:.2f}%")
            kpi2.metric("SMA 200", f"{local_df['SMA_200'].iloc[-1]:.2f}")
            kpi3.metric("RSI (14)", f"{local_df['RSI'].iloc[-1]:.1f}")
            vol = local_df['Volume'].iloc[-1]
            kpi4.metric("الحجم", f"{vol:,.0f}")
            
            # Tabs
            tab_tech, tab_global, tab_ai = st.tabs(["📈 الرسم الفني", "🌍 الارتباطات", "🤖 توقعات AI"])
            
            # Tab 1: Technical
            with tab_tech:
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=local_df.index,
                                open=local_df['Open'], high=local_df['High'],
                                low=local_df['Low'], close=local_df['Close'], name='السعر'))
                fig.add_trace(go.Scatter(x=local_df.index, y=local_df['BB_High'], line=dict(color='gray', width=1, dash='dot'), name='Bollinger High'))
                fig.add_trace(go.Scatter(x=local_df.index, y=local_df['BB_Low'], line=dict(color='gray', width=1, dash='dot'), name='Bollinger Low'))
                fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, use_container_width=True)
            
            # Tab 2: Global Correlations
            with tab_global:
                if include_global and 'S&P 500' in full_df.columns:
                    corr_cols = ['Close', 'S&P 500', 'Brent Oil', 'Gold', 'US 10Y Bond']
                    # التأكد من وجود الأعمدة قبل حساب الارتباط
                    avail_cols = [c for c in corr_cols if c in full_df.columns]
                    
                    corr_matrix = full_df[avail_cols].corr()
                    
                    fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', title="مصفوفة الارتباط")
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                    st.caption("القيم القريبة من 1 تعني علاقة طردية قوية (يتحركون معاً)، والقريبة من -1 تعني علاقة عكسية.")
                else:
                    st.info("لم يتم تفعيل البيانات العالمية أو فشل تحميلها.")

            # Tab 3: AI Prediction
            with tab_ai:
                st.subheader(f"توقعات الذكاء الاصطناعي لفترة {forecast_days} يوم")
                
                model, score, mae, future_pred, importance, preds, y_test_vals = train_prediction_model(full_df, horizon=forecast_days)
                
                if model:
                    col_res1, col_res2 = st.columns(2)
                    with col_res1:
                        st.success(f"السعر المتوقع: **{future_pred:.2f} ريال**")
                        diff_pred = future_pred - last_close
                        st.metric("التغير المتوقع", f"{diff_pred:.2f}", f"{(diff_pred/last_close)*100:.2f}%")
                    
                    with col_res2:
                        st.write("دقة النموذج:")
                        st.progress(max(0.0, min(1.0, score)))
                        st.caption(f"R² Score: {score:.2f} | الخطأ المتوسط: {mae:.2f}")

                    # الرسم البياني للتوقع
                    st.markdown("---")
                    st.write("##### أهم العوامل المؤثرة:")
                    st.plotly_chart(px.bar(importance.head(7), x='Importance', y='Feature', orientation='h'), use_container_width=True)
                else:
                    st.error("لم يتمكن النموذج من التدريب (قد تكون البيانات غير كافية).")

else:
    st.info("ابدأ بإدخال رمز السهم في القائمة الجانبية (مثل 1120 للراجحي أو 2222 لأرامكو).")
