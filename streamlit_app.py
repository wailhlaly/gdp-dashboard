import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from ta import add_all_ta_features
from ta.utils import dropna
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
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

# تخصيص CSS لدعم الاتجاه من اليمين لليسار (RTL) وتحسين الخطوط
st.markdown("""
<style>
    .main { direction: rtl; }
    h1, h2, h3, h4, p, div { font-family: 'Tajawal', sans-serif; text-align: right; }
    .stMetric { text-align: right !important; direction: rtl; }
    /* تعديل محاذاة الجداول */
    .stDataFrame { direction: ltr; } 
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. وظائف جلب البيانات (Caching & Data Fetching)
# ---------------------------------------------------------

@st.cache_data(ttl=3600)  # تخزين مؤقت لمدة ساعة
def get_stock_data(ticker, start_date, end_date):
    """جلب بيانات السهم المحلي"""
    # التأكد من وجود اللاحقة .SR للسوق السعودي
    if not ticker.endswith('.SR'):
        ticker = f"{ticker}.SR"
    
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    # معالجة مشكلة MultiIndex في yfinance الحديث
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    if df.empty:
        return None
    
    # حساب المؤشرات الفنية الأساسية محلياً
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['RSI'] = compute_rsi(df['Close'], 14)
    
    # مؤشر البولنجر
    df['BB_High'] = df['Close'].rolling(20).mean() + (df['Close'].rolling(20).std() * 2)
    df['BB_Low'] = df['Close'].rolling(20).mean() - (df['Close'].rolling(20).std() * 2)
    
    return df

@st.cache_data(ttl=3600)
def get_global_indices(start_date, end_date):
    """جلب المؤشرات العالمية وتوحيد التواريخ"""
    tickers = {
        'S&P 500': '^GSPC',
        'Brent Oil': 'BZ=F',
        'Gold': 'GC=F',
        'USD Index': 'DX-Y.NYB', # بديل DXY
        'US 10Y Bond': '^TNX'
    }
    
    global_df = pd.DataFrame()
    
    for name, sym in tickers.items():
        data = yf.download(sym, start=start_date, end=end_date, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        if not data.empty:
            # نستخدم سعر الإغلاق فقط
            temp = data[['Close']].rename(columns={'Close': name})
            if global_df.empty:
                global_df = temp
            else:
                global_df = global_df.join(temp, how='outer')
    
    # ملء القيم المفقودة (بسبب اختلاف العطلات بين السعودية والعالم)
    global_df.fillna(method='ffill', inplace=True)
    global_df.fillna(method='bfill', inplace=True)
    
    return global_df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# ---------------------------------------------------------
# 3. الدمج والتحليل (Processing)
# ---------------------------------------------------------

def prepare_dataset(local_df, global_df):
    """دمج البيانات المحلية والعالمية وتجهيز الميزات"""
    # دمج البيانات بناء على التاريخ
    combined = local_df.join(global_df, how='inner')
    
    # حساب التغيرات النسبية للمؤشرات العالمية (Feature Engineering)
    global_cols = global_df.columns
    for col in global_cols:
        combined[f'{col}_Pct_Change'] = combined[col].pct_change()
        # إضافة Lag (تأخير زمني) لتمثيل التأثير المتأخر
        combined[f'{col}_Lag1'] = combined[f'{col}_Pct_Change'].shift(1)
        combined[f'{col}_Lag3'] = combined[f'{col}_Pct_Change'].shift(3)

    combined.dropna(inplace=True)
    return combined

# ---------------------------------------------------------
# 4. النمذجة (Machine Learning - XGBoost)
# ---------------------------------------------------------

def train_prediction_model(df, target_col='Close', horizon=30):
    """
    تدريب نموذج لتوقع السعر بعد عدد محدد من الأيام
    """
    data = df.copy()
    
    # الهدف: التنبؤ بالسعر بعد horizon يوم
    data['Target'] = data[target_col].shift(-horizon)
    
    # الميزات (Features)
    feature_cols = [c for c in data.columns if c not in ['Target', 'Open', 'High', 'Low', 'Volume', 'Adj Close']]
    # نستبعد الأعمدة غير الرقمية ونبقي المؤشرات المحسوبة والعالمية
    
    data.dropna(inplace=True)
    
    X = data[feature_cols]
    y = data['Target']
    
    # تقسيم البيانات (آخر 20% للاختبار)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # نموذج XGBoost
    model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # تقييم النموذج
    preds = model.predict(X_test)
    score = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    
    # التنبؤ للمستقبل (بناءً على آخر بيانات متوفرة)
    last_row = X.iloc[[-1]]
    future_pred = model.predict(last_row)[0]
    
    # استخراج أهم الميزات
    importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    return model, score, mae, future_pred, importance, preds, y_test

# ---------------------------------------------------------
# 5. واجهة المستخدم (UI Layout)
# ---------------------------------------------------------

# --- الشريط الجانبي ---
st.sidebar.header("📊 إعدادات التحليل")
ticker_input = st.sidebar.text_input("رمز السهم (مثال: 1120 للراجحي)", value="1120")
years_back = st.sidebar.slider("فترة البيانات (سنوات)", 1, 10, 3)
forecast_days = st.sidebar.selectbox("أفق التوقعات (أيام)", [7, 14, 30, 90], index=2)

st.sidebar.markdown("---")
st.sidebar.subheader("خيارات النموذج")
include_global = st.sidebar.checkbox("تضمين المؤشرات العالمية", value=True)

if st.sidebar.button("تشغيل التحليل 🚀"):
    
    # تحديد التواريخ
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years_back*365)
    
    with st.spinner('جاري جلب بيانات تاسي والمؤشرات العالمية...'):
        # 1. جلب البيانات
        local_df = get_stock_data(ticker_input, start_date, end_date)
        
        if local_df is None:
            st.error("لم يتم العثور على بيانات للسهم. تأكد من الرمز.")
        else:
            global_df = get_global_indices(start_date, end_date)
            
            # تجهيز البيانات المشتركة
            full_df = prepare_dataset(local_df, global_df) if include_global else local_df.dropna()
            
            # ---------------------------------------------------------
            # لوحة القيادة الرئيسية
            # ---------------------------------------------------------
            st.title(f"تحليل سهم: {ticker_input}.SR")
            
            # KPIs
            last_price = local_df['Close'].iloc[-1]
            prev_price = local_df['Close'].iloc[-2]
            change = last_price - prev_price
            pct_change = (change / prev_price) * 100
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("آخر إغلاق", f"{last_price:.2f} SAR", f"{pct_change:.2f}%")
            col2.metric("RSI (14)", f"{local_df['RSI'].iloc[-1]:.1f}", "تشبع" if local_df['RSI'].iloc[-1] > 70 else "عادي")
            col3.metric("SMA 50", f"{local_df['SMA_50'].iloc[-1]:.2f}")
            col4.metric("حجم التداول", f"{local_df['Volume'].iloc[-1]:,.0f}")
            
            # التبويبات
            tab1, tab2, tab3 = st.tabs(["📈 التحليل الفني", "🌍 الارتباط العالمي", "🤖 التوقعات الذكية"])
            
            # --- Tab 1: التحليل الفني ---
            with tab1:
                st.subheader("حركة السعر والمؤشرات الفنية")
                
                fig = go.Figure()
                # الشموع
                fig.add_trace(go.Candlestick(x=local_df.index,
                                open=local_df['Open'], high=local_df['High'],
                                low=local_df['Low'], close=local_df['Close'], name='السعر'))
                # المتوسطات
                fig.add_trace(go.Scatter(x=local_df.index, y=local_df['SMA_50'], line=dict(color='orange', width=1), name='SMA 50'))
                fig.add_trace(go.Scatter(x=local_df.index, y=local_df['BB_High'], line=dict(color='gray', width=1, dash='dot'), name='BB High'))
                fig.add_trace(go.Scatter(x=local_df.index, y=local_df['BB_Low'], line=dict(color='gray', width=1, dash='dot'), name='BB Low'))
                
                fig.update_layout(height=600, title_text="الرسم البياني للسعر مع البولنجر باند والمتوسطات")
                st.plotly_chart(fig, use_container_width=True)
            
            # --- Tab 2: الارتباط العالمي ---
            with tab2:
                if include_global:
                    st.subheader("مدى تأثر السهم بالأسواق العالمية")
                    
                    # حساب مصفوفة الارتباط
                    corr_matrix = full_df[['Close', 'S&P 500', 'Brent Oil', 'Gold', 'US 10Y Bond']].corr()
                    
                    # عرض Heatmap
                    fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', title="مصفوفة الارتباط (Correlation Matrix)")
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                    # استنتاج نصي
                    oil_corr = corr_matrix.loc['Close', 'Brent Oil']
                    sp500_corr = corr_matrix.loc['Close', 'S&P 500']
                    
                    st.markdown(f"""
                    ### 💡 استنتاجات المحلل الآلي:
                    * **علاقة النفط:** معامل الارتباط هو **{oil_corr:.2f}**. {'علاقة طردية قوية، السهم يتحرك مع النفط.' if oil_corr > 0.5 else 'لا يوجد تأثير قوي مباشر لسعر النفط على السهم حالياً.'}
                    * **الأسواق الأمريكية:** معامل الارتباط مع S&P500 هو **{sp500_corr:.2f}**.
                    """)
                else:
                    st.warning("تم تعطيل خيار المؤشرات العالمية.")

            # --- Tab 3: التوقعات الذكية (ML) ---
            with tab3:
                st.subheader(f"نموذج الذكاء الاصطناعي (XGBoost) - توقع {forecast_days} يوم")
                
                model, score, mae, future_pred, importance, preds, y_test_vals = train_prediction_model(full_df, horizon=forecast_days)
                
                c1, c2 = st.columns(2)
                with c1:
                    st.info(f"💵 السعر المتوقع بعد {forecast_days} يوم: **{future_pred:.2f} SAR**")
                    direction = "صعود 🟢" if future_pred > last_price else "هبوط 🔴"
                    st.metric("الاتجاه المتوقع", direction, f"{((future_pred - last_price)/last_price)*100:.2f}%")
                
                with c2:
                    st.text("دقة النموذج (R² Score):")
                    st.progress(max(0.0, min(1.0, score)))  # Clipping between 0 and 1
                    st.caption(f"هامش الخطأ المتوسط (MAE): {mae:.2f} ريال")

                st.markdown("---")
                
                # عرض العوامل المؤثرة
                st.write("#### 🔍 ما الذي يؤثر في هذا التوقع؟")
                top_features = importance.head(5)
                fig_imp = px.bar(top_features, x='Importance', y='Feature', orientation='h', title="أهم العوامل المؤثرة في القرار")
                st.plotly_chart(fig_imp, use_container_width=True)
                
                # رسم التوقع التاريخي (Test Set vs Predictions)
                st.write("#### أداء النموذج على البيانات التاريخية (آخر فترة)")
                comparison_df = pd.DataFrame({'Actual': y_test_vals, 'Predicted': preds}, index=y_test_vals.index)
                fig_pred = px.line(comparison_df, title="مقارنة السعر الحقيقي مقابل توقعات النموذج")
                st.plotly_chart(fig_pred, use_container_width=True)
                
                st.warning("⚠️ إخلاء مسؤولية: هذه التوقعات مبنية على نماذج رياضية واحتمالات إحصائية ولا تشكل نصيحة استثمارية ملزمة.")

else:
    st.info("👈 قم بإدخال رمز السهم في القائمة الجانبية واضغط 'تشغيل التحليل' للبدء.")

