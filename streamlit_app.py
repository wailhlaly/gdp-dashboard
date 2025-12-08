import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime, timedelta
import os

# ---------------------------------------------------------
# 1. إعدادات الصفحة والتهيئة
# ---------------------------------------------------------
st.set_page_config(
    page_title="محلل تاسي الذكي (TASI AI Analyzer)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { direction: rtl; }
    h1, h2, h3, h4, p, div { font-family: 'Tajawal', sans-serif; text-align: right; }
    .stMetric { text-align: right !important; direction: rtl; }
    .stDataFrame { direction: ltr; } 
    div[data-testid="stSidebar"] { text-align: right; }
    button[data-baseweb="tab"] { font-family: 'Tajawal', sans-serif; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. وظائف مساعدة (Helpers)
# ---------------------------------------------------------

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def analyze_seasonality(df):
    data = df.copy()
    data['Return'] = data['Close'].pct_change() * 100
    data['Month'] = data.index.month
    data['Day'] = data.index.day_name()
    data['Year'] = data.index.year
    
    monthly_seasonality = data.groupby('Month')['Return'].mean()
    days_order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday']
    daily_seasonality = data.groupby('Day')['Return'].mean().reindex(days_order)
    monthly_heatmap = data.groupby(['Year', 'Month'])['Return'].sum().unstack()
    
    return monthly_seasonality, daily_seasonality, monthly_heatmap

# ---------------------------------------------------------
# 3. تحميل قائمة الأسهم من الملف المحلي (الجديد)
# ---------------------------------------------------------
@st.cache_data
def load_tickers_from_file():
    """تحميل ملف الرموز من مجلد data"""
    # المسارات المحتملة للملف
    file_path_csv = os.path.join("data", "saudi_tickers.csv")
    file_path_xlsx = os.path.join("data", "saudi_tickers.xlsx")
    
    df = None
    if os.path.exists(file_path_csv):
        try:
            df = pd.read_csv(file_path_csv)
        except:
            pass
    elif os.path.exists(file_path_xlsx):
        try:
            df = pd.read_excel(file_path_xlsx)
        except:
            pass
            
    if df is not None:
        # تنظيف البيانات: نفترض وجود عمود للرمز وعمود للاسم
        # سنحاول تخمين أسماء الأعمدة إذا لم تكن قياسية
        cols = df.columns.astype(str).str.lower()
        
        symbol_col = next((c for c in df.columns if 'symbol' in str(c).lower() or 'code' in str(c).lower() or 'رمز' in str(c)), None)
        name_col = next((c for c in df.columns if 'name' in str(c).lower() or 'company' in str(c).lower() or 'اسم' in str(c)), None)
        
        # إذا لم نجد أعمدة بالاسم، نأخذ العمود الأول كرمز والثاني كاسم
        if not symbol_col:
            symbol_col = df.columns[0]
        if not name_col and len(df.columns) > 1:
            name_col = df.columns[1]
            
        # إنشاء قاموس للعرض: "الاسم (الرمز)" -> "الرمز"
        ticker_map = {}
        for index, row in df.iterrows():
            sym = str(row[symbol_col]).replace('.SR', '').strip()
            name = str(row[name_col]).strip() if name_col else ""
            display_label = f"{sym} - {name}"
            ticker_map[display_label] = sym
            
        return ticker_map
    return None

# ---------------------------------------------------------
# 4. جلب البيانات وتحليلها
# ---------------------------------------------------------

@st.cache_data(ttl=3600)
def get_stock_data(ticker, start_date, end_date):
    if not ticker.endswith('.SR'):
        ticker = f"{ticker}.SR"
    
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
    except Exception as e:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.get_level_values(0)
        except:
            pass
    
    df = df.loc[:, ~df.columns.duplicated()]

    if df.empty or 'Close' not in df.columns:
        return None
    
    cols_to_numeric = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['RSI'] = compute_rsi(df['Close'], 14)
    df['BB_High'] = df['Close'].rolling(20).mean() + (df['Close'].rolling(20).std() * 2)
    df['BB_Low'] = df['Close'].rolling(20).mean() - (df['Close'].rolling(20).std() * 2)
    
    return df

@st.cache_data(ttl=3600)
def get_global_indices(start_date, end_date):
    tickers = {
        'S&P 500': '^GSPC', 'Brent Oil': 'BZ=F', 
        'Gold': 'GC=F', 'USD Index': 'DX-Y.NYB', 'US 10Y Bond': '^TNX'
    }
    global_df = pd.DataFrame()
    for name, sym in tickers.items():
        try:
            data = yf.download(sym, start=start_date, end=end_date, progress=False, auto_adjust=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data = data.loc[:, ~data.columns.duplicated()]
            if not data.empty and 'Close' in data.columns:
                temp = data[['Close']].rename(columns={'Close': name})
                if global_df.empty: global_df = temp
                else: global_df = global_df.join(temp, how='outer')
        except: continue
    global_df.fillna(method='ffill', inplace=True)
    global_df.fillna(method='bfill', inplace=True)
    return global_df

def prepare_dataset(local_df, global_df):
    combined = local_df.join(global_df, how='inner')
    combined = combined.loc[:, ~combined.columns.duplicated()]
    
    combined['Month_Feat'] = combined.index.month
    combined['DayOfWeek_Feat'] = combined.index.dayofweek
    combined['Quarter_Feat'] = combined.index.quarter

    for col in global_df.columns:
        if col in combined.columns:
            combined[f'{col}_Pct'] = combined[col].pct_change()
            combined[f'{col}_Lag1'] = combined[f'{col}_Pct'].shift(1)
            combined[f'{col}_Lag3'] = combined[f'{col}_Pct'].shift(3)

    combined.dropna(inplace=True)
    return combined

def train_prediction_model(df, target_col='Close', horizon=30):
    data = df.copy()
    data = data.loc[:, ~data.columns.duplicated()]
    
    if target_col not in data.columns:
        return None, 0, 0, 0, pd.DataFrame(), [], []

    try:
        horizon = int(horizon)
        data['Target'] = data[target_col].shift(-horizon)
    except:
        return None, 0, 0, 0, pd.DataFrame(), [], []

    drop_cols = ['Target', 'Open', 'High', 'Low', 'Volume', 'Adj Close']
    feature_cols = [c for c in data.columns if c not in drop_cols]
    
    data.dropna(inplace=True)
    if len(data) < 50: return None, 0, 0, 0, pd.DataFrame(), [], []

    X = data[feature_cols]
    y = data['Target']
    split_idx = int(len(X) * 0.85)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    model = XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    score = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    last_row_features = X.iloc[[-1]]
    future_pred = model.predict(last_row_features)[0]
    
    importance = pd.DataFrame({'Feature': feature_cols, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)
    
    return model, score, mae, future_pred, importance, preds, y_test

# ---------------------------------------------------------
# 5. واجهة المستخدم (Main UI)
# ---------------------------------------------------------

st.sidebar.header("📊 إعدادات التحليل")

# --- التغيير هنا: تحميل القائمة المنسدلة ---
ticker_map = load_tickers_from_file()
selected_ticker = "1120" # الافتراضي

if ticker_map:
    # عرض قائمة منسدلة إذا وجدنا الملف
    st.sidebar.success(f"تم تحميل {len(ticker_map)} شركة من الملف.")
    selected_label = st.sidebar.selectbox("اختر الشركة", options=list(ticker_map.keys()))
    selected_ticker = ticker_map[selected_label]
else:
    # العودة للإدخال اليدوي إذا لم يوجد الملف
    st.sidebar.warning("لم يتم العثور على ملف data/saudi_tickers.csv")
    selected_ticker = st.sidebar.text_input("رمز السهم", value="1120", help="أدخل الرمز يدوياً")
# -------------------------------------------

years_back = st.sidebar.slider("البيانات التاريخية (سنوات)", 1, 10, 3)
forecast_days = st.sidebar.selectbox("فترة التوقع (أيام)", [7, 14, 30, 90], index=2)
include_global = st.sidebar.checkbox("تضمين المؤشرات العالمية", value=True)

if st.sidebar.button("تشغيل التحليل 🚀"):
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years_back*365)
    
    with st.spinner(f'جاري تحليل السهم {selected_ticker}...'):
        local_df = get_stock_data(selected_ticker, start_date, end_date)
        
        if local_df is None:
            st.error(f"لم يتم العثور على بيانات للرمز {selected_ticker}.")
        else:
            full_df = local_df.copy()
            if include_global:
                global_df = get_global_indices(start_date, end_date)
                full_df = prepare_dataset(local_df, global_df)
            else:
                full_df = local_df.dropna()

            # --- العرض ---
            st.title(f"تحليل سهم: {selected_ticker} (TASI)")
            
            # KPIs
            last_close = local_df['Close'].iloc[-1]
            prev_close = local_df['Close'].iloc[-2]
            chg_pct = ((last_close - prev_close) / prev_close) * 100
            
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("آخر سعر", f"{last_close:.2f}", f"{chg_pct:.2f}%")
            kpi2.metric("SMA 200", f"{local_df['SMA_200'].iloc[-1]:.2f}")
            kpi3.metric("RSI", f"{local_df['RSI'].iloc[-1]:.1f}")
            kpi4.metric("الحجم", f"{local_df['Volume'].iloc[-1]:,.0f}")
            
            tab_tech, tab_season, tab_global, tab_ai = st.tabs(["📈 الرسم الفني", "📅 الموسمية", "🌍 الارتباطات", "🤖 توقعات AI"])
            
            with tab_tech:
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=local_df.index, open=local_df['Open'], high=local_df['High'], low=local_df['Low'], close=local_df['Close'], name='السعر'))
                fig.add_trace(go.Scatter(x=local_df.index, y=local_df['BB_High'], line=dict(color='gray', width=1, dash='dot'), name='BB High'))
                fig.add_trace(go.Scatter(x=local_df.index, y=local_df['BB_Low'], line=dict(color='gray', width=1, dash='dot'), name='BB Low'))
                fig.update_layout(height=550, title="السعر مع نطاق بولنجر")
                st.plotly_chart(fig, use_container_width=True)
            
            with tab_season:
                monthly_avg, daily_avg, heatmap_data = analyze_seasonality(local_df)
                c1, c2 = st.columns(2)
                with c1:
                    fig_m = go.Figure(go.Bar(x=monthly_avg.index, y=monthly_avg.values, marker_color=['#2ecc71' if x>0 else '#e74c3c' for x in monthly_avg]))
                    fig_m.update_layout(title="الأداء الشهري", xaxis_title="الشهر")
                    st.plotly_chart(fig_m, use_container_width=True)
                with c2:
                    fig_d = go.Figure(go.Bar(x=daily_avg.index, y=daily_avg.values, marker_color=['#2ecc71' if x>0 else '#e74c3c' for x in daily_avg]))
                    fig_d.update_layout(title="الأداء اليومي", xaxis_title="اليوم")
                    st.plotly_chart(fig_d, use_container_width=True)
                fig_heat = px.imshow(heatmap_data, labels=dict(x="الشهر", y="السنة", color="العائد %"), color_continuous_scale='RdBu')
                st.plotly_chart(fig_heat, use_container_width=True)

            with tab_global:
                if include_global and 'S&P 500' in full_df.columns:
                    corr_cols = ['Close', 'S&P 500', 'Brent Oil', 'Gold', 'US 10Y Bond']
                    avail = [c for c in corr_cols if c in full_df.columns]
                    fig_corr = px.imshow(full_df[avail].corr(), text_auto=True, color_continuous_scale='RdBu_r', title="مصفوفة الارتباط")
                    st.plotly_chart(fig_corr, use_container_width=True)
                else: st.info("البيانات العالمية غير متوفرة.")

            with tab_ai:
                st.subheader(f"توقعات الذكاء الاصطناعي ({forecast_days} يوم)")
                model, score, mae, future_pred, importance, preds, y_test = train_prediction_model(full_df, horizon=forecast_days)
                if model:
                    c1, c2 = st.columns(2)
                    diff = future_pred - last_close
                    c1.metric("السعر المتوقع", f"{future_pred:.2f}", f"{(diff/last_close)*100:.2f}%")
                    c2.progress(max(0.0, min(1.0, score)))
                    c2.caption(f"دقة النموذج R²: {score:.2f}")
                    st.plotly_chart(px.bar(importance.head(10), x='Importance', y='Feature', orientation='h', title="أهم المؤشرات المؤثرة"), use_container_width=True)
                else: st.error("البيانات غير كافية.")
else:
    st.info("👈 اختر الشركة واضغط 'تشغيل التحليل'")
