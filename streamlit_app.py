import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime, timedelta
import time

# ---------------------------------------------------------
# 1. إعدادات الصفحة
# ---------------------------------------------------------
st.set_page_config(
    page_title="محلل تاسي المتقدم (Pro)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { direction: rtl; }
    h1, h2, h3, h4, p, div { font-family: 'Tajawal', sans-serif; text-align: right; }
    .stMetric { text-align: right !important; direction: rtl; }
    div[data-testid="stSidebar"] { text-align: right; }
    /* تنسيق صناديق الشرح */
    .explanation-box {
        background-color: #f0f2f6;
        border-right: 5px solid #ff4b4b;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
        color: #31333F;
    }
    .positive-impact { border-right-color: #2ecc71; }
    .negative-impact { border-right-color: #e74c3c; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. دوال التحليل المساعدة
# ---------------------------------------------------------

def generate_explanation(feature_name, importance, correlation):
    """توليد تعليل نصي لتأثير العامل"""
    impact_type = "طردية" if correlation > 0 else "عكسية"
    direction = "يرتفع" if correlation > 0 else "ينخفض"
    
    # ترجمة الأسماء للعربية
    name_map = {
        'S&P 500': 'السوق الأمريكي (S&P500)',
        'Brent Oil': 'سعر نفط برنت',
        'Gold': 'سعر الذهب',
        'US 10Y Bond': 'عائد السندات الأمريكية',
        'RSI': 'مؤشر القوة النسبية (RSI)',
        'SMA_50': 'المتوسط المتحرك 50',
        'SMA_200': 'المتوسط المتحرك 200',
        'Month_Feat': 'موسمية الشهر الحالي'
    }
    ar_name = name_map.get(feature_name, feature_name)
    
    strength = "تأثير قوي جداً" if importance > 0.2 else "تأثير متوسط"
    
    explanation = f"""
    **{ar_name}**: ({strength})
    * **لماذا هو مؤثر؟** لأن البيانات التاريخية تظهر علاقة **{impact_type}** ({correlation:.2f}) مع سهمك.
    * **كيف سيؤثر؟** بناءً على وضعه الحالي، عندما يرتفع هذا المؤشر، يميل سهمك لأن **{direction}**.
    """
    return explanation, "positive-impact" if correlation > 0 else "negative-impact"

def analyze_market_breadth(tickers_list):
    """تحليل شامل لقائمة أسهم (أداء الشهر + السيولة مقابل الوزن)"""
    market_data = []
    
    # تحديد تواريخ الشهر الحالي
    end_date = datetime.now()
    start_date = end_date.replace(day=1) # بداية الشهر الحالي
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, ticker in enumerate(tickers_list):
        try:
            # تنظيف الرمز
            clean_ticker = str(ticker).strip()
            if not clean_ticker.endswith('.SR'):
                clean_ticker = f"{clean_ticker}.SR"
            
            # جلب بيانات سريعة
            stock = yf.Ticker(clean_ticker)
            hist = stock.history(start=start_date, end=end_date)
            
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                start_price = hist['Open'].iloc[0]
                pct_change = ((current_price - start_price) / start_price) * 100
                total_volume = hist['Volume'].sum()
                avg_volume = hist['Volume'].mean()
                traded_value = total_volume * current_price # سيولة تقريبية
                
                # محاولة جلب القيمة السوقية (قد تكون بطيئة قليلاً)
                # نستخدم الحجم كبديل للوزن إذا لم تتوفر القيمة السوقية لتسريع العملية
                # هنا سنفترض أن الحجم * السعر هو مؤشر للوزن في المحفظة اليومية
                
                market_data.append({
                    'Ticker': clean_ticker.replace('.SR', ''),
                    'Price': current_price,
                    'Change%': pct_change,
                    'Liquidity': traded_value,
                    'Volume': avg_volume,
                    # معادلة خفة السهم: (السيولة / السعر) كلما زادت السيولة مع سعر أقل كان أخف، 
                    # أو ببساطة: الأسهم الخفيفة هي ذات القيمة السوقية المنخفضة. 
                    # هنا سنستخدم اللوغاريتم للرسم البياني
                    'Weight_Proxy': current_price * avg_volume # مؤشر تقريبي للوزن
                })
        except Exception:
            continue
        
        # تحديث شريط التقدم
        progress = (i + 1) / len(tickers_list)
        progress_bar.progress(progress)
        status_text.text(f"جاري تحليل {clean_ticker}...")
    
    status_text.empty()
    progress_bar.empty()
    
    return pd.DataFrame(market_data)

# ---------------------------------------------------------
# 3. جلب البيانات الأساسية
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def get_main_data(ticker, period_years):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_years*365)
    
    if not ticker.endswith('.SR'): ticker = f"{ticker}.SR"
    
    df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        try: df.columns = df.columns.get_level_values(0)
        except: pass
    df = df.loc[:, ~df.columns.duplicated()]
    
    if df.empty or 'Close' not in df.columns: return None
    
    # معالجة وتنظيف
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # المؤشرات
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['RSI'] = 100 - (100 / (1 + df['Close'].diff().apply(lambda x: x if x>0 else 0).rolling(14).mean() / df['Close'].diff().apply(lambda x: -x if x<0 else 0).rolling(14).mean()))
    
    # الموسمية
    df['Month_Feat'] = df.index.month
    
    return df

@st.cache_data(ttl=3600)
def get_global_data(start_date):
    tickers = {'S&P 500': '^GSPC', 'Brent Oil': 'BZ=F', 'Gold': 'GC=F'}
    global_df = pd.DataFrame()
    for name, sym in tickers.items():
        try:
            d = yf.download(sym, start=start_date, progress=False, auto_adjust=False)
            if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
            d = d.loc[:, ~d.columns.duplicated()]
            if 'Close' in d.columns:
                global_df[name] = d['Close']
        except: continue
    
    # ملء الفراغات وتوحيد التاريخ
    global_df = global_df.resample('D').ffill()
    return global_df

# ---------------------------------------------------------
# 4. النمذجة (مع حساب الارتباط للتفسير)
# ---------------------------------------------------------
def train_explainable_model(df, horizon=30):
    data = df.copy().dropna()
    data['Target'] = data['Close'].shift(-int(horizon))
    
    features = [c for c in data.columns if c not in ['Target', 'Open', 'High', 'Low', 'Volume', 'Adj Close']]
    data.dropna(inplace=True)
    
    X = data[features]
    y = data['Target']
    
    # حساب الارتباط (Correlation) لغرض التفسير
    correlations = data[features].corrwith(data['Close']) # ارتباط مع السعر الحالي كبديل للفهم
    
    split = int(len(X)*0.85)
    model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05)
    model.fit(X.iloc[:split], y.iloc[:split])
    
    future_pred = model.predict(X.iloc[[-1]])[0]
    
    # تجميع البيانات للتفسير
    importances = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_,
        'Correlation': correlations.values
    }).sort_values('Importance', ascending=False)
    
    return future_pred, importances

# ---------------------------------------------------------
# 5. الواجهة الرئيسية
# ---------------------------------------------------------

# الشريط الجانبي
st.sidebar.title("🛠️ أدوات التحكم")
mode = st.sidebar.radio("اختر النمط:", ["تحليل سهم واحد", "تحليل ملف السوق"])

if mode == "تحليل سهم واحد":
    ticker = st.sidebar.text_input("رمز السهم", "1120")
    horizon = st.sidebar.selectbox("فترة التوقع", [7, 30, 90], index=1)
    
    if st.sidebar.button("ابدأ التحليل 🚀"):
        with st.spinner("جاري تحليل البيانات وربط العلاقات..."):
            local = get_main_data(ticker, 3)
            if local is not None:
                glob = get_global_data(local.index[0])
                full = local.join(glob, how='left').fillna(method='ffill')
                
                # 1. التوقع والتعليل
                pred, feats = train_explainable_model(full, horizon)
                last_price = local['Close'].iloc[-1]
                
                # --- العرض ---
                st.title(f"التحليل الذكي لسهم {ticker}")
                
                c1, c2 = st.columns(2)
                diff = pred - last_price
                color = "green" if diff > 0 else "red"
                c1.markdown(f"### السعر المتوقع ({horizon} يوم): <span style='color:{color}'>{pred:.2f} ريال</span>", unsafe_allow_html=True)
                c1.metric("التغير المتوقع", f"{diff:.2f}", f"{(diff/last_price)*100:.2f}%")
                
                # --- قسم التعليل (الجديد) ---
                st.markdown("---")
                st.subheader("🧐 لماذا هذا التوقع؟ (تحليل العوامل المؤثرة)")
                
                col_exp, col_chart = st.columns([1, 1])
                
                with col_exp:
                    # أخذ أهم 3 عوامل وشرحها
                    top_3 = feats.head(3)
                    for index, row in top_3.iterrows():
                        text, style_class = generate_explanation(row['Feature'], row['Importance'], row['Correlation'])
                        st.markdown(f"""
                        <div class="explanation-box {style_class}">
                        {text}
                        </div>
                        """, unsafe_allow_html=True)

                with col_chart:
                    fig = px.bar(feats.head(7), x='Importance', y='Feature', orientation='h', 
                                 title="وزن العوامل في اتخاذ القرار", color='Correlation',
                                 color_continuous_scale='RdBu')
                    st.plotly_chart(fig, use_container_width=True)

            else:
                st.error("لم يتم العثور على البيانات.")

elif mode == "تحليل ملف السوق":
    st.sidebar.markdown("---")
    uploaded_file = st.sidebar.file_uploader("ارفع ملف saudi_tickers (csv/xlsx)", type=['csv', 'xlsx'])
    
    if uploaded_file and st.sidebar.button("تحليل السوق بالكامل 📊"):
        # قراءة الملف
        try:
            if uploaded_file.name.endswith('.csv'):
                df_tickers = pd.read_csv(uploaded_file)
            else:
                df_tickers = pd.read_excel(uploaded_file)
            
            # نفترض أن العمود اسمه 'Ticker' أو 'Symbol' أو أول عمود
            ticker_col = [c for c in df_tickers.columns if 'ticker' in c.lower() or 'symbol' in c.lower() or 'رمز' in c.lower()]
            if ticker_col:
                tickers_list = df_tickers[ticker_col[0]].tolist()
            else:
                tickers_list = df_tickers.iloc[:, 0].tolist()
            
            # تقليص القائمة للتجربة (اختياري، يمكنك إزالته لتحليل الكل)
            # tickers_list = tickers_list[:30] 
            
            st.title("📊 تقرير مراقبة السوق (Market Watch)")
            st.write(f"جاري تحليل أداء {len(tickers_list)} شركة لهذا الشهر...")
            
            market_df = analyze_market_breadth(tickers_list)
            
            if not market_df.empty:
                # 1. شارت تقدم وتراجع السوق
                st.subheader("1. أداء السوق لشهر الحالي")
                
                positive = market_df[market_df['Change%'] > 0].shape[0]
                negative = market_df[market_df['Change%'] < 0].shape[0]
                
                c1, c2 = st.columns([1, 2])
                with c1:
                    fig_pie = px.pie(names=['صاعد', 'هابط'], values=[positive, negative], 
                                     color_discrete_sequence=['#2ecc71', '#e74c3c'], hole=0.4)
                    st.plotly_chart(fig_pie, use_container_width=True)
                with c2:
                    # أعلى الرابحين والخاسرين
                    top_gainers = market_df.nlargest(5, 'Change%')
                    st.write("**الأكثر ارتفاعاً:**")
                    st.dataframe(top_gainers[['Ticker', 'Price', 'Change%']])
                
                st.markdown("---")
                
                # 2. شارت السيولة مقابل الوزن (خفة الأسهم)
                st.subheader("2. خريطة السيولة والوزن (Lightness Map)")
                st.info("💡 **كيف تقرأ هذا الشارت؟** الدوائر الكبيرة تعني سيولة عالية. الأسهم في الجهة اليسرى (وزن تقريبي منخفض) مع ارتفاع للأعلى تعني أسهم خفيفة دخلتها سيولة عالية (فرص مضاربية).")
                
                # استخدام مقياس لوغاريتمي للوزن والسيولة لرؤية أفضل
                fig_bubble = px.scatter(
                    market_df,
                    x="Weight_Proxy",
                    y="Liquidity",
                    size="Liquidity",
                    color="Change%",
                    hover_name="Ticker",
                    log_x=True,
                    log_y=True,
                    color_continuous_scale="RdBu",
                    labels={"Weight_Proxy": "الوزن التقريبي (سعر × حجم)", "Liquidity": "قيمة التداول (السيولة)"},
                    title="توزيع الشركات: الوزن مقابل السيولة (لون الدائرة يمثل التغير %)"
                )
                st.plotly_chart(fig_bubble, use_container_width=True)
                
                # جدول تفصيلي
                with st.expander("عرض الجدول الكامل للبيانات"):
                    st.dataframe(market_df.sort_values('Change%', ascending=False))
                    
        except Exception as e:
            st.error(f"حدث خطأ أثناء قراءة الملف: {e}")
