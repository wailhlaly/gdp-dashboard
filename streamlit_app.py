import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from streamlit_option_menu import option_menu
from scipy.signal import argrelextrema
import datetime

# --- استيراد البيانات ---
try:
    from data.saudi_tickers import STOCKS_DB
except ImportError:
    st.error("🚨 ملف البيانات مفقود.")
    st.stop()

# قواميس للبحث
TICKERS = {item['symbol']: item['name'] for item in STOCKS_DB}
SECTORS = {item['symbol']: item['sector'] for item in STOCKS_DB} # الرمز هو المفتاح

# --- 1. إعداد الصفحة ---
st.set_page_config(page_title="TASI Statistics Pro", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .stApp { background-color: #131722; color: #d1d4dc; }
    /* تحسين البطاقات الإحصائية */
    div[data-testid="stMetric"] {
        background-color: #1e222d !important;
        border: 1px solid #2a2e39;
        padding: 10px; border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    [data-testid="stMetricLabel"] { color: #818589 !important; font-size: 0.8rem; }
    [data-testid="stMetricValue"] { color: #e0e0e0 !important; font-size: 1.2rem; }
    
    /* الأزرار */
    div.stButton > button {
        background-color: #2962ff; color: white; border: none; width: 100%; padding: 10px; font-weight: bold; border-radius: 6px;
    }
    
    /* علامات التبويب */
    .stTabs [data-baseweb="tab-list"] { gap: 5px; }
    .stTabs [data-baseweb="tab"] { background-color: #1e222d; color: #d1d4dc; border-radius: 4px; border: 1px solid #2a2e39; }
    .stTabs [aria-selected="true"] { background-color: #2962ff !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# --- 2. التحكم والجلسة ---
if 'market_data' not in st.session_state: st.session_state['market_data'] = pd.DataFrame()
if 'historical_data' not in st.session_state: st.session_state['historical_data'] = {}

# --- 3. دوال المعالجة الإحصائية ---
def calculate_advanced_stats(df_hist):
    """حساب إحصائيات متقدمة للسهم الواحد"""
    # 1. التغير
    change = ((df_hist['Close'].iloc[-1] - df_hist['Close'].iloc[-2]) / df_hist['Close'].iloc[-2]) * 100
    
    # 2. RSI
    delta = df_hist['Close'].diff()
    gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # 3. Volatility (Standard Deviation of returns)
    returns = df_hist['Close'].pct_change()
    volatility = returns.std() * 100 # كنسبة مئوية
    
    # 4. 52-Week Position (موقع السعر بالنسبة لأعلى/أدنى سنوي)
    low_52 = df_hist['Low'].min()
    high_52 = df_hist['High'].max()
    current = df_hist['Close'].iloc[-1]
    position_52 = ((current - low_52) / (high_52 - low_52)) * 100 # 0% عند القاع، 100% عند القمة
    
    return change, rsi.iloc[-1], volatility, position_52, df_hist['Volume'].iloc[-1]

# --- 4. واجهة التحديث ---
with st.sidebar:
    st.header("⚙️ البيانات")
    if st.button("🔄 تحديث الإحصائيات الشاملة"):
        progress = st.progress(0); status = st.empty()
        tickers = list(TICKERS.keys())
        all_stats = []
        
        # نسحب سنة كاملة لحساب إحصائيات دقيقة
        chunk_size = 30
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i:i + chunk_size]
            status.text(f"معالجة {i} من {len(tickers)}...")
            try:
                # نحتاج سنة كاملة لحساب الـ 52-week High/Low بدقة
                raw = yf.download(chunk, period="1y", interval="1d", group_by='ticker', progress=False)
                if not raw.empty:
                    for sym in chunk:
                        try:
                            df = raw[sym].copy()
                            # إصلاح الأسماء المكررة للأعمدة
                            if isinstance(df.columns, pd.MultiIndex):
                                df.columns = df.columns.get_level_values(0)
                            
                            df = df.dropna()
                            if len(df) > 50:
                                chg, rsi, vol, pos52, volume = calculate_advanced_stats(df)
                                
                                all_stats.append({
                                    "Symbol": sym,
                                    "Name": TICKERS.get(sym, sym),
                                    "Sector": SECTORS.get(sym, "أخرى"),
                                    "Price": df['Close'].iloc[-1],
                                    "Change": chg,
                                    "RSI": rsi,
                                    "Volatility": vol, # الانحراف المعياري (المخاطرة)
                                    "Pos_52W": pos52, # الموقع من القمة السنوية
                                    "Volume": volume,
                                    "Turnover": df['Close'].iloc[-1] * volume # قيمة التداول
                                })
                        except: continue
            except: pass
            progress.progress(min((i + chunk_size) / len(tickers), 1.0))
            
        st.session_state['market_data'] = pd.DataFrame(all_stats)
        progress.empty(); status.success("تم!")

# --- 5. لوحة العرض الرئيسية ---
selected_tab = option_menu(
    menu_title=None,
    options=["نظرة عامة", "📊 الإحصائيات العميقة", "الخريطة الحرارية", "الشارت"],
    icons=["speedometer", "bar-chart-line", "grid", "graph-up"],
    orientation="horizontal",
    styles={"container": {"background-color": "transparent"}, "nav-link-selected": {"background-color": "#2962ff"}}
)

if not st.session_state['market_data'].empty:
    df = st.session_state['market_data']
    
    # --- التبويب 1: نظرة عامة ---
    if selected_tab == "نظرة عامة":
        # كروت المعلومات
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("عدد الشركات", len(df))
        c2.metric("المرتفعة 🟢", len(df[df['Change'] > 0]))
        c3.metric("المنخفضة 🔴", len(df[df['Change'] < 0]))
        c4.metric("صافي السيولة", f"{(df['Turnover'].sum() / 1_000_000):.1f}M")
        
        # رسم بياني: توزيع الشركات (Breadth)
        fig_breadth = px.pie(
            names=['مرتفعة', 'منخفضة', 'ثابتة'],
            values=[len(df[df['Change'] > 0]), len(df[df['Change'] < 0]), len(df[df['Change'] == 0])],
            color_discrete_sequence=['#00e676', '#ff1744', '#757575'],
            hole=0.5, title="اتساع السوق (Market Breadth)"
        )
        fig_breadth.update_layout(paper_bgcolor='#1e222d', font_color='white', height=300)
        
        # رسم بياني: أعلى القطاعات سيولة
        sector_liq = df.groupby('Sector')['Turnover'].sum().sort_values(ascending=False).head(10)
        fig_sec = px.bar(
            sector_liq, x=sector_liq.index, y=sector_liq.values,
            title="أعلى القطاعات سيولة", color_discrete_sequence=['#2962ff']
        )
        fig_sec.update_layout(paper_bgcolor='#1e222d', plot_bgcolor='#1e222d', font_color='white', height=300)
        
        col_chart1, col_chart2 = st.columns(2)
        col_chart1.plotly_chart(fig_breadth, use_container_width=True)
        col_chart2.plotly_chart(fig_sec, use_container_width=True)

    # --- التبويب 2: الإحصائيات العميقة (The New Advanced Stats) ---
    elif selected_tab == "📊 الإحصائيات العميقة":
        
        # 1. توزيع RSI (Overbought vs Oversold)
        st.subheader("1. مناطق التشبع (RSI Distribution)")
        bins = [0, 30, 70, 100]
        labels = ['تشبع بيعي (فرص)', 'منطقة عادية', 'تشبع شرائي (خطر)']
        df['RSI_Cat'] = pd.cut(df['RSI'], bins=bins, labels=labels)
        rsi_counts = df['RSI_Cat'].value_counts()
        
        fig_rsi = px.bar(
            x=rsi_counts.index, y=rsi_counts.values,
            color=rsi_counts.index,
            color_discrete_map={'تشبع بيعي (فرص)': '#00e676', 'منطقة عادية': '#757575', 'تشبع شرائي (خطر)': '#ff1744'},
            title="توزيع الشركات حسب مؤشر RSI"
        )
        fig_rsi.update_layout(paper_bgcolor='#131722', plot_bgcolor='#131722', font_color='white', height=300)
        st.plotly_chart(fig_rsi, use_container_width=True)
        
        # عرض أسهم الفرص (RSI < 30)
        oversold = df[df['RSI'] < 30].sort_values('RSI')
        if not oversold.empty:
            st.markdown("**💎 أسهم في مناطق ارتداد محتملة (RSI < 30):**")
            st.dataframe(oversold[['Name', 'Price', 'RSI']].T, use_container_width=True)

        st.divider()

        # 2. تحليل المخاطر (Volatility vs Return)
        st.subheader("2. خريطة المخاطر (Risk vs Return)")
        st.caption("الأسهم في اليمين عالية التذبذب (خطرة)، في الأعلى تحقق أرباحاً.")
        
        fig_vol = px.scatter(
            df, x="Volatility", y="Change",
            size="Turnover", color="Sector",
            hover_name="Name", text="Symbol",
            title="العائد اليومي مقابل المخاطرة (Volatility)",
            labels={"Volatility": "المخاطرة (التقلب)", "Change": "التغير %"}
        )
        fig_vol.update_layout(paper_bgcolor='#131722', plot_bgcolor='#131722', font_color='white', height=500)
        st.plotly_chart(fig_vol, use_container_width=True)

        st.divider()

        # 3. تحليل القمم والقيعان السنوية (52-Week High/Low)
        st.subheader("3. القرب من القمم والقيعان السنوية")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 🚀 أسهم تخترق القمة السنوية (اقوى اتجاه)")
            breakouts = df[df['Pos_52W'] > 95].sort_values('Pos_52W', ascending=False)
            st.dataframe(breakouts[['Name', 'Price', 'Change']].head(10), use_container_width=True)
            
        with c2:
            st.markdown("##### ⚓ أسهم عند القاع السنوي (دعم تاريخي)")
            bottoms = df[df['Pos_52W'] < 5].sort_values('Pos_52W')
            st.dataframe(bottoms[['Name', 'Price', 'Change']].head(10), use_container_width=True)

    # --- التبويب 3: الخريطة الحرارية ---
    elif selected_tab == "الخريطة الحرارية":
        # ألوان احترافية (TradingView)
        fig_map = px.treemap(
            df, path=[px.Constant("السوق"), 'Sector', 'Name'], values='Turnover',
            color='Change',
            color_continuous_scale=[(0, "#f23645"), (0.5, "#2a2e39"), (1, "#089981")],
            range_color=[-3, 3],
            custom_data=['Symbol', 'Price', 'Change']
        )
        fig_map.update_traces(
            texttemplate="<b>%{label}</b><br>%{customdata[2]:.2f}%",
            hovertemplate="<b>%{label}</b><br>السعر: %{customdata[1]:.2f}<br>التغير: %{customdata[2]:.2f}%"
        )
        fig_map.update_layout(margin=dict(t=0, l=0, r=0, b=0), height=600, paper_bgcolor='#131722')
        st.plotly_chart(fig_map, use_container_width=True)

    # --- التبويب 4: الشارت ---
    elif selected_tab == "الشارت":
        # (نفس كود Lightweight Charts السابق، لم أكرره لتوفير المساحة، يمكنك نسخه من الرد السابق إذا أردت)
        st.info("للشارت التفاعلي، يرجى استخدام الكود السابق الخاص بـ Lightweight Charts أو Plotly.")
        
else:
    st.info("👋 اضغط زر التحديث في القائمة الجانبية.")
