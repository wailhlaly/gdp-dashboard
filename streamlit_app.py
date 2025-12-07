import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import json
import plotly.express as px
import streamlit.components.v1 as components

# --- استيراد البيانات ---
try:
    from data.saudi_tickers import STOCKS_DB
except ImportError:
    st.error("🚨 ملف البيانات مفقود.")
    st.stop()

# قواميس للبحث السريع
TICKERS = {item['symbol']: item['name'] for item in STOCKS_DB}
SECTORS = {item['symbol']: item['sector'] for item in STOCKS_DB} # الرمز هو المفتاح الآن

# --- 1. إعداد الصفحة ---
st.set_page_config(page_title="TASI Pro Touch", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .stApp { background-color: #131722; color: #d1d4dc; }
    
    /* تحسين زر التحديث */
    div.stButton > button {
        background-color: #2962ff; color: white; border: none;
        width: 100%; padding: 10px; font-weight: bold; border-radius: 6px;
    }
    div.stButton > button:hover { background-color: #1e53e5; }
    
    /* إخفاء الهوامش */
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

# --- 2. التحكم في الجلسة (Session State) ---
if 'selected_symbol' not in st.session_state:
    st.session_state['selected_symbol'] = "1120.SR" # الراجحي افتراضياً
if 'market_data' not in st.session_state:
    st.session_state['market_data'] = pd.DataFrame()

# --- 3. الدوال الفنية ---
def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    return ranges.max(axis=1).ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def get_boxes_data(df, atr_mult=1.5):
    # خوارزمية الصناديق
    df['ATR'] = calculate_atr(df)
    boxes = []
    in_series = False; mode = None; start_open = 0.0; start_time = 0
    
    records = df.to_dict('records')
    for i in range(len(records)):
        row = records[i]
        close = row['Close']; open_p = row['Open']
        # تحويل التاريخ إلى Unix Timestamp (ثواني) ليتوافق مع Lightweight Charts
        time_val = int(row['Date'].timestamp())
        
        is_green = close > open_p
        is_red = close < open_p
        
        if pd.isna(row['ATR']): continue
        
        if not in_series:
            if is_green: in_series = True; mode = 'bull'; start_open = open_p; start_time = time_val
            elif is_red: in_series = True; mode = 'bear'; start_open = open_p; start_time = time_val
        elif in_series:
            if (mode == 'bull' and is_red) or (mode == 'bear' and is_green):
                # كسر السلسلة
                end_close = records[i-1]['Close'] # إغلاق الشمعة السابقة
                price_move = abs(end_close - start_open)
                
                if price_move >= row['ATR'] * atr_mult:
                    box_top = max(start_open, end_close)
                    box_bottom = min(start_open, end_close)
                    
                    boxes.append({
                        "start": start_time,
                        "end": time_val, 
                        "top": box_top,
                        "bottom": box_bottom,
                        "mid": (box_top + box_bottom) / 2,
                        "color": "rgba(8, 153, 129, 0.2)" if mode == 'bull' else "rgba(242, 54, 69, 0.2)",
                        "border": "#089981" if mode == 'bull' else "#f23645"
                    })
                
                # إعادة تعيين
                in_series = True
                mode = 'bull' if is_green else 'bear'
                start_open = open_p; start_time = time_val
                
    return boxes

# --- 4. جلب البيانات (مرة واحدة) ---
with st.sidebar:
    st.header("⚙️ التحكم")
    if st.button("🔄 تحديث بيانات السوق"):
        tickers = list(TICKERS.keys())
        # نسحب آخر بيانات (يومي) لعمل الخريطة الحرارية
        # نقسمها دفعات لتجنب الأخطاء
        all_data = []
        chunk_size = 50
        status = st.empty()
        
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i:i + chunk_size]
            status.text(f"تحميل {i}...")
            try:
                # نحتاج فقط آخر يومين لحساب التغير للخريطة
                raw = yf.download(chunk, period="5d", interval="1d", group_by='ticker', progress=False)
                if not raw.empty:
                    for sym in chunk:
                        try:
                            df = raw[sym]
                            last = df.iloc[-1]
                            change = ((last['Close'] - df.iloc[-2]['Close']) / df.iloc[-2]['Close']) * 100
                            all_data.append({
                                "Symbol": sym,
                                "Name": TICKERS.get(sym, sym),
                                "Sector": SECTORS.get(sym, "أخرى"),
                                "Price": last['Close'],
                                "Change": change,
                                "Volume": last['Volume']
                            })
                        except: continue
            except: pass
        
        status.empty()
        st.session_state['market_data'] = pd.DataFrame(all_data)

# --- 5. الخريطة الحرارية (Heatmap) ---
if not st.session_state['market_data'].empty:
    df_map = st.session_state['market_data']
    
    # تحسين الألوان (TradingView Style)
    # أحمر غامق (-3%) -> رمادي (0%) -> أخضر غامق (+3%)
    fig_map = px.treemap(
        df_map, 
        path=[px.Constant("السوق"), 'Sector', 'Symbol'], # استخدام الرموز فقط
        values='Price', # حجم المربع (يمكن تغييره لـ Volume)
        color='Change',
        color_continuous_scale=[
            (0, "rgb(242, 54, 69)"),   # أحمر TV
            (0.5, "rgb(43, 43, 67)"),  # رمادي غامق (حيادي)
            (1, "rgb(8, 153, 129)")    # أخضر TV
        ],
        range_color=[-3, 3], # تثبيت النطاق لتوحيد الألوان
        custom_data=['Name', 'Price', 'Change']
    )
    
    fig_map.update_traces(
        textinfo="label+text",
        texttemplate="%{label}<br>%{customdata[2]:.2f}%", # يظهر الرمز والنسبة
        hovertemplate="<b>%{customdata[0]}</b><br>السعر: %{customdata[1]:.2f}<br>التغير: %{customdata[2]:.2f}%",
        textfont=dict(size=14, color='white')
    )
    
    fig_map.update_layout(
        margin=dict(t=0, l=0, r=0, b=0),
        height=350, # ارتفاع مناسب للخريطة
        paper_bgcolor='#131722'
    )
    
    # ميزة التفاعل: عند الضغط يتم تحديث الجلسة
    selected_points = st.plotly_chart(fig_map, use_container_width=True, on_select="rerun")
    
    # منطق التقاط الضغط
    if selected_points and len(selected_points['selection']['points']) > 0:
        clicked_point = selected_points['selection']['points'][0]
        # التأكد من ضغط سهم (الرمز) وليس قطاع
        if 'label' in clicked_point and clicked_point['label'] in TICKERS:
            st.session_state['selected_symbol'] = clicked_point['label']

# --- 6. الشارت (Lightweight Charts - Native Touch) ---
current_symbol = st.session_state['selected_symbol']
st.markdown(f"### 📈 {TICKERS.get(current_symbol, current_symbol)} ({current_symbol.replace('.SR','')})")

# جلب بيانات السهم المختار
@st.cache_data
def load_chart_data(symbol):
    df = yf.download(symbol, period="2y", interval="1d", progress=False)
    if df.empty: return None, None
    df.reset_index(inplace=True)
    
    # بيانات الشموع
    candles = []
    volumes = []
    
    # بيانات الصناديق
    boxes = get_boxes_data(df)
    
    # حساب المتوسطات
    df['EMA20'] = df['Close'].ewm(span=20).mean()
    df['EMA50'] = df['Close'].ewm(span=50).mean()
    
    ema20 = []; ema50 = []

    for _, row in df.iterrows():
        t = int(row['Date'].timestamp())
        candles.append({"time": t, "open": row['Open'], "high": row['High'], "low": row['Low'], "close": row['Close']})
        
        # تلوين الفوليوم
        vol_col = "rgba(8, 153, 129, 0.3)" if row['Close'] >= row['Open'] else "rgba(242, 54, 69, 0.3)"
        volumes.append({"time": t, "value": row['Volume'], "color": vol_col})
        
        if not pd.isna(row['EMA20']): ema20.append({"time": t, "value": row['EMA20']})
        if not pd.isna(row['EMA50']): ema50.append({"time": t, "value": row['EMA50']})
        
    return json.dumps(candles), json.dumps(volumes), json.dumps(boxes), json.dumps(ema20), json.dumps(ema50)

candles_json, vol_json, boxes_json, ema20_json, ema50_json = load_chart_data(current_symbol)

if candles_json:
    # كود HTML/JS لرسم الشارت مع دعم اللمس الكامل ورسم الصناديق
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
        <style>
            body {{ margin: 0; padding: 0; background-color: #131722; overflow: hidden; }}
            #chart {{ position: absolute; width: 100%; height: 100%; }}
        </style>
    </head>
    <body>
        <div id="chart"></div>
        <script>
            // --- بيانات ---
            const candleData = {candles_json};
            const volumeData = {vol_json};
            const boxData = {boxes_json};
            const ema20Data = {ema20_json};
            const ema50Data = {ema50_json};

            // --- إعداد الشارت ---
            const chart = LightweightCharts.createChart(document.getElementById('chart'), {{
                layout: {{ background: {{ type: 'solid', color: '#131722' }}, textColor: '#d1d4dc' }},
                grid: {{ vertLines: {{ color: '#2B2B43' }}, horzLines: {{ color: '#2B2B43' }} }},
                crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
                rightPriceScale: {{ borderColor: '#2B2B43' }},
                timeScale: {{ borderColor: '#2B2B43', timeVisible: true }},
                // تفعيل إيماءات اللمس الطبيعية
                handleScale: {{ axisPressedMouseMove: true, mouseWheel: true, pinch: true }},
                handleScroll: {{ mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false }}
            }});

            // 1. الشموع
            const mainSeries = chart.addCandlestickSeries({{
                upColor: '#089981', downColor: '#f23645',
                borderUpColor: '#089981', borderDownColor: '#f23645',
                wickUpColor: '#089981', wickDownColor: '#f23645',
            }});
            mainSeries.setData(candleData);

            // 2. الفوليوم (Overlay)
            const volSeries = chart.addHistogramSeries({{
                priceFormat: {{ type: 'volume' }},
                priceScaleId: '', // دمج
            }});
            volSeries.priceScale().applyOptions({{ scaleMargins: {{ top: 0.8, bottom: 0 }} }});
            volSeries.setData(volumeData);

            // 3. المتوسطات
            const ema20 = chart.addLineSeries({{ color: '#2962ff', lineWidth: 1, title: 'EMA 20' }});
            ema20.setData(ema20Data);
            const ema50 = chart.addLineSeries({{ color: '#ff9800', lineWidth: 1, title: 'EMA 50' }});
            ema50.setData(ema50Data);

            // --- 4. رسم الصناديق (Box Plugin) ---
            // نستخدم Canvas لرسم الصناديق كطبقة إضافية
            
            // تعريف الـ Plugin
            class BoxPainter {{
                constructor() {{ this._data = boxData; }}
                draw(target, priceConverter) {{
                    target.useBitmapCoordinateSpace(scope => this._drawImpl(scope, priceConverter));
                }}
                _drawImpl(scope, priceConverter) {{
                    const ctx = scope.context;
                    const timeScale = scope.timeScale;
                    
                    this._data.forEach(box => {{
                        const x1 = timeScale.timeToCoordinate(box.start);
                        const x2 = timeScale.timeToCoordinate(box.end);
                        // إذا كان الصندوق خارج الشاشة، لا ترسم
                        if (x1 === null || x2 === null) return;
                        
                        const yTop = priceConverter.priceToCoordinate(box.top);
                        const yBottom = priceConverter.priceToCoordinate(box.bottom);
                        const yMid = priceConverter.priceToCoordinate(box.mid);
                        
                        // رسم الخلفية
                        ctx.fillStyle = box.color;
                        ctx.fillRect(x1, yTop, x2 - x1, yBottom - yTop);
                        
                        // رسم الحدود
                        ctx.strokeStyle = box.border;
                        ctx.lineWidth = 1;
                        ctx.strokeRect(x1, yTop, x2 - x1, yBottom - yTop);
                        
                        // خط المنتصف
                        ctx.beginPath();
                        ctx.setLineDash([4, 4]);
                        ctx.strokeStyle = '#2962ff';
                        ctx.moveTo(x1, yMid); ctx.lineTo(x2, yMid);
                        ctx.stroke();
                        ctx.setLineDash([]);
                    }});
                }}
            }}
            
            // ربط الـ Plugin بالسلسلة
            const boxPrimitive = {{
                _renderer: new BoxPainter(),
                attached: () => {{}},
                detached: () => {{}},
                paneViews: () => [{{ renderer: new BoxPainter() }}],
                priceAxisViews: () => [],
                timeAxisViews: () => [],
                updateAllViews: () => {{}}
            }};
            mainSeries.attachPrimitive(boxPrimitive);

            // ضبط الحجم
            new ResizeObserver(entries => {{
                if (entries.length === 0) return;
                const newRect = entries[0].contentRect;
                chart.applyOptions({{ height: newRect.height, width: newRect.width }});
            }}).observe(document.getElementById('chart'));
            
            chart.timeScale().fitContent();
        </script>
    </body>
    </html>
    """
    
    components.html(html_code, height=600)
else:
    st.info("اضغط 'تحديث بيانات السوق' للبدء.")
