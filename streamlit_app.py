import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import json
import streamlit.components.v1 as components

# --- استيراد البيانات ---
try:
    from data.saudi_tickers import STOCKS_DB
except ImportError:
    st.error("🚨 ملف البيانات مفقود.")
    st.stop()

TICKERS = {item['symbol']: item['name'] for item in STOCKS_DB}

# --- إعداد الصفحة ---
st.set_page_config(page_title="TASI Native Pro", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .stApp { background-color: #131722; color: #d1d4dc; }
    .block-container { padding-top: 1rem; padding-bottom: 0rem; }
    h1 { font-family: 'Arial'; }
    div.stButton > button { background-color: #2962ff; color: white; border: none; width: 100%; }
</style>
""", unsafe_allow_html=True)

# --- القائمة الجانبية ---
with st.sidebar:
    st.header("⚙️ التحكم")
    selected_symbol = st.selectbox("السهم", list(TICKERS.keys()), format_func=lambda x: f"{TICKERS[x]} ({x.replace('.SR','')})")
    
    st.divider()
    st.subheader("📦 إعدادات المؤشر")
    ATR_LENGTH = st.number_input("ATR Length", value=14)
    ATR_MULT = st.number_input("ATR Multiplier", value=1.5, step=0.1)
    BOX_LOOKBACK = st.slider("Lookback Candles", 20, 200, 100)

# --- الدوال الفنية (Python Logic) ---
def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    return ranges.max(axis=1).ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def get_boxes_data(df):
    # خوارزمية الصناديق (نفس منطق Pine Script)
    df['ATR'] = calculate_atr(df, ATR_LENGTH)
    
    in_series = False; mode = None; start_open = 0.0; end_close = 0.0; start_time = 0
    boxes = []
    
    # تحويل البيانات إلى قائمة للسرعة
    records = df.to_dict('records')
    
    for i in range(len(records)):
        row = records[i]
        close = row['Close']; open_p = row['Open']
        time_val = int(row['Date'].timestamp()) # وقت الشمعة
        
        is_green = close > open_p
        is_red = close < open_p
        current_atr = row['ATR']
        
        if np.isnan(current_atr): continue
        
        if not in_series:
            if is_green: in_series = True; mode = 'bull'; start_open = open_p; start_time = time_val
            elif is_red: in_series = True; mode = 'bear'; start_open = open_p; start_time = time_val
        elif in_series:
            if mode == 'bull' and is_green: end_close = close
            elif mode == 'bear' and is_red: end_close = close
            elif (mode == 'bull' and is_red) or (mode == 'bear' and is_green):
                final_close = end_close if end_close != 0 else start_open
                price_move = abs(final_close - start_open)
                
                if price_move >= current_atr * ATR_MULT:
                    box_top = max(start_open, final_close)
                    box_bottom = min(start_open, final_close)
                    
                    # إضافة الصندوق للقائمة
                    boxes.append({
                        "start": start_time,
                        "end": time_val, # نهاية الصندوق عند الكسر
                        "top": box_top,
                        "bottom": box_bottom,
                        "mid": (box_top + box_bottom) / 2,
                        "type": mode, # bull or bear
                        "color": "rgba(41, 98, 255, 0.2)" if mode == 'bull' else "rgba(255, 82, 82, 0.2)",
                        "borderColor": "#2962ff" if mode == 'bull' else "#ff5252"
                    })
                
                # إعادة تعيين
                in_series = True
                mode = 'bull' if is_green else 'bear'
                start_open = open_p; end_close = close; start_time = time_val
                
    # تمديد الصناديق الأخيرة لتصل إلى الحاضر (اختياري، هنا نوقفها عند الكسر)
    return boxes

# --- تجهيز البيانات ---
@st.cache_data
def get_chart_json(symbol):
    df = yf.download(symbol, period="2y", interval="1d", progress=False)
    if df.empty: return None, None
    
    df.reset_index(inplace=True)
    
    # حساب الصناديق
    boxes = get_boxes_data(df)
    
    # تجهيز الشموع
    candles = []
    for _, row in df.iterrows():
        t = int(row['Date'].timestamp())
        candles.append({
            "time": t, "open": row['Open'], "high": row['High'], "low": row['Low'], "close": row['Close']
        })
        
    return json.dumps(candles), json.dumps(boxes)

# --- التطبيق ---
st.title(f"📈 {TICKERS[selected_symbol]} (مع مؤشر الصناديق)")

candles_json, boxes_json = get_chart_json(selected_symbol)

if candles_json:
    # --- كود الجافاسكربت السحري (Native Plugin) ---
    # هذا الكود ينشئ "Custom Primitive" لرسم المربعات مباشرة على الشارت
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
            // 1. تعريف "راسم الصناديق" (Custom Box Renderer)
            class BoxSeriesRenderer {{
                constructor() {{ this._data = null; }}
                draw(target, priceConverter) {{
                    target.useBitmapCoordinateSpace(scope => this._drawImpl(scope, priceConverter));
                }}
                update(data, options) {{ this._data = data; }}
                _drawImpl(scope, priceConverter) {{
                    if (this._data === null) return;
                    const ctx = scope.context;
                    const timeScale = scope.timeScale;
                    
                    this._data.forEach(box => {{
                        // تحويل الوقت والسعر إلى إحداثيات بكسل
                        const x1 = timeScale.timeToCoordinate(box.start);
                        const x2 = timeScale.timeToCoordinate(box.end);
                        const yTop = priceConverter.priceToCoordinate(box.top);
                        const yBottom = priceConverter.priceToCoordinate(box.bottom);
                        const yMid = priceConverter.priceToCoordinate(box.mid);
                        
                        // التحقق من أن الصندوق داخل الشاشة
                        if (x1 === null || x2 === null) return; 
                        
                        const width = x2 - x1;
                        const height = yBottom - yTop; // في Canvas الـ Y يزيد للأسفل

                        // رسم المستطيل
                        ctx.fillStyle = box.color;
                        ctx.fillRect(x1, yTop, width, height);
                        
                        // رسم الحدود
                        ctx.lineWidth = 1;
                        ctx.strokeStyle = box.borderColor;
                        ctx.strokeRect(x1, yTop, width, height);
                        
                        // رسم خط المنتصف
                        ctx.beginPath();
                        ctx.setLineDash([4, 4]); // خط منقط
                        ctx.moveTo(x1, yMid);
                        ctx.lineTo(x2, yMid);
                        ctx.stroke();
                        ctx.setLineDash([]);
                    }});
                }}
            }}

            // 2. تعريف "المكون الإضافي" (Custom Primitive)
            class BoxPrimitive {{
                constructor(data) {{
                    this._data = data;
                    this._renderer = new BoxSeriesRenderer();
                }}
                updateAllViews() {{ this._renderer.update(this._data, null); }}
                paneViews() {{
                    return [{{
                        renderer: this._renderer,
                    }}];
                }}
                priceAxisViews() {{ return []; }}
                timeAxisViews() {{ return []; }}
                attached(params) {{ 
                    this._renderer.update(this._data, null);
                    params.requestUpdate(); 
                }}
                detached() {{ }}
            }}

            // 3. إعداد الشارت الرئيسي
            const chart = LightweightCharts.createChart(document.getElementById('chart'), {{
                layout: {{ background: {{ type: 'solid', color: '#131722' }}, textColor: '#d1d4dc' }},
                grid: {{ vertLines: {{ color: '#2B2B43' }}, horzLines: {{ color: '#2B2B43' }} }},
                rightPriceScale: {{ borderColor: '#2B2B43' }},
                timeScale: {{ borderColor: '#2B2B43', timeVisible: true }},
                crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
            }});

            // 4. إضافة الشموع
            const candleSeries = chart.addCandlestickSeries({{
                upColor: '#089981', downColor: '#f23645',
                borderUpColor: '#089981', borderDownColor: '#f23645',
                wickUpColor: '#089981', wickDownColor: '#f23645',
            }});
            
            const candlesData = {candles_json};
            candleSeries.setData(candlesData);

            // 5. حقن مؤشر الصناديق
            const boxesData = {boxes_json};
            const boxPrimitive = new BoxPrimitive(boxesData);
            candleSeries.attachPrimitive(boxPrimitive); // تركيب المؤشر على السلسلة

            // 6. التجاوب مع حجم الشاشة
            new ResizeObserver(entries => {{
                if (entries.length === 0 || entries[0].target !== document.getElementById('chart')) return;
                const newRect = entries[0].contentRect;
                chart.applyOptions({{ height: newRect.height, width: newRect.width }});
            }}).observe(document.getElementById('chart'));
            
            chart.timeScale().fitContent();
        </script>
    </body>
    </html>
    """
    
    # عرض الشارت بارتفاع كبير
    components.html(html_code, height=750)

else:
    st.warning("جاري تحميل البيانات...")
