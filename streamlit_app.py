import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import json
import streamlit.components.v1 as components

# --- 1. استيراد البيانات ---
try:
    from data.saudi_tickers import STOCKS_DB
except ImportError:
    st.error("🚨 ملف البيانات مفقود.")
    st.stop()

TICKERS = {item['symbol']: item['name'] for item in STOCKS_DB}

# --- 2. إعداد الصفحة ---
st.set_page_config(page_title="TASI Custom Engine", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #d1d4dc; }
    h1 { font-family: 'Arial'; color: white; }
    .block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# --- 3. القائمة الجانبية ---
with st.sidebar:
    st.header("⚙️ المحرك الخاص")
    selected_symbol = st.selectbox("السهم", list(TICKERS.keys()))
    ATR_MULT = st.number_input("ATR Multiplier", 1.0, 3.0, 1.5)
    BOX_LOOKBACK = st.slider("Lookback", 20, 200, 100)

# --- 4. معالجة البيانات في بايثون ---
def get_data(symbol):
    df = yf.download(symbol, period="1y", interval="1d", progress=False)
    if df.empty: return None
    
    df.reset_index(inplace=True)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns] # تنظيف الأعمدة
    
    # حساب ATR للصناديق
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['ATR'] = ranges.max(axis=1).ewm(alpha=1/14, adjust=False).mean()
    
    # حساب المتوسط (EMA)
    df['EMA'] = df['Close'].ewm(span=20, adjust=False).mean()

    # تحويل البيانات لقائمة قواميس (JSON ready)
    chart_data = []
    boxes = []
    
    # منطق الصناديق (Python Logic)
    in_series = False; mode = None; start_open = 0.0; end_close = 0.0; start_idx = 0
    
    for i, row in df.iterrows():
        # بيانات الشارت الأساسية
        chart_data.append({
            "d": row['Date'].strftime('%Y-%m-%d'),
            "o": round(row['Open'], 2),
            "h": round(row['High'], 2),
            "l": round(row['Low'], 2),
            "c": round(row['Close'], 2),
            "v": int(row['Volume']),
            "ema": round(row['EMA'], 2) if not pd.isna(row['EMA']) else None
        })
        
        # منطق الصناديق (نفس الخوارزمية السابقة)
        close = row['Close']; open_p = row['Open']
        is_green = close > open_p; is_red = close < open_p
        
        if pd.isna(row['ATR']): continue
        
        if not in_series:
            if is_green: in_series = True; mode = 'bull'; start_open = open_p; start_idx = i
            elif is_red: in_series = True; mode = 'bear'; start_open = open_p; start_idx = i
        elif in_series:
            if mode == 'bull' and is_green: end_close = close
            elif mode == 'bear' and is_red: end_close = close
            elif (mode == 'bull' and is_red) or (mode == 'bear' and is_green):
                final_close = end_close if end_close != 0 else start_open
                if abs(final_close - start_open) >= row['ATR'] * ATR_MULT:
                    boxes.append({
                        "start": start_idx, "end": i, 
                        "top": max(start_open, final_close), "bottom": min(start_open, final_close),
                        "type": mode
                    })
                in_series = True; mode = 'bull' if is_green else 'bear'; start_open = open_p; end_close = close; start_idx = i

    return json.dumps(chart_data), json.dumps(boxes)

# --- 5. العرض ---
st.title(f"🎨 {TICKERS[selected_symbol]} (محرك رسم خاص)")

json_data, json_boxes = get_data(selected_symbol)

if json_data:
    # --- هنا السحر: كود HTML/JS يبني الشارت من الصفر ---
    # نستخدم Canvas API للرسم بالبكسل
    
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ background-color: #131722; margin: 0; overflow: hidden; font-family: sans-serif; }}
            #canvas-container {{ position: relative; width: 100%; height: 700px; cursor: crosshair; }}
            canvas {{ position: absolute; top: 0; left: 0; }}
            #tooltip {{ 
                position: absolute; display: none; pointer-events: none;
                background: rgba(30, 34, 45, 0.9); border: 1px solid #2a2e39; color: white;
                padding: 8px; border-radius: 4px; font-size: 12px; z-index: 10;
            }}
        </style>
    </head>
    <body>
        <div id="canvas-container">
            <canvas id="mainLayer"></canvas>
            <canvas id="crosshairLayer"></canvas>
            <div id="tooltip"></div>
        </div>

        <script>
            // 1. استقبال البيانات من بايثون
            const data = {json_data};
            const boxes = {json_boxes};
            
            // إعدادات الشارت
            const config = {{
                bg: "#131722",
                grid: "#1e222d",
                up: "#089981",
                down: "#f23645",
                text: "#d1d4dc",
                wickWidth: 1,
                candleGap: 0.2 // نسبة الفراغ بين الشموع
            }};

            // متغيرات الحالة
            let offsetX = 0; // للإزاحة (Pan)
            let scaleX = 10; // عرض الشمعة (Zoom)
            let canvasWidth, canvasHeight;
            
            const container = document.getElementById('canvas-container');
            const mainCanvas = document.getElementById('mainLayer');
            const crossCanvas = document.getElementById('crosshairLayer');
            const ctx = mainCanvas.getContext('2d');
            const ctxCross = crossCanvas.getContext('2d');
            const tooltip = document.getElementById('tooltip');

            // ضبط الحجم
            function resize() {{
                canvasWidth = container.clientWidth;
                canvasHeight = container.clientHeight;
                mainCanvas.width = crossCanvas.width = canvasWidth;
                mainCanvas.height = crossCanvas.height = canvasHeight;
                draw();
            }}
            window.addEventListener('resize', resize);

            // --- 2. محرك الرسم (Core Rendering Engine) ---
            function draw() {{
                // مسح الشاشة
                ctx.fillStyle = config.bg;
                ctx.fillRect(0, 0, canvasWidth, canvasHeight);

                // تحديد البيانات المرئية (Visible Range)
                const visibleCandles = Math.ceil(canvasWidth / scaleX);
                const startIndex = Math.max(0, Math.floor(data.length - visibleCandles - offsetX));
                const endIndex = Math.min(data.length, Math.ceil(startIndex + visibleCandles + 1));
                const viewData = data.slice(startIndex, endIndex);

                if (viewData.length === 0) return;

                // حساب المقياس السعري (Y-Axis Scaling)
                let minPrice = Infinity, maxPrice = -Infinity;
                let maxVol = 0;
                
                viewData.forEach(d => {{
                    if (d.l < minPrice) minPrice = d.l;
                    if (d.h > maxPrice) maxPrice = d.h;
                    if (d.v > maxVol) maxVol = d.v;
                }});
                
                // إضافة هامش للسعر
                const padding = (maxPrice - minPrice) * 0.1;
                maxPrice += padding; minPrice -= padding;
                const priceRange = maxPrice - minPrice;

                // دوال التحويل (Math to Pixels)
                const getX = (index) => (index - startIndex) * scaleX;
                const getY = (price) => canvasHeight - ((price - minPrice) / priceRange) * canvasHeight;

                // --- رسم الشبكة (Grid) ---
                ctx.strokeStyle = config.grid;
                ctx.lineWidth = 1;
                // خطوط عرضية (سعرية)
                const steps = 10;
                for(let i=0; i<steps; i++) {{
                    const p = minPrice + (priceRange/steps)*i;
                    const y = getY(p);
                    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvasWidth, y); ctx.stroke();
                    // كتابة السعر
                    ctx.fillStyle = config.text;
                    ctx.fillText(p.toFixed(2), canvasWidth - 50, y - 5);
                }}

                // --- رسم الصناديق (Custom Boxes) ---
                boxes.forEach(box => {{
                    // التحقق إذا كان الصندوق في النطاق المرئي
                    if (box.end < startIndex || box.start > endIndex) return;

                    const x1 = getX(box.start);
                    const x2 = getX(box.end);
                    const yTop = getY(box.top);
                    const yBottom = getY(box.bottom);
                    const yMid = getY((box.top + box.bottom)/2);

                    ctx.fillStyle = box.type === 'bull' ? "rgba(8, 153, 129, 0.15)" : "rgba(242, 54, 69, 0.15)";
                    ctx.fillRect(x1, yTop, x2 - x1, yBottom - yTop);
                    
                    ctx.strokeStyle = box.type === 'bull' ? config.up : config.down;
                    ctx.strokeRect(x1, yTop, x2 - x1, yBottom - yTop);
                    
                    // خط المنتصف
                    ctx.beginPath();
                    ctx.setLineDash([5, 5]);
                    ctx.strokeStyle = "#2962ff";
                    ctx.moveTo(x1, yMid); ctx.lineTo(x2, yMid);
                    ctx.stroke();
                    ctx.setLineDash([]);
                }});

                // --- رسم الحجم (Volume) ---
                const volHeight = canvasHeight * 0.2; // 20% من الشاشة
                viewData.forEach((d, i) => {{
                    const x = getX(startIndex + i);
                    const h = (d.v / maxVol) * volHeight;
                    const w = scaleX * (1 - config.candleGap);
                    
                    ctx.fillStyle = d.c >= d.o ? "rgba(8, 153, 129, 0.3)" : "rgba(242, 54, 69, 0.3)";
                    ctx.fillRect(x + (scaleX * config.candleGap)/2, canvasHeight - h, w, h);
                }});

                // --- رسم الشموع (Candles) ---
                const candleWidth = scaleX * (1 - config.candleGap);
                
                viewData.forEach((d, i) => {{
                    const xCenter = getX(startIndex + i) + scaleX/2;
                    const yOpen = getY(d.o);
                    const yClose = getY(d.c);
                    const yHigh = getY(d.h);
                    const yLow = getY(d.l);
                    
                    ctx.strokeStyle = d.c >= d.o ? config.up : config.down;
                    ctx.fillStyle = d.c >= d.o ? config.up : config.down;
                    
                    // الذيل (Wick)
                    ctx.beginPath();
                    ctx.moveTo(xCenter, yHigh);
                    ctx.lineTo(xCenter, yLow);
                    ctx.stroke();
                    
                    // الجسم (Body)
                    const bodyTop = Math.min(yOpen, yClose);
                    const bodyHeight = Math.max(Math.abs(yClose - yOpen), 1); // 1px minimum
                    ctx.fillRect(xCenter - candleWidth/2, bodyTop, candleWidth, bodyHeight);
                }});

                // --- رسم المتوسط (EMA Line) ---
                ctx.beginPath();
                ctx.strokeStyle = "#ff9800";
                ctx.lineWidth = 2;
                let first = true;
                viewData.forEach((d, i) => {{
                    if (d.ema) {{
                        const x = getX(startIndex + i) + scaleX/2;
                        const y = getY(d.ema);
                        if (first) {{ ctx.moveTo(x, y); first = false; }}
                        else {{ ctx.lineTo(x, y); }}
                    }}
                }});
                ctx.stroke();
            }}

            // --- 3. التفاعل (Interactivity) ---
            
            // السحب (Pan)
            let isDragging = false;
            let lastX = 0;

            crossCanvas.addEventListener('mousedown', e => {{ isDragging = true; lastX = e.clientX; }});
            crossCanvas.addEventListener('mouseup', () => {{ isDragging = false; }});
            crossCanvas.addEventListener('mouseleave', () => {{ isDragging = false; }});
            
            crossCanvas.addEventListener('mousemove', e => {{
                const rect = crossCanvas.getBoundingClientRect();
                const mouseX = e.clientX - rect.left;
                const mouseY = e.clientY - rect.top;

                // منطق السحب
                if (isDragging) {{
                    const dx = e.clientX - lastX;
                    offsetX += dx / scaleX; // تحويل البكسل إلى عدد شموع
                    // حدود السحب
                    if (offsetX > 0) offsetX = 0; 
                    // if (offsetX < -data.length) offsetX = -data.length;
                    lastX = e.clientX;
                    draw();
                }}

                // منطق المؤشر (Crosshair)
                ctxCross.clearRect(0, 0, canvasWidth, canvasHeight);
                
                // خطوط
                ctxCross.strokeStyle = 'rgba(255, 255, 255, 0.2)';
                ctxCross.setLineDash([5, 5]);
                ctxCross.beginPath();
                ctxCross.moveTo(mouseX, 0); ctxCross.lineTo(mouseX, canvasHeight);
                ctxCross.moveTo(0, mouseY); ctxCross.lineTo(canvasWidth, mouseY);
                ctxCross.stroke();
                ctxCross.setLineDash([]);

                // إظهار البيانات
                // نحتاج معرفة أي شمعة نحن فوقها
                const visibleCandles = Math.ceil(canvasWidth / scaleX);
                const startIndex = Math.max(0, Math.floor(data.length - visibleCandles - offsetX));
                const indexHover = Math.floor(mouseX / scaleX) + startIndex;
                
                if (indexHover >= 0 && indexHover < data.length) {{
                    const d = data[indexHover];
                    tooltip.style.display = 'block';
                    tooltip.style.left = (mouseX + 15) + 'px';
                    tooltip.style.top = (mouseY + 15) + 'px';
                    tooltip.innerHTML = `
                        <b>${{d.d}}</b><br>
                        O: ${{d.o}}<br>H: ${{d.h}}<br>L: ${{d.l}}<br>C: ${{d.c}}
                    `;
                }} else {{
                    tooltip.style.display = 'none';
                }}
            }});

            // التقريب (Zoom) بالعجلة
            crossCanvas.addEventListener('wheel', e => {{
                e.preventDefault();
                const zoomSpeed = 0.1;
                if (e.deltaY < 0) {{
                    scaleX *= (1 + zoomSpeed); // Zoom In
                }} else {{
                    scaleX *= (1 - zoomSpeed); // Zoom Out
                }}
                scaleX = Math.max(2, Math.min(scaleX, 100)); // حدود الزوم
                draw();
            }});

            // التشغيل الأولي
            resize();
        </script>
    </body>
    </html>
    """
    
    components.html(html_code, height=720)

else:
    st.info("👋 اختر السهم من القائمة الجانبية.")
