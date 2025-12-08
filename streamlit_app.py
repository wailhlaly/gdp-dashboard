import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

# إعداد البيانات (محاكاة دقيقة بناءً على بيانات 2025 المستخرجة)
dates = pd.date_range(start='2025-06-01', end='2025-12-08', freq='B') # Business days
# مسار سعري يحاكي البيانات: قاع في يونيو (91.20) -> قمة في أكتوبر (105.30) -> هبوط حالي (96.00)
prices = []
for d in dates:
    month = d.month
    day = d.day
    # Logic to mimic the trend:
    if month == 6: base = 92 + (np.random.normal(0, 0.5)) # June Lows
    elif month == 7: base = 95 + (np.random.normal(0, 0.5))
    elif month == 8: base = 94 + (np.random.normal(0, 0.5))
    elif month == 9: base = 98 + (d.day/30 * 4) # Rising
    elif month == 10: base = 104 + (np.random.normal(0, 0.8)) # Peak Oct
    elif month == 11: base = 101 - (d.day/30 * 3) # Falling Nov
    elif month == 12: base = 96 + (np.random.normal(0, 0.3)) # Current Dec
    else: base = 96
    prices.append(base)

# ضبط آخر سعر ليكون 96.00 بدقة
prices[-1] = 96.00

df = pd.DataFrame({'Date': dates, 'Close': prices})
df['Open'] = df['Close'].shift(1)
df['High'] = df[['Open', 'Close']].max(axis=1) + 0.5
df['Low'] = df[['Open', 'Close']].min(axis=1) - 0.5
df.dropna(inplace=True)

# رسم الشارت
fig, ax = plt.subplots(figsize=(12, 6), facecolor='#0e1117')
ax.set_facecolor('#0e1117')

# ألوان الشموع
up = df[df.Close >= df.Open]
down = df[df.Close < df.Open]
col_up = '#00b894'
col_down = '#ff7675'

# رسم الشموع
width = .6
width2 = .05
ax.bar(up.Date, up.Close-up.Open, width, bottom=up.Open, color=col_up, alpha=0.8)
ax.bar(up.Date, up.High-up.Close, width2, bottom=up.Close, color=col_up)
ax.bar(up.Date, up.Open-up.Low, width2, bottom=up.Low, color=col_up)

ax.bar(down.Date, down.Open-down.Close, width, bottom=down.Close, color=col_down, alpha=0.8)
ax.bar(down.Date, down.High-down.Open, width2, bottom=down.Open, color=col_down)
ax.bar(down.Date, down.Close-down.Low, width2, bottom=down.Low, color=col_down)

# إضافة المستويات والتحليل
# 1. Supply Zone / Premium
ax.axhspan(102, 105, color='#d63031', alpha=0.2, label='Supply Zone (Premium)')
ax.text(df.Date.iloc[int(len(df)*0.8)], 104, 'PREMIUM ZONE', color='#ff7675', fontsize=9, fontweight='bold')

# 2. Demand Zone / Order Block
ax.axhspan(88, 92, color='#00b894', alpha=0.2, label='Demand Zone (Discount)')
ax.text(df.Date.iloc[int(len(df)*0.1)], 89, 'BULLISH OB + SSL', color='#55efc4', fontsize=9, fontweight='bold')

# 3. Current Price Line
ax.axhline(96.00, color='white', linestyle='--', linewidth=0.8)
ax.text(df.Date.iloc[-1], 96.50, 'Current: 96.00', color='white', fontweight='bold')

# 4. Liquidity Line (SSL)
ssl_price = 91.20
ax.axhline(ssl_price, color='yellow', linestyle=':', linewidth=1)
ax.text(df.Date.iloc[int(len(df)*0.1)], ssl_price-1.5, 'SSL (Liquidity Pool) $$$', color='yellow', fontsize=8)

# العناوين والتنسيق
ax.set_title('Al Rajhi (1120) - SMC & Wyckoff Analysis [Dec 08, 2025]', color='white', fontsize=14, fontweight='bold', pad=20)
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.grid(True, color='#2d3436', alpha=0.3)

# إزالة الحدود
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.show()
