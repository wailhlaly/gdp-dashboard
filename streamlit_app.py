import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# محاكاة بيانات الراجحي لعام 2024 بناءً على البيانات الحقيقية التقريبية
dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="B")
# إنشاء مسار سعري يحاكي الواقع: هبوط في منتصف السنة، تعافي قوي في الربع الرابع
prices = []
base_price = 87.0
trend = np.linspace(0, 5, len(dates)) # اتجاه عام
noise = np.random.normal(0, 1, len(dates))

# محاكاة الحركة: Q1 تذبذب، Q2/Q3 هبوط (تجميع)، Q4 صعود قوي (تأثير الفائدة)
for i, date in enumerate(dates):
    month = date.month
    val = base_price
    if month < 5: val += np.sin(i/20)*2 + 2 # تذبذب حول 88-90
    elif 5 <= month < 9: val -= (3 + np.sin(i/10)*1.5) # هبوط لمستويات 80-84 (Spring)
    elif month >= 9: val += (i/len(dates))*15 - 5 # تعافي قوي نحو 96
    prices.append(val + noise[i])

df = pd.DataFrame({'Date': dates, 'Close': prices})
df.set_index('Date', inplace=True)

# الرسم البياني
plt.figure(figsize=(12, 7))
plt.style.use('dark_background')

# رسم السعر
plt.plot(df.index, df['Close'], color='#00ffcc', linewidth=1.5, label='Price Action')

# مناطق ICT / SMC
# 1. Sell Side Liquidity (SSL) Sweep - منتصف العام
plt.axhline(y=df['Close'].min(), color='red', linestyle='--', alpha=0.5)
plt.text(df.index[int(len(df)/2)], df['Close'].min()-1, '❌ SSL Swept (Wyckoff Spring)', color='red', fontsize=9)

# 2. Market Structure Shift (MSS) - الربع الرابع
mss_level = 90.0
plt.axhline(y=mss_level, color='yellow', linestyle=':', alpha=0.8)
plt.text(df.index[-60], mss_level+0.5, '⚡ MSS (Bullish Change)', color='yellow', fontsize=9)

# 3. Fair Value Gap (FVG) - منطقة الدخول المتوقعة
plt.fill_between(df.index[-30:], 93.5, 95.0, color='lime', alpha=0.2, label='Bullish FVG (Entry Zone)')

# 4. Draw on Liquidity (Targets)
plt.axhline(y=105, color='white', linestyle='--', alpha=0.6)
plt.text(df.index[-10], 105.5, '🎯 TP1 (Old Highs)', color='white', fontsize=9)

plt.title('Al Rajhi Bank (1120.SE) - Institutional Analysis Map (As of Jan 1, 2025)', fontsize=14, color='white')
plt.ylabel('Price (SAR)')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.1)

# إخفاء المحور الأيمن والأعلى
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
