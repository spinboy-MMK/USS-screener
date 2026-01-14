import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import io
import matplotlib
import matplotlib.pyplot as plt
from ta.trend import SMAIndicator, MACD, CCIIndicator
from ta.momentum import StochasticOscillator, RSIIndicator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

# 設定 Matplotlib 後端與字型 (解決中文顯示)
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial'] # 嘗試多種字型以相容Linux環境
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 頁面設定與全域變數
# ==========================================
st.set_page_config(page_title="全方位台股策略雷達", layout="wide")

# 模擬 Session State 來儲存掃描結果
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = {}

stock_name_map = {} # 實際部署時建議連接外部資料庫或 API 獲取名稱

# ==========================================
# 2. 核心運算引擎 (完整保留邏輯)
# ==========================================
class TechnicalScorer:
    @staticmethod
    def calculate(df):
        if len(df) < 60: return 0, "資料不足", {}
        c = df['Close']; h = df['High']; l = df['Low']; v = df['Volume']
        
        ma5 = SMAIndicator(c, 5).sma_indicator()
        ma10 = SMAIndicator(c, 10).sma_indicator()
        ma20 = SMAIndicator(c, 20).sma_indicator()
        ma60 = SMAIndicator(c, 60).sma_indicator()
        
        k = StochasticOscillator(h, l, c).stoch()
        d = StochasticOscillator(h, l, c).stoch_signal()
        rsi = RSIIndicator(c).rsi()
        macd = MACD(c); macd_diff = macd.macd_diff()
        cci = CCIIndicator(h, l, c, window=20).cci()
        wr = WilliamsRIndicator(h, l, c).williams_r()
        bb = BollingerBands(c)
        obv = OnBalanceVolumeIndicator(c, v).on_balance_volume()
        v_ma5 = v.rolling(5).mean()
        
        curr = df.iloc[-1]; prev = df.iloc[-2]
        
        checks = {
            "股價站上MA5": curr['Close'] > ma5.iloc[-1],
            "股價站上MA20": curr['Close'] > ma20.iloc[-1],
            "股價站上MA60": curr['Close'] > ma60.iloc[-1],
            "短均多頭 (5>10)": ma5.iloc[-1] > ma10.iloc[-1],
            "中均多頭 (20>60)": ma20.iloc[-1] > ma60.iloc[-1],
            "均線發散向上": (ma5.iloc[-1] > ma20.iloc[-1]) and (ma5.iloc[-1] - ma20.iloc[-1]) > (ma5.iloc[-2] - ma20.iloc[-2]),
            "KD黃金交叉": k.iloc[-1] > d.iloc[-1] and k.iloc[-2] <= d.iloc[-2],
            "KD強勢區 (>50)": k.iloc[-1] > 50,
            "RSI強勢 (>50)": rsi.iloc[-1] > 50,
            "RSI趨勢向上": rsi.iloc[-1] > rsi.iloc[-2],
            "MACD柱狀體翻紅": macd_diff.iloc[-1] > 0 and macd_diff.iloc[-2] <= 0,
            "MACD多方控盤": macd_diff.iloc[-1] > 0,
            "CCI轉強 (>0)": cci.iloc[-1] > 0,
            "威廉指標強勢 (>-50)": wr.iloc[-1] > -50,
            "布林中軌之上": curr['Close'] > bb.bollinger_mavg().iloc[-1],
            "布林開口擴張": bb.bollinger_wband().iloc[-1] > bb.bollinger_wband().iloc[-5],
            "成交量 > 5日均量": curr['Volume'] > v_ma5.iloc[-1],
            "量增價漲": (curr['Volume'] > prev['Volume']) and (curr['Close'] > prev['Close']),
            "OBV趨勢向上": obv.iloc[-1] > obv.iloc[-5],
            "乖離率適中 (<15%)": ((curr['Close'] - ma20.iloc[-1]) / ma20.iloc[-1]) * 100 < 15
        }
        
        score = sum(1 for v in checks.values() if v) * 5
        if score >= 85: suggestion = "強力買進 🔥"
        elif score >= 65: suggestion = "偏多操作 📈"
        elif score >= 45: suggestion = "觀望/整理 ⚠️"
        else: suggestion = "偏空/賣出 ❄️"
        return score, suggestion, checks

# ==========================================
# 3. 輔助功能
# ==========================================
def get_tw_symbols():
    # 這裡簡化為固定清單範例，實際部署建議連接 API 或爬蟲
    # 為了演示，我們放入幾檔熱門股
    return ["2330.TW", "2317.TW", "2454.TW", "2603.TW", "2609.TW", "2615.TW", "3008.TW"]

def send_discord_webhook(url, strategy, data, atr_mul, rr_ratio):
    try:
        # 1. 標題
        requests.post(url, json={"content": f"📢 **{strategy} 精選戰報 (圖片版)**"})
        
        chunk_size = 8
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i+chunk_size]
            
            fig, ax = plt.subplots(figsize=(10, len(chunk) * 0.8 + 1))
            ax.axis('tight'); ax.axis('off')
            
            table_data = []
            for item in chunk:
                row = [
                    item['代碼'], 
                    f"{item['現價']:.2f}",
                    f"{item['建議買點']:.2f}",
                    f"{item['停利']:.2f}",
                    f"{item['停損']:.2f}"
                ]
                table_data.append(row)
            
            col_labels = ["代碼", "現價", "買點", "停利", "停損"]
            col_colors = ["#ffebcd"] * 5
            
            table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center', colColours=col_colors)
            table.auto_set_font_size(False); table.set_fontsize(12); table.scale(1, 1.8)
            
            for (row, col), cell in table.get_celld().items():
                if row == 0:
                    cell.set_facecolor('#444444'); cell.set_text_props(color='white', weight='bold')
                else:
                    cell.set_facecolor('#f9f9f9' if row % 2 == 0 else '#e0e0e0')
                    if col == 3: cell.set_text_props(color='red')
                    if col == 4: cell.set_text_props(color='green')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0); plt.close(fig)
            requests.post(url, files={"file": ("report.png", buf)})
            
        st.success("圖片戰報已發送至 Discord")
    except Exception as e:
        st.error(f"發送失敗: {e}")

# ==========================================
# 4. Streamlit 介面佈局
# ==========================================
# 側邊欄設定
st.sidebar.header("⚙️ 全域參數設定")
webhook_url = st.sidebar.text_input("Discord Webhook URL", value="https://discord.com/api/webhooks/...")
atr_multiplier = st.sidebar.number_input("ATR 止損倍數", value=1.5, step=0.1)
rr_ratio = st.sidebar.number_input("盈虧比 (R/R)", value=2.0, step=0.1)

st.title("🚀 台股全方位決策系統 (Web版)")

# 策略分頁
tabs = st.tabs(['🌊 VCP 波段', '📈 策略一: 單均線', '🚀 策略二: 三線多排', '📊 策略三: 長期KD', '⚡ 策略四: 中期KD', '💎 策略五: KD+MA'])

def render_strategy_page(tab_name, strategy_key):
    # 參數輸入區 (模擬原本的 Frame)
    col1, col2, col3, col4 = st.columns(4)
    min_vol = col1.number_input(f"{strategy_key} 最小交易量(張)", value=1000, key=f"vol_{strategy_key}")
    
    # 策略專屬參數 (範例：VCP)
    if strategy_key == 'VCP':
        vcp_change = col2.number_input("漲幅 >%", value=2.0, key="vcp_c")
        vcp_bias = col3.number_input("乖離 <%", value=8.0, key="vcp_b")
        is_red = col4.checkbox("收紅K", value=True, key="vcp_red")
    else:
        is_red = col2.checkbox("收紅K", value=True, key=f"red_{strategy_key}")

    # 執行按鈕
    if st.button(f"開始掃描 ({strategy_key})", key=f"btn_{strategy_key}"):
        with st.spinner('正在掃描市場數據...'):
            symbols = get_tw_symbols()
            results = []
            
            # 簡化的掃描邏輯 (將原本的 Thread 邏輯移至此)
            for code in symbols:
                try:
                    df = yf.download(code, period="1y", progress=False)
                    if len(df) < 120: continue
                    
                    # 單位換算: 張 -> 股
                    if df['Volume'].iloc[-1] < min_vol * 1000: continue
                    
                    # 這裡放入您原本嚴格的策略邏輯 if ...
                    match = True # 演示用，預設通過
                    
                    if match:
                        score, suggestion, checks = TechnicalScorer.calculate(df)
                        atr = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range().iloc[-1]
                        buy_point = df['Close'].iloc[-1]
                        sl = buy_point - (atr * atr_multiplier)
                        tp = buy_point + ((buy_point - sl) * rr_ratio)
                        
                        results.append({
                            "代碼": code, "現價": df['Close'].iloc[-1], "成交量": int(df['Volume'].iloc[-1]),
                            "評分": score, "建議": suggestion, "買點": buy_point, "停損": sl, "停利": tp,
                            "歷史數據": df[-120:] # 用於畫圖
                        })
                except: pass
            
            st.session_state.scan_results[strategy_key] = results
            st.success(f"掃描完成！共找到 {len(results)} 檔")

    # 顯示結果
    if strategy_key in st.session_state.scan_results:
        data = st.session_state.scan_results[strategy_key]
        
        # 功能按鈕區
        c1, c2 = st.columns(2)
        if c1.button("📢 發送 Discord 戰報", key=f"dis_{strategy_key}"):
            send_discord_webhook(webhook_url, strategy_key, data, atr_multiplier, rr_ratio)
        
        # Excel 下載
        if data:
            df_export = pd.DataFrame(data).drop(columns=['歷史數據'])
            csv = df_export.to_csv(index=False).encode('utf-8-sig')
            c2.download_button("📥 下載 Excel (CSV)", csv, f"{strategy_key}_report.csv", "text/csv")

        # 卡片式列表展示
        for item in data:
            with st.container():
                st.markdown(f"### {item['代碼']} | 評分: {item['評分']} ({item['建議']})")
                k_col, info_col = st.columns([3, 1])
                
                with k_col:
                    # 繪製 K 線圖
                    fig, ax = plt.subplots(figsize=(10, 3))
                    sub = item['歷史數據']
                    ax.plot(sub.index, sub['Close'], label='Close', color='white')
                    ax.plot(sub.index, sub['Close'].rolling(5).mean(), label='MA5', color='cyan')
                    ax.plot(sub.index, sub['Close'].rolling(20).mean(), label='MA20', color='orange')
                    ax.set_facecolor('#0e1117')
                    fig.patch.set_facecolor('#0e1117')
                    ax.tick_params(axis='x', colors='white')
                    ax.tick_params(axis='y', colors='white')
                    st.pyplot(fig)
                
                with info_col:
                    st.metric("現價", f"{item['現價']:.2f}")
                    st.metric("建議買點", f"{item['買點']:.2f}")
                    st.metric("停利", f"{item['停利']:.2f}", delta=f"R/R {rr_ratio}")
                    st.metric("停損", f"{item['停損']:.2f}", delta_color="inverse")
                
                st.divider()

# 渲染各分頁
with tabs[0]: render_strategy_page("VCP 波段", "VCP")
with tabs[1]: render_strategy_page("策略一", "SMA1")
# ... 其他分頁依此類推 ...