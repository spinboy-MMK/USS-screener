import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import io
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import mplfinance as mpf
from ta.trend import SMAIndicator, MACD, CCIIndicator, ADXIndicator
from ta.momentum import StochasticOscillator, RSIIndicator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice

# ==========================================
# 0. 環境設定與字型處理
# ==========================================
matplotlib.use('Agg')

# 自動下載並設定中文字型
def get_chinese_font_path():
    font_filename = "NotoSansCJKtc-Regular.otf"
    if not os.path.exists(font_filename):
        url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf"
        try:
            r = requests.get(url)
            with open(font_filename, 'wb') as f:
                f.write(r.content)
        except:
            return None
    return font_filename

font_path = get_chinese_font_path()
if font_path:
    fe = fm.FontEntry(fname=font_path, name='NotoSansCJKtc')
    fm.fontManager.ttflist.insert(0, fe)
    plt.rcParams['font.sans-serif'] = ['NotoSansCJKtc', 'Arial', 'Microsoft JhengHei']
else:
    plt.rcParams['font.sans-serif'] = ['Arial', 'Microsoft JhengHei']

plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="美股全方位決策系統 (Nasdaq Edition)", layout="wide")

if 'scan_results' not in st.session_state:
    st.session_state.scan_results = {}
if 'stock_map' not in st.session_state:
    st.session_state.stock_map = {}

# 全域設定
GLOBAL_CONFIG = {
    "WEBHOOK_URL": "https://discord.com/api/webhooks/1458447744187240678/rzJZmn_XDMBa0fMnZ2CuWkOKRtMVwE5R-o5TYHkodEonIoSggwlXE8kUW0gFxjBnHA__",
    "ATR_MULTIPLIER": 2.0, 
    "RR_RATIO": 2.0
}

# ==========================================
# 1. 核心運算引擎
# ==========================================
class TechnicalScorer:
    @staticmethod
    def calculate(df):
        if len(df) < 60: return 0, "Data Insufficient", {}
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
        bb = BollingerBands(c); atr = AverageTrueRange(h, l, c)
        obv = OnBalanceVolumeIndicator(c, v).on_balance_volume()
        v_ma5 = v.rolling(5).mean()
        
        curr = df.iloc[-1]; prev = df.iloc[-2]
        
        checks = {
            "Price > MA5": curr['Close'] > ma5.iloc[-1],
            "Price > MA20": curr['Close'] > ma20.iloc[-1],
            "Price > MA60": curr['Close'] > ma60.iloc[-1],
            "MA5 > MA10": ma5.iloc[-1] > ma10.iloc[-1],
            "MA20 > MA60": ma20.iloc[-1] > ma60.iloc[-1],
            "MA Divergence": (ma5.iloc[-1] > ma20.iloc[-1]) and (ma5.iloc[-1] - ma20.iloc[-1]) > (ma5.iloc[-2] - ma20.iloc[-2]),
            "KD Golden Cross": k.iloc[-1] > d.iloc[-1] and k.iloc[-2] <= d.iloc[-2],
            "KD > 50": k.iloc[-1] > 50,
            "RSI > 50": rsi.iloc[-1] > 50,
            "RSI Uptrend": rsi.iloc[-1] > rsi.iloc[-2],
            "MACD > 0": macd_diff.iloc[-1] > 0 and macd_diff.iloc[-2] <= 0,
            "MACD Bullish": macd_diff.iloc[-1] > 0,
            "CCI > 0": cci.iloc[-1] > 0,
            "WillR > -50": wr.iloc[-1] > -50,
            "BBand Upper": curr['Close'] > bb.bollinger_mavg().iloc[-1],
            "BBand Expand": bb.bollinger_wband().iloc[-1] > bb.bollinger_wband().iloc[-5],
            "Vol > MA5": curr['Volume'] > v_ma5.iloc[-1],
            "Vol & Price Up": (curr['Volume'] > prev['Volume']) and (curr['Close'] > prev['Close']),
            "OBV Uptrend": obv.iloc[-1] > obv.iloc[-5],
            "Bias < 15%": ((curr['Close'] - ma20.iloc[-1]) / ma20.iloc[-1]) * 100 < 15
        }
        
        score = sum(1 for v in checks.values() if v) * 5
        if score >= 85: suggestion = "Strong Buy 🔥"
        elif score >= 65: suggestion = "Buy 📈"
        elif score >= 45: suggestion = "Hold ⚠️"
        else: suggestion = "Sell ❄️"
        return score, suggestion, checks

# ==========================================
# 2. 爬蟲與輔助功能 (Method A: NASDAQ FTP)
# ==========================================
@st.cache_data(ttl=86400) 
def get_us_symbols():
    """
    [Method A] 從 NASDAQ 官方 FTP 獲取全市場代碼
    """
    try:
        # 下載 NASDAQ Traded 列表
        url = "http://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
        df = pd.read_csv(url, sep='|')
        
        # 篩選條件：排除測試股、ETF (這份清單ETF標記不明顯，主要濾除測試代碼)
        # 這裡過濾掉 Test Issue 為 'Y' 的
        df = df[df['Test Issue'] == 'N']
        
        # 排除 ETF (簡單過濾，若需要更精準需結合其他 API)
        # 這裡我們保留主要代碼，移除包含 $ 或 . 的特殊證券
        mask = df['Symbol'].str.contains(r'[\$\.]', regex=True)
        df = df[~mask]
        
        # 建立名稱對應表 (這份表只有 Symbol 和 Name)
        stock_map = dict(zip(df['Symbol'], df['Security Name']))
        st.session_state.stock_map = stock_map
        
        symbols = df['Symbol'].tolist()
        return sorted(list(set(symbols)))
        
    except Exception as e:
        # 備用方案：S&P 500
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            df = pd.read_html(url)[0]
            symbols = df['Symbol'].tolist()
            for idx, row in df.iterrows():
                st.session_state.stock_map[row['Symbol']] = row['Security']
            return symbols
        except:
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA']

def get_stock_name(symbol):
    return st.session_state.stock_map.get(symbol, symbol)

def send_discord_webhook(url, strategy, data):
    try:
        requests.post(url, json={"content": f"📢 **{strategy} US Stock Report**"})
        chunk_size = 10
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i+chunk_size]
            
            fig, ax = plt.subplots(figsize=(14, len(chunk) * 0.6 + 1))
            ax.axis('tight'); ax.axis('off')
            
            table_data = []
            for item in chunk:
                s_name = item.get('名稱', item['代碼'])
                if len(s_name) > 20: s_name = s_name[:20] + "..."
                
                row = [
                    item['代碼'], s_name,
                    f"{item['現價']:.2f}", f"{item['買點']:.2f}", 
                    f"{item['停利']:.2f}", f"{item['停損']:.2f}"
                ]
                table_data.append(row)
            
            col_labels = ["Symbol", "Name", "Price", "Buy", "TP", "SL"]
            col_colors = ["#ffebcd"] * 6
            
            table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center', colColours=col_colors)
            table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1, 1.8)
            
            for (row, col), cell in table.get_celld().items():
                if row == 0:
                    cell.set_facecolor('#444444')
                    cell.set_text_props(color='white', weight='bold')
                else:
                    cell.set_facecolor('#f9f9f9' if row % 2 == 0 else '#e0e0e0')
                    text_color = 'black'
                    if col == 4: cell.set_text_props(color='#d62728', weight='bold')
                    if col == 5: cell.set_text_props(color='#2ca02c', weight='bold')
                    else: cell.set_text_props(color='black')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0); plt.close(fig)
            requests.post(url, files={"file": ("report.png", buf)})
            
        st.success("Report Sent to Discord")
    except Exception as e:
        st.error(f"Failed to send: {e}")

# ==========================================
# 3. 側邊欄
# ==========================================
st.sidebar.title("⚙️ Settings (US)")
webhook_url = st.sidebar.text_input("Webhook URL", value=GLOBAL_CONFIG["WEBHOOK_URL"])
atr_mul = st.sidebar.number_input("ATR Multiplier", 2.0, step=0.1)
rr_ratio = st.sidebar.number_input("R/R Ratio", 2.0, step=0.1)

symbols_list = get_us_symbols()
st.sidebar.success(f"📊 Market Symbols: **{len(symbols_list)}**")

st.title("🚀 US Stock Screener (Nasdaq Edition)")

# ==========================================
# 4. 策略邏輯
# ==========================================
def run_scan(strategy_key, params):
    progress = st.progress(0, text="Initializing...")
    symbols = get_us_symbols()
    results = []
    batch_size = 50
    total = len(symbols)
    
    for i in range(0, total, batch_size):
        batch = symbols[i:i+batch_size]
        progress.progress(min((i+batch_size)/total, 1.0), text=f"Scanning... {i}/{total}")
        
        try:
            data = yf.download(batch, period="2y", group_by='ticker', progress=False, threads=True)
            for code in batch:
                try:
                    if len(batch) > 1: df = data[code].dropna()
                    else: df = data.dropna()
                    
                    if len(df) < 250: continue
                    
                    c = df['Close']; o = df['Open']; h = df['High']; l = df['Low']; v = df['Volume']
                    curr = float(c.iloc[-1])
                    vma5 = v.rolling(5).mean().iloc[-1]
                    
                    if vma5 < params['min_vol']: continue
                    
                    match = False
                    buy_point = curr
                    
                    if strategy_key == 'VCP':
                        change = (curr - float(c.iloc[-2]))/float(c.iloc[-2])*100
                        if change < params['change']: continue
                        if params['red'] and curr <= float(o.iloc[-1]): continue
                        
                        ma20 = c.rolling(20).mean().iloc[-1]
                        if ((curr - ma20)/ma20)*100 > params['bias']: continue
                        if float(v.iloc[-1])/vma5 < params['vol_ratio']: continue
                        if params['v5'] and curr < c.rolling(5).mean().iloc[-1]: continue
                        if params['v20'] and curr < ma20: continue
                        
                        atr = AverageTrueRange(h, l, c).average_true_range()
                        if (atr.iloc[-1]/atr.tail(20).mean()) > params['vcp']: continue
                        
                        k_val = StochasticOscillator(h, l, c).stoch().iloc[-1]
                        if k_val > params['k']: continue
                        match = True; buy_point = ma20

                    elif strategy_key == 'SMA1':
                        ma = c.rolling(params['ma']).mean().iloc[-1]
                        if curr > ma:
                            if params['red'] and curr <= float(o.iloc[-1]): pass
                            else: match = True; buy_point = ma

                    elif strategy_key == 'SMA3':
                        s = c.rolling(params['s']).mean().iloc[-1]
                        m = c.rolling(params['m']).mean().iloc[-1]
                        long_ma = c.rolling(params['l']).mean().iloc[-1]
                        if s > m > long_ma and curr > s:
                            if params['red'] and curr <= float(o.iloc[-1]): pass
                            else: match = True; buy_point = s

                    elif strategy_key in ['KD50', 'KD20', 'KD20MA']:
                        stoch = StochasticOscillator(h, l, c, window=params['n'], smooth_window=3)
                        k = stoch.stoch().iloc[-1]; d = stoch.stoch_signal().iloc[-1]
                        pk = stoch.stoch().iloc[-2]; pd_ = stoch.stoch_signal().iloc[-2]
                        cond = (k < params['k'] or pk < params['k']) and k > d and pk <= pd_
                        
                        if strategy_key == 'KD20MA':
                            ma = c.rolling(params['ma']).mean().iloc[-1]
                            cond = cond and curr > ma
                            buy_point = ma
                        
                        if cond:
                            if params['red'] and curr <= float(o.iloc[-1]): pass
                            else: match = True

                    if match:
                        score, suggestion, checks = TechnicalScorer.calculate(df)
                        atr_val = AverageTrueRange(h, l, c).average_true_range().iloc[-1]
                        local_atr = params.get('atr_mul', atr_mul)
                        sl = buy_point - (atr_val * local_atr)
                        tp = buy_point + ((buy_point - sl) * rr_ratio)
                        
                        stock_name = get_stock_name(code)
                        
                        results.append({
                            "代碼": code, 
                            "名稱": stock_name, 
                            "現價": curr, "成交量": int(v.iloc[-1]),
                            "評分": score, "建議": suggestion,
                            "買點": buy_point, "停損": sl, "停利": tp,
                            "歷史數據": df[-250:],
                            "checks": checks
                        })
                except: continue
        except: continue
    
    progress.empty()
    st.session_state.scan_results[strategy_key] = results
    if not results: st.warning("No stocks found.")

# ==========================================
# 5. 介面
# ==========================================
tabs = st.tabs(['🌊 VCP', '📈 SMA1', '🚀 SMA3', '📊 KD50', '⚡ KD20', '💎 KD20MA'])

def render_ui(idx, key, name):
    with tabs[idx]:
        st.subheader(name)
        with st.form(f"f_{key}"):
            c1, c2, c3, c4 = st.columns(4)
            min_vol = c1.number_input("Min Vol (Shares)", 500000, step=100000)
            is_red = c2.checkbox("Green Candle (Up)", True)
            params = {'min_vol': min_vol, 'red': is_red}
            
            if key == 'VCP':
                v_chg = c3.number_input("Change >%", 2.0)
                v_bias = c4.number_input("Bias <%", 10.0)
                v_vr = st.number_input("Vol Ratio >", 1.5)
                v_k = st.number_input("KD <", 80)
                v_vcp = st.number_input("VCP <", 1.3)
                v_atr = st.number_input("ATR Mul", 2.0)
                v_v5 = st.checkbox("Above MA5", True)
                v_v20 = st.checkbox("Above MA20", True)
                params.update({'change':v_chg, 'bias':v_bias, 'vol_ratio':v_vr, 'k':v_k, 'vcp':v_vcp, 'atr_mul':v_atr, 'v5':v_v5, 'v20':v_v20})
            elif key == 'SMA1':
                p_ma = c3.number_input("MA Period", 20)
                params['ma'] = p_ma
            elif key == 'SMA3':
                p_s = c3.number_input("Short MA", 10)
                p_m = c4.number_input("Mid MA", 20)
                p_l = st.number_input("Long MA", 50)
                params.update({'s':p_s, 'm':p_m, 'l':p_l})
            elif key in ['KD50', 'KD20']:
                def_n = 50 if key=='KD50' else 20
                p_n = c3.number_input("N", def_n)
                p_k = c4.number_input("K <", 20)
                params.update({'n':p_n, 'k':p_k})
            elif key == 'KD20MA':
                p_n = c3.number_input("N", 20)
                p_k = c4.number_input("K <", 50)
                p_ma = st.number_input("Above MA", 20)
                params.update({'n':p_n, 'k':p_k, 'ma':p_ma})

            if st.form_submit_button("Start Scan"):
                if key in st.session_state.scan_results:
                    del st.session_state.scan_results[key]
                run_scan(key, params)

        if key in st.session_state.scan_results:
            data = st.session_state.scan_results[key]
            if data:
                st.markdown("---")
                b1, b2 = st.columns(2)
                if b1.button("📢 Send to Discord", key=f"btn_dis_{key}"):
                    send_discord_webhook(webhook_url, key, data)
                
                df_data = []
                for item in data:
                    row = item.copy()
                    if '歷史數據' in row: del row['歷史數據']
                    if 'checks' in row: del row['checks']
                    df_data.append(row)
                df_exp = pd.DataFrame(df_data)
                
                csv = df_exp.to_csv(index=False).encode('utf-8-sig')
                b2.download_button("📥 Download Excel", csv, f"{key}_US.csv", "text/csv")

                for item in data:
                    stock_name = item.get('名稱', item['代碼'])
                    with st.expander(f"{item['代碼']} {stock_name} | ${item['現價']:.2f}", expanded=True):
                        kc, ic = st.columns([3, 1])
                        with kc:
                            sub = item['歷史數據']
                            fig, ax = plt.subplots(figsize=(10, 4))
                            fig.patch.set_facecolor('black')
                            ax.set_facecolor('black')
                            
                            # 美股配色：綠漲紅跌
                            up = sub[sub.Close >= sub.Open]
                            down = sub[sub.Close < sub.Open]
                            ax.bar(up.index, up.Close - up.Open, 0.8, bottom=up.Open, color='#2ca02c')
                            ax.bar(up.index, up.High - up.Close, 0.1, bottom=up.Close, color='#2ca02c')
                            ax.bar(up.index, up.Low - up.Open, 0.1, bottom=up.Open, color='#2ca02c')
                            ax.bar(down.index, down.Close - down.Open, 0.8, bottom=down.Open, color='#d62728')
                            ax.bar(down.index, down.High - down.Open, 0.1, bottom=down.Open, color='#d62728')
                            ax.bar(down.index, down.Low - down.Close, 0.1, bottom=down.Close, color='#d62728')
                            
                            ma5 = sub['Close'].rolling(5).mean()
                            ma20 = sub['Close'].rolling(20).mean()
                            ax.plot(sub.index, ma5, color='cyan', label='MA5')
                            ax.plot(sub.index, ma20, color='orange', label='MA20')
                            
                            ax.text(0.02, 0.95, "MA5: Cyan / MA20: Orange", transform=ax.transAxes, color='white', fontsize=10, fontweight='bold')
                            ax.tick_params(colors='white')
                            st.pyplot(fig)
                        
                        with ic:
                            st.metric("Buy Point", f"{item['買點']:.2f}")
                            st.metric("Take Profit", f"{item['停利']:.2f}")
                            st.metric("Stop Loss", f"{item['停損']:.2f}")
                            st.caption(f"Vol: {int(item['成交量']):,}")
                        
                        with st.expander("📋 Technical Indicators"):
                            chk = item['checks']
                            cols = st.columns(4)
                            for i, (k, v) in enumerate(chk.items()):
                                color = "green" if v else "red"
                                icon = "✅" if v else "❌"
                                cols[i%4].markdown(f":{color}[{icon} {k}]")

render_ui(0, 'VCP', 'VCP (Volatility Contraction)')
render_ui(1, 'SMA1', 'Strategy 1: Single MA')
render_ui(2, 'SMA3', 'Strategy 2: Triple MA')
render_ui(3, 'KD50', 'Strategy 3: Long KD')
render_ui(4, 'KD20', 'Strategy 4: Mid KD')
render_ui(5, 'KD20MA', 'Strategy 5: KD + MA')
