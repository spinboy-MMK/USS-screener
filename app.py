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

st.set_page_config(page_title="美股全方位決策系統 (Nasdaq版)", layout="wide")

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
        bb = BollingerBands(c); atr = AverageTrueRange(h, l, c)
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
# 2. 爬蟲與輔助功能 (三段式修復版)
# ==========================================
@st.cache_data(ttl=86400) 
def get_us_symbols():
    """
    [修復版] 三段式獲取代碼機制，確保不只有 6 檔
    """
    symbols = []
    
    # ---------------------------------------------------
    # 1. 嘗試從 NASDAQ 官方 FTP 獲取全市場 (~8000檔)
    # ---------------------------------------------------
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        url = "http://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            df = pd.read_csv(io.StringIO(r.text), sep='|')
            # 排除測試股與 ETF
            df = df[(df['Test Issue'] == 'N') & (df['ETF'] == 'N')]
            st.session_state.stock_map = dict(zip(df['Symbol'], df['Security Name']))
            return sorted(df['Symbol'].tolist())
    except:
        pass # 失敗則進入下一階段

    # ---------------------------------------------------
    # 2. 嘗試從 GitHub 獲取 S&P 500 成分股 (~500檔)
    # ---------------------------------------------------
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        df = pd.read_csv(url)
        st.session_state.stock_map = dict(zip(df['Symbol'], df['Name']))
        return sorted(df['Symbol'].tolist())
    except:
        pass

    # ---------------------------------------------------
    # 3. 最後防線：內建 Nasdaq 100 重點成分股 (100檔)
    # ---------------------------------------------------
    # 如果以上都失敗，回傳這 100 檔，絕對比 6 檔好
    nasdaq_100 = [
        'AAPL', 'MSFT', 'AMZN', 'GOOG', 'GOOGL', 'META', 'TSLA', 'NVDA', 'PYPL', 'ADBE', 
        'NFLX', 'PEP', 'CSCO', 'CMCSA', 'INTC', 'AMGN', 'COST', 'TXN', 'AVGO', 'TMUS', 
        'QCOM', 'CHTR', 'SBUX', 'AMD', 'INTU', 'GILD', 'ISRG', 'FISV', 'BKNG', 'MDLZ', 
        'ADP', 'ATVI', 'CSX', 'MU', 'AMAT', 'ILMN', 'ADSK', 'BIIB', 'ADI', 'LRCX', 
        'REGN', 'JD', 'MELI', 'VRTX', 'KHC', 'NXPI', 'WBA', 'MAR', 'ROST', 'BIDU', 
        'MNST', 'LULU', 'KLAC', 'EXC', 'EA', 'AEP', 'DOCU', 'ALGN', 'DXCM', 'IDXX', 
        'WDAY', 'CDNS', 'PAYX', 'SNPS', 'CTSH', 'ORLY', 'MCHP', 'NTES', 'SGEN', 'SPLK', 
        'VRSK', 'XEL', 'PCAR', 'FAST', 'DLTR', 'ANSS', 'XLNX', 'CTAS', 'SWKS', 'VRSN', 
        'CPRT', 'CDW', 'CERN', 'INCY', 'MXIM', 'CHKP', 'TCOM', 'ASML', 'OKTA', 'ZM'
    ]
    return sorted(nasdaq_100)

def get_stock_name(symbol):
    return st.session_state.stock_map.get(symbol, symbol)

def send_discord_webhook(url, strategy, data):
    try:
        requests.post(url, json={"content": f"📢 **{strategy} 美股戰報 (Web版)**"})
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
            
            col_labels = ["代碼", "名稱", "現價", "買點", "停利", "停損"]
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
                    if col == 4: cell.set_text_props(color='#d62728', weight='bold') # 停利紅
                    if col == 5: cell.set_text_props(color='#2ca02c', weight='bold') # 停損綠
                    else: cell.set_text_props(color='black')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0); plt.close(fig)
            requests.post(url, files={"file": ("report.png", buf)})
            
        st.success("戰報發送成功")
    except Exception as e:
        st.error(f"發送失敗: {e}")

# ==========================================
# 3. 側邊欄
# ==========================================
st.sidebar.title("⚙️ 美股全域設定")
webhook_url = st.sidebar.text_input("Discord Webhook 網址", value=GLOBAL_CONFIG["WEBHOOK_URL"])
atr_mul = st.sidebar.number_input("ATR 止損倍數", 2.0, step=0.1)
rr_ratio = st.sidebar.number_input("盈虧比 (R/R)", 2.0, step=0.1)

# 獲取並顯示檔數
symbols_list = get_us_symbols()
# 根據數量判斷是哪種來源
source_hint = "全市場" if len(symbols_list) > 2000 else ("S&P 500" if len(symbols_list) > 400 else "Nasdaq 100")
st.sidebar.success(f"📊 監控範圍: {source_hint} | 共 **{len(symbols_list)}** 檔")

st.title("🚀 美股全方位決策系統 (Nasdaq 旗艦版)")

# ==========================================
# 4. 策略邏輯
# ==========================================
def run_scan(strategy_key, params):
    progress = st.progress(0, text="系統初始化中...")
    symbols = get_us_symbols()
    results = []
    batch_size = 50
    total = len(symbols)
    
    for i in range(0, total, batch_size):
        batch = symbols[i:i+batch_size]
        progress.progress(min((i+batch_size)/total, 1.0), text=f"全市場掃描中... {i}/{total}")
        
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
                    
                    # --- 策略邏輯 ---
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
    if not results: st.warning("⚠️ 掃描完成，未發現符合條件的標的。")

# ==========================================
# 5. 介面 (中文化)
# ==========================================
tabs = st.tabs(['🌊 VCP 波段', '📈 策略一: 單均線', '🚀 策略二: 三線多排', '📊 策略三: 長期KD', '⚡ 策略四: 中期KD', '💎 策略五: KD+MA'])

def render_ui(idx, key, name):
    with tabs[idx]:
        st.subheader(name)
        with st.form(f"f_{key}"):
            c1, c2, c3, c4 = st.columns(4)
            min_vol = c1.number_input("最小交易量 (股)", 500000, step=100000)
            is_red = c2.checkbox("收紅K", True)
            params = {'min_vol': min_vol, 'red': is_red}
            
            if key == 'VCP':
                v_chg = c3.number_input("漲幅 >%", 2.0)
                v_bias = c4.number_input("乖離 <%", 10.0)
                v_vr = st.number_input("量比 >", 1.5)
                v_k = st.number_input("KD <", 80)
                v_vcp = st.number_input("VCP <", 1.3)
                v_atr = st.number_input("ATR 倍數", 2.0)
                v_v5 = st.checkbox("站上5MA", True)
                v_v20 = st.checkbox("站上20MA", True)
                params.update({'change':v_chg, 'bias':v_bias, 'vol_ratio':v_vr, 'k':v_k, 'vcp':v_vcp, 'atr_mul':v_atr, 'v5':v_v5, 'v20':v_v20})
            elif key == 'SMA1':
                p_ma = c3.number_input("MA 週期", 20)
                params['ma'] = p_ma
            elif key == 'SMA3':
                p_s = c3.number_input("短MA", 10)
                p_m = c4.number_input("中MA", 20)
                p_l = st.number_input("長MA", 50)
                params.update({'s':p_s, 'm':p_m, 'l':p_l})
            elif key in ['KD50', 'KD20']:
                def_n = 50 if key=='KD50' else 20
                p_n = c3.number_input("KD 週期", def_n)
                p_k = c4.number_input("K值 <", 20)
                params.update({'n':p_n, 'k':p_k})
            elif key == 'KD20MA':
                p_n = c3.number_input("KD 週期", 20)
                p_k = c4.number_input("K值 <", 50)
                p_ma = st.number_input("站上MA", 20)
                params.update({'n':p_n, 'k':p_k, 'ma':p_ma})

            if st.form_submit_button("開始掃描"):
                if key in st.session_state.scan_results:
                    del st.session_state.scan_results[key]
                run_scan(key, params)

        if key in st.session_state.scan_results:
            data = st.session_state.scan_results[key]
            if data:
                st.markdown("---")
                b1, b2 = st.columns(2)
                if b1.button("📢 發送圖片戰報", key=f"btn_dis_{key}"):
                    send_discord_webhook(webhook_url, key, data)
                
                df_data = []
                for item in data:
                    row = item.copy()
                    if '歷史數據' in row: del row['歷史數據']
                    if 'checks' in row: del row['checks']
                    df_data.append(row)
                df_exp = pd.DataFrame(df_data)
                
                csv = df_exp.to_csv(index=False).encode('utf-8-sig')
                b2.download_button("📥 下載 Excel 報表", csv, f"{key}_US.csv", "text/csv")

                for item in data:
                    stock_name = item.get('名稱', item['代碼'])
                    with st.expander(f"{item['代碼']} {stock_name} | 現價: ${item['現價']:.2f}", expanded=True):
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
                            st.metric("建議買點", f"{item['買點']:.2f}")
                            st.metric("停利價格", f"{item['停利']:.2f}")
                            st.metric("停損價格", f"{item['停損']:.2f}")
                            vol_txt = f"{int(item['成交量']/10000)} 萬股" if item['成交量'] > 10000 else f"{item['成交量']} 股"
                            st.caption(f"成交量: {vol_txt}")
                        
                        with st.expander("📋 20項指標詳細診斷"):
                            chk = item['checks']
                            cols = st.columns(4)
                            for i, (k, v) in enumerate(chk.items()):
                                color = "green" if v else "red"
                                icon = "✅" if v else "❌"
                                cols[i%4].markdown(f":{color}[{icon} {k}]")

render_ui(0, 'VCP', 'VCP 波段')
render_ui(1, 'SMA1', '策略一')
render_ui(2, 'SMA3', '策略二')
render_ui(3, 'KD50', '策略三')
render_ui(4, 'KD20', '策略四')
render_ui(5, 'KD20MA', '策略五')
