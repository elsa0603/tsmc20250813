
import streamlit as st

st.set_page_config(page_title="News & Reviews Sentiment Lab", page_icon="🧪", layout="wide")

st.title("🧪 News & Reviews Sentiment Lab")
st.write("""
歡迎使用多分頁的情緒分析實驗室：

- **TSMC / 台積電**：抓取 Yahoo 財經相關新聞，進行情緒分析並輸出近一週傾向（Good/Neutral/Bad）。
- **圓山大飯店**：抓取住宿評論（Google Places API 或上傳 CSV），分析好/壞並產出圖表與彙總。

> 本專案僅供教學示範，**不構成投資建議或評價保證**。
""")

st.markdown("---")
st.subheader("如何使用")
st.markdown("""
1. 左側的 **Pages** 切換分頁。  
2. 依頁面提示設定參數（Ticker、抓取天數、API Key 或上傳 CSV 等）。  
3. 直接在雲端（Streamlit Cloud）或本機執行。

建議環境：Python 3.10+，並啟用雲端 GPU（若要快速載入模型）。
""")

st.info("請切換到左側『Pages』的分頁開始使用。")
