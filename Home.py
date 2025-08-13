
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
1. 請確認你是從**專案根目錄**啟動：`streamlit run Home.py`（根目錄要有 `pages/` 資料夾）。  
2. 若左側沒有出現 Pages 導覽，可能是資料夾層級不正確，或 Streamlit 版本過舊。  
3. 你也可以用下方的**直接連結**切換分頁（需要 Streamlit ≥ 1.22）。
""")

# 額外保險：直接提供分頁連結（即使側邊的 Pages 導覽未顯示，一樣能切換）
st.markdown("### 直接連結")
col1, col2 = st.columns(2)
with col1:
    st.page_link("pages/1_TSMC_News_Sentiment.py", label="📈 TSMC / 台積電｜新聞情緒", icon="📈")
with col2:
    st.page_link("pages/2_Grand_Hotel_Reviews.py", label="🏨 圓山大飯店｜住宿評論情緒", icon="🏨")

st.info("若仍看不到分頁導覽，請確認資料夾名稱為 **pages/**（小寫），且檔案副檔名為 .py。")
