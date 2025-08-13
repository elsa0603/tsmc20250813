
import re
import time
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from transformers import pipeline
import matplotlib.pyplot as plt

st.title("📈 TSMC / 台積電｜Yahoo 財經新聞情緒（Demo）")
st.caption("教學示範：抓取 yfinance 聚合新聞，做中英混合情緒分析，輸出未來一週 Good/Bad/Neutral 傾向。非投資建議。")

with st.sidebar:
    st.header("設定")
    tickers = st.text_input("Ticker（逗號分隔）", value="2330.TW, TSM")
    lookback_days = st.slider("抓取最近幾天的新聞", min_value=3, max_value=21, value=10, step=1)
    week_days = st.slider("未來傾向評估視窗（天）", min_value=3, max_value=14, value=7, step=1)
    half_life = st.slider("時間半衰期（天）", min_value=1, max_value=7, value=3, step=1)
    pos_th = st.slider("偏多門檻（>）", min_value=0.2, max_value=3.0, value=1.0, step=0.1)
    neg_th = st.slider("偏空門檻（<）", min_value=-3.0, max_value=-0.2, value=-1.0, step=0.1)
    run = st.button("🚀 取得新聞並分析")

def is_chinese(s: str) -> bool:
    return bool(re.search(r"[\\u4e00-\\u9fff]", s or ""))

@st.cache_resource(show_spinner=False)
def get_pipelines(device: int = -1):
    zh_model = "uer/roberta-base-finetuned-jd-binary-chinese"
    en_model = "ProsusAI/finbert"
    zh_clf = pipeline("text-classification", model=zh_model, tokenizer=zh_model, device=device)
    en_clf = pipeline("text-classification", model=en_model, tokenizer=en_model, device=device, top_k=None)
    return zh_clf, en_clf

def classify_record(zh_clf, en_clf, title: str, summary: str):
    text = (title or "") + " " + (summary or "")
    try:
        if is_chinese(text):
            out = zh_clf(text[:512])[0]
            label = out.get("label", "").lower()  # positive / negative
            score = float(out.get("score", 0.0))
        else:
            out = en_clf(text[:512])
            if isinstance(out, list):
                best = sorted(out, key=lambda x: -float(x["score"]))[0]
            else:
                best = out
            label = str(best.get("label", "")).lower()  # positive/negative/neutral
            score = float(best.get("score", 0.0))
        return label, score, None
    except Exception as e:
        return None, None, str(e)

def sent_to_num(label: str) -> float:
    if label == "positive":
        return 1.0
    if label == "negative":
        return -1.0
    return 0.0

def to_dt(ts):
    import pandas as _pd
    import datetime as _dt
    try:
        return _dt.datetime.utcfromtimestamp(int(ts)).replace(tzinfo=_dt.timezone.utc).astimezone()
    except Exception:
        return _pd.NaT

def time_decay(ts, now, half_life_days: float = 3.0):
    days = max((now - ts).total_seconds() / 86400.0, 0.0)
    lam = np.log(2) / float(half_life_days)
    return float(np.exp(-lam * days))

if run:
    st.info("開始抓取新聞…")
    symbols = [t.strip() for t in tickers.split(",") if t.strip()]
    news_raw = []
    for t in symbols:
        try:
            news = yf.Ticker(t).news or []
            news_raw.extend(news)
            time.sleep(0.2)
        except Exception as e:
            st.warning(f"[警告] 無法抓取 {t} 新聞：{e}")

    if not news_raw:
        st.error("抓不到任何新聞，請稍後再試或更換 Ticker。")
        st.stop()

    df = pd.DataFrame(news_raw).drop_duplicates(subset=["uuid"]).reset_index(drop=True)
    df["published"] = df["providerPublishTime"].apply(to_dt)
    df["title"] = df["title"].astype(str)
    df["link"] = df["link"].astype(str)
    df["publisher"] = df.get("publisher", pd.Series([""] * len(df))).astype(str)
    df["summary"] = df.get("summary", pd.Series([""] * len(df))).astype(str)

    cut = dt.datetime.now().astimezone() - dt.timedelta(days=int(lookback_days))
    df = df[df["published"] >= cut].copy()

    kw = re.compile(r"(台積電|台积电|TSMC|2330)", re.IGNORECASE)
    df = df[df["title"].str.contains(kw) | df["summary"].str.contains(kw)].copy()

    st.write(f"共擷取到 **{len(df)}** 則符合條件的新聞。")
    if len(df) == 0: st.stop()

    with st.spinner("載入情緒模型…（第一次較久）"):
        zh_clf, en_clf = get_pipelines(device=-1)

    labels, scores, errs = [], [], []
    for _, r in df.iterrows():
        label, score, err = classify_record(zh_clf, en_clf, r["title"], r["summary"])
        labels.append(label); scores.append(score); errs.append(err)
    df["label"] = labels; df["score"] = scores; df["error"] = errs
    df = df[df["label"].notna()].copy()

    df["sent_num"] = df["label"].map(sent_to_num) * df["score"]

    now = dt.datetime.now().astimezone()
    df["w"] = df["published"].apply(lambda ts: time_decay(ts, now, half_life))
    df["weighted_sent"] = df["sent_num"] * df["w"]

    cutN = now - dt.timedelta(days=int(week_days))
    dfN = df[df["published"] >= cutN].copy()

    overall_score = float(dfN["weighted_sent"].sum()) if len(dfN) else 0.0
    if overall_score > pos_th:
        outlook = "Good（偏多）"; badge = "✅"
    elif overall_score < neg_th:
        outlook = "Bad（偏空）"; badge = "⚠️"
    else:
        outlook = "Neutral（中性）"; badge = "⏸️"

    c1, c2, c3 = st.columns(3)
    c1.metric("近一週樣本數", len(dfN))
    c2.metric("加權情緒分數", f"{overall_score:.3f}")
    c3.metric("一週傾向", f"{badge} {outlook}")

    st.divider()
    st.subheader("每日平均情緒（近一週）")
    if len(dfN):
        daily = dfN.copy()
        daily["date"] = daily["published"].dt.date
        agg = daily.groupby("date")["sent_num"].mean().reset_index()

        plt.figure(figsize=(8,4))
        plt.plot(agg["date"], agg["sent_num"], marker="o")
        plt.xticks(rotation=45)
        plt.title("每日平均情緒")
        plt.xlabel("Date")
        plt.ylabel("Avg Sentiment (-1~1)")
        plt.grid(True)
        st.pyplot(plt.gcf()); plt.close()
    else:
        st.write("近一週沒有可用的新聞樣本。")

    st.divider()
    st.subheader("明細（可排序）")
    show_cols = ["published","publisher","label","score","sent_num","w","weighted_sent","title","link"]
    st.dataframe(dfN.sort_values("published", ascending=False)[show_cols], use_container_width=True)

    st.download_button(
        label="下載結果 CSV",
        data=dfN.sort_values("published", ascending=False)[show_cols].to_csv(index=False).encode("utf-8-sig"),
        file_name="tsmc_news_sentiment.csv",
        mime="text/csv"
    )

    st.caption("模型：中文 `uer/roberta-base-finetuned-jd-binary-chinese`；英文 `ProsusAI/finbert`。此工具僅供教學示範，非投資建議。")
