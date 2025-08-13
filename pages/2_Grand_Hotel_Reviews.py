
import json
import time
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import requests
from typing import Optional
from transformers import pipeline
import matplotlib.pyplot as plt
import re

st.title("🏨 圓山大飯店｜住宿評論情緒分析（Demo）")
st.caption("支援 Google Places API 擷取評論，或上傳 CSV（text[, rating, time]）。僅供教學示範。")

with st.sidebar:
    st.header("資料來源")
    mode = st.selectbox("選擇來源", ["Google Places API", "上傳 CSV"])
    if mode == "Google Places API":
        api_key = st.text_input("Google Places API Key（不會儲存）", type="password")
        query = st.text_input("搜尋關鍵字", value="圓山大飯店 台北")
        max_reviews = st.slider("最多評論數（嘗試抓取）", min_value=10, max_value=200, value=80, step=10)
        fetch_btn = st.button("🔎 搜尋並抓取評論")
    else:
        uploaded = st.file_uploader("上傳 CSV（需有 text 欄）", type=["csv"])
        fetch_btn = st.button("📤 讀取 CSV")

    st.header("分析設定")
    half_life = st.slider("時間半衰期（天）", min_value=1, max_value=30, value=5, step=1)
    pos_th = st.slider("偏好門檻（>）", min_value=0.2, max_value=3.0, value=0.8, step=0.1)
    neg_th = st.slider("偏差門檻（<）", min_value=-3.0, max_value=-0.2, value=-0.8, step=0.1)

def is_chinese(s: str) -> bool:
    return bool(re.search(r"[\\u4e00-\\u9fff]", s or ""))

@st.cache_resource(show_spinner=False)
def get_pipelines(device: int = -1):
    zh_model = "uer/roberta-base-finetuned-dianping-chinese"
    multi_model = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    zh_clf = pipeline("text-classification", model=zh_model, tokenizer=zh_model, device=device)
    multi_clf = pipeline("text-classification", model=multi_model, tokenizer=multi_model, device=device, top_k=None)
    return zh_clf, multi_clf

def classify_text(zh_clf, multi_clf, text: str):
    try:
        if is_chinese(text):
            out = zh_clf(text[:512])[0]
            label = out.get("label", "").lower()  # positive/negative
            score = float(out.get("score", 0.0))
        else:
            out = multi_clf(text[:512])
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

def parse_time(ts: Optional[str]) -> pd.Timestamp:
    if ts is None:
        return pd.NaT
    try:
        return pd.to_datetime(ts, utc=True).tz_convert(tz=None)
    except Exception:
        try:
            return pd.to_datetime(ts)
        except Exception:
            return pd.NaT

def time_decay(ts: pd.Timestamp, now: pd.Timestamp, half_life_days: float) -> float:
    if pd.isna(ts):
        return 1.0
    days = max((now - ts).total_seconds() / 86400.0, 0.0)
    lam = np.log(2) / float(half_life_days)
    return float(np.exp(-lam * days))

def places_text_search(api_key: str, text_query: str):
    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.id,places.displayName"
    }
    payload = {"textQuery": text_query}
    resp = requests.post(url, headers=headers, json=payload, timeout=20)
    if resp.status_code != 200:
        raise RuntimeError(f"TextSearch HTTP {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    return data.get("places", [])

def places_details_reviews(api_key: str, place_id: str, max_reviews: int = 100):
    url = f"https://places.googleapis.com/v1/places/{place_id}"
    headers = {
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "id,displayName,reviews"
    }
    resp = requests.get(url, headers=headers, timeout=20)
    if resp.status_code != 200:
        raise RuntimeError(f"Details HTTP {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    reviews = data.get("reviews", [])[:max_reviews]
    rows = []
    for r in reviews:
        text = r.get("originalText", {}).get("text") or r.get("text", "")
        rating = r.get("rating")
        time_str = r.get("publishTime")
        rows.append({"text": text, "rating": rating, "time": time_str})
    return pd.DataFrame(rows)

def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    if "text" not in df.columns:
        raise ValueError("CSV 需包含 text 欄")
    keep = [c for c in ["text","rating","time"] if c in df.columns]
    return df[keep].copy()

if fetch_btn:
    if mode == "Google Places API":
        if not api_key:
            st.error("請輸入 Google Places API Key。")
            st.stop()
        try:
            st.info("搜尋地點…")
            places = places_text_search(api_key, query)
            if not places:
                st.error("找不到符合的地點，請調整關鍵字。")
                st.stop()
            place = places[0]
            place_id = place.get("id")
            name = place.get("displayName", {}).get("text", "Unknown")
            st.success(f"找到地點：{name}（{place_id}）")

            st.info("抓取評論…")
            df = places_details_reviews(api_key, place_id, max_reviews=max_reviews)
            if df.empty:
                st.warning("沒有抓到評論，可能受配額或欄位遮罩限制。請稍後再試，或改用 CSV 上傳。")
                st.stop()
        except Exception as e:
            st.error(f"抓取失敗：{e}")
            st.stop()
    else:
        if not uploaded:
            st.error("請上傳 CSV。")
            st.stop()
        try:
            df = load_csv(uploaded)
            st.success(f"讀取 {len(df)} 筆評論。")
        except Exception as e:
            st.error(f"CSV 讀取錯誤：{e}")
            st.stop()

    with st.spinner("載入情緒模型並分析…（第一次較久）"):
        zh_clf, multi_clf = get_pipelines(device=-1)

    labels, scores, errs = [], [], []
    for _, r in df.iterrows():
        label, score, err = classify_text(zh_clf, multi_clf, str(r.get("text","")))
        labels.append(label); scores.append(score); errs.append(err)
    df["label"] = labels; df["score"] = scores; df["error"] = errs
    df = df[df["label"].notna()].copy()

    df["sent_num"] = df["label"].map(lambda x: 1.0 if x=="positive" else (-1.0 if x=="negative" else 0.0)) * df["score"]
    now = pd.Timestamp.now(tz=None)
    df["t"] = df.get("time", pd.Series([None]*len(df))).apply(parse_time)
    df["w"] = df["t"].apply(lambda ts: time_decay(ts, now, half_life))
    df["weighted_sent"] = df["sent_num"] * df["w"]

    total_score = float(df["weighted_sent"].sum())
    if total_score > pos_th:
        outlook = "Good（偏好）"; badge = "✅"
    elif total_score < neg_th:
        outlook = "Bad（偏差）"; badge = "⚠️"
    else:
        outlook = "Neutral（中性）"; badge = "⏸️"

    c1, c2 = st.columns(2)
    c1.metric("評論數（有效）", len(df))
    c2.metric("加權情緒分數", f"{total_score:.3f}")
    st.metric("總體判斷", f"{badge} {outlook}")

    st.divider()
    st.subheader("每日平均情緒")
    df["date"] = df["t"].dt.date
    daily = df.dropna(subset=["date"]).groupby("date")["sent_num"].mean().reset_index()
    if not daily.empty:
        plt.figure(figsize=(8,4))
        plt.plot(daily["date"], daily["sent_num"], marker="o")
        plt.xticks(rotation=45)
        plt.title("每日平均情緒")
        plt.xlabel("Date"); plt.ylabel("Avg Sentiment (-1~1)")
        plt.grid(True)
        st.pyplot(plt.gcf()); plt.close()
    else:
        st.write("缺少時間資訊，無法繪圖。")

    st.subheader("明細資料")
    show_cols = [c for c in ["t","label","score","sent_num","w","weighted_sent","rating","text"] if c in df.columns]
    st.dataframe(df.sort_values("t", ascending=False)[show_cols], use_container_width=True)

    st.download_button(
        label="下載結果 CSV",
        data=df.sort_values("t", ascending=False)[show_cols].to_csv(index=False).encode("utf-8-sig"),
        file_name="grand_hotel_reviews_sentiment.csv",
        mime="text/csv"
    )

    st.caption("模型：中文 `uer/roberta-base-finetuned-dianping-chinese`；多語 `cardiffnlp/twitter-xlm-roberta-base-sentiment`。此工具僅供教學示範。")
