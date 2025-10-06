import streamlit as st
import pandas as pd
import joblib

# 頁面設定
st.set_page_config(page_title="信用卡詐欺偵測 Demo", page_icon="💳", layout="centered")
st.title("💳 信用卡詐欺偵測 Demo")
st.caption("模型：Logistic Regression（class_weight='balanced'） | 資料：Kaggle/ULB | 僅供教育用途")

# 載入模型（快取避免重覆讀檔）
@st.cache_resource
def load_model():
    return joblib.load("fraud_lr.pkl")

model = load_model()

# 訓練時的特徵欄位（跟 Kaggle 資料集一模一樣，不包含 Class）
EXPECTED_FEATURES = [
    'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
    'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
    'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
]

# 門檻調整滑桿
thr = st.slider("⚠️ 風險門檻 (Threshold)", 0.01, 0.99, 0.5, 0.01,
                help="調低可抓到更多詐欺(Recall)，但誤報多；調高則相反")

# 檔案上傳
uploaded_file = st.file_uploader("📂 上傳交易 CSV 檔案（必須包含 Time, V1-V28, Amount 欄位）", type=["csv"])

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)

    # 欄位檢查
    missing_cols = [col for col in EXPECTED_FEATURES if col not in df_raw.columns]
    if missing_cols:
        st.error(f"❌ 檔案缺少必要欄位：{missing_cols}")
        st.stop()

    # 重新排序欄位 & 丟掉多餘欄位
    df = df_raw[EXPECTED_FEATURES]

    st.success(f"✅ 檔案已成功讀取，筆數：{len(df)}，欄位檢查通過")
    st.write(df.head())

    # 預測
    prob = model.predict_proba(df)[:, 1]
    pred = (prob >= thr).astype(int)

    # 合併結果
    out = df.copy()
    out["fraud_prob"] = prob
    out["prediction"] = pred

    # 統計
    total_fraud = int((pred == 1).sum())
    st.metric(label="疑似詐欺筆數", value=total_fraud)

    # 顯示前20筆
    st.subheader("🔍 預測結果（前 20 筆）")
    st.dataframe(out.head(20))

    # 下載按鈕
    st.download_button(
        label="💾 下載完整預測結果 CSV",
        data=out.to_csv(index=False),
        file_name="predictions.csv",
        mime="text/csv"
    )
else:
    st.info("⬆️ 請上傳交易資料 CSV 檔進行分析")