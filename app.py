import streamlit as st
import pandas as pd
import joblib

# ===== 頁面設定 =====
st.set_page_config(page_title="信用卡即時風控系統 Demo", page_icon="💳", layout="wide")

# ===== 顯示標題與系統流程圖 =====
st.title("💳 信用卡即時風控系統 Demo")
st.caption("本系統模擬真實銀行環境中即時信用卡交易風險偵測流程，數據經匿名化處理（Kaggle/ULB），僅作教育用途。")

with st.expander("📈 系統流程圖說明"):
    st.markdown("""
    1. **刷卡交易流入**（模擬由API或批次檔匯入）
    2. **模型計算風險分數**（輸出0~1之間的詐欺機率）
    3. **門檻判斷**（Threshold：風控策略的關鍵調節）
    4. **標記高風險交易**（進入人工審核或自動暫停）
    """)

# ===== 載入模型 =====
@st.cache_resource
def load_model():
    return joblib.load("fraud_lr.pkl")

model = load_model()

# ===== 模型需要的特徵欄位清單 =====
EXPECTED_FEATURES = [
    'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
    'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
    'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
]

# ===== 欄位解說表 =====
feature_desc = {
    'Time': '距離第一筆交易的秒數（觀察交易時間分佈）',
    'Amount': '交易金額 (€)',
}
for i in range(1, 29):
    feature_desc[f'V{i}'] = f'匿名化特徵 {i}（由原始交易經PCA降維）'

# ===== 檔案上傳 =====
uploaded_file = st.file_uploader("📂 上傳交易 CSV 檔案", type=["csv"])

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)

    # ===== 欄位檢查 =====
    missing_cols = [col for col in EXPECTED_FEATURES if col not in df_raw.columns]
    if missing_cols:
        st.error(f"❌ 檔案缺少必要欄位：{missing_cols}")
        st.stop()

    # 丟掉多餘欄位，並排序
    df = df_raw[EXPECTED_FEATURES]

    # ===== 顯示欄位說明表 =====
    st.subheader("📝 欄位中文解說")
    desc_df = pd.DataFrame.from_dict(feature_desc, orient='index', columns=["說明"])
    st.table(desc_df)

    # ===== 門檻滑桿 =====
    thr = st.slider(
        "⚠️ 風險門檻 (Threshold)",
        0.01, 0.99, 0.5, 0.01,
        help="調低可抓到更多可疑交易(Recall↑)，但誤報多；調高誤報少但可能漏掉詐欺"
    )

    # ===== 預測 =====
    prob = model.predict_proba(df)[:, 1]
    pred = (prob >= thr).astype(int)

    out = df.copy()
    out["fraud_prob"] = prob
    out["prediction"] = pred

    # ===== 統計卡片 =====
    total_txn = len(out)
    total_fraud = int((pred == 1).sum())
    fraud_ratio = (total_fraud / total_txn) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("📊 總交易數", total_txn)
    col2.metric("🚨 疑似詐欺筆數", total_fraud)
    col3.metric("📈 疑似詐欺比例 (%)", f"{fraud_ratio:.2f}")

    # ===== 高風險交易表格 =====
    st.subheader("🔍 高風險交易（前 20 筆）")
    st.dataframe(out.sort_values(by="fraud_prob", ascending=False).head(20))

    # ===== 下載完整結果 =====
    st.download_button(
        label="💾 下載完整分析結果 CSV",
        data=out.to_csv(index=False),
        file_name="predictions.csv",
        mime="text/csv"
    )

    # ===== 比賽講解提示 =====
    with st.expander("📢 講解提示"):
        st.markdown("""
        - **欄位解說**：先介紹Time、Amount，再說V1~V28是匿名化特徵。
        - **門檻調整**：展示不同Threshold下抓到的可疑交易數變化，解釋Precision/Recall的取捨。
        - **結果解析**：指著高機率的交易，說明系統會優先讓人工審核。
        - **場景應用**：真實銀行可每日批次匯入交易，或即時接API流，每筆交易立刻計算風險分數。
        """)

else:
    st.info("⬆️ 請上傳交易資料 CSV 開始分析（必須包含 Time, V1-V28, Amount 欄位）")