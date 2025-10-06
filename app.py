import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time

# ===== 頁面設定 =====
st.set_page_config(page_title="信用卡即時風控系統 Demo", page_icon="💳", layout="wide")
st.title("💳 信用卡即時風控系統 Demo")
st.caption("模擬銀行即時信用卡交易風險偵測流程，數據來自 Kaggle/ULB，經匿名化處理，僅供教育用途。")

# ===== 流程說明 =====
with st.expander("📈 系統流程圖說明"):
    st.markdown("""
    1. **刷卡交易流入**（模擬用CSV上傳）
    2. **模型計算風險分數**（輸出詐欺機率 0~1）
    3. **門檻判斷**（Threshold：風控策略即時調整）
    4. **標記高風險交易**（人工審核或暫停交易）
    """)

# ===== 載入模型 =====
@st.cache_resource
def load_model():
    return joblib.load("fraud_lr.pkl")
model = load_model()

# ===== 模型特徵欄位 =====
EXPECTED_FEATURES = [
    'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
    'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
    'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
]

# ===== 欄位解釋表 =====
feature_desc = {'Time': '距第一筆交易的秒數（交易時間差）', 'Amount': '交易金額 (€)'}
for i in range(1, 29):
    feature_desc[f'V{i}'] = f'匿名化特徵 {i}（由原始交易經 PCA 降維）'

# ===== 上傳檔案 =====
uploaded_file = st.file_uploader("📂 上傳交易 CSV 檔案", type=["csv"])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)

    # 欄位檢查
    missing_cols = [col for col in EXPECTED_FEATURES if col not in df_raw.columns]
    if missing_cols:
        st.error(f"❌ 檔案缺少必要欄位: {missing_cols}")
        st.stop()
    df = df_raw[EXPECTED_FEATURES]

    # ===== 顯示欄位解釋 =====
    st.subheader("📝 欄位中文解釋")
    st.table(pd.DataFrame.from_dict(feature_desc, orient='index', columns=["說明"]))

    # ===== 門檻滑桿 =====
    thr = st.slider("⚠️ 風險門檻 (Threshold)", 0.01, 0.99, 0.5, 0.01,
                    help="低門檻 → Recall高但誤報多；高門檻 → 誤報少但漏報多")

    # ===== 加入等待提示 =====
    with st.spinner("⏳ 請稍等，正在分析交易資料..."):
        time.sleep(1.5)  # 模擬延遲

        # 預測
        prob = model.predict_proba(df)[:, 1]
        pred = (prob >= thr).astype(int)
        out = df.copy()
        out["fraud_prob"] = prob
        out["prediction"] = pred

    # ===== 統計卡片 =====
    total_txn = len(out)
    total_fraud = int((pred == 1).sum())
    fraud_ratio = total_fraud / total_txn * 100
    col1, col2, col3 = st.columns(3)
    col1.metric("📊 總交易數", total_txn)
    col2.metric("🚨 疑似詐欺筆數", total_fraud)
    col3.metric("📈 疑似詐欺比例(%)", f"{fraud_ratio:.2f}")

    # ===== 高風險交易表 =====
    st.subheader("🔍 高風險交易（前 20 筆）")
    top20 = out.sort_values(by="fraud_prob", ascending=False).head(20)
    st.dataframe(top20)

    # ===== 單筆交易風險原因分析 =====
    st.subheader("📢 風險原因解釋")
    coeffs = model.named_steps['clf'].coef_[0]  # Logistic Regression 係數
    selected_index = st.selectbox("選擇交易索引編號", top20.index.tolist())

    if selected_index:
        txn = out.loc[selected_index, EXPECTED_FEATURES]
        contrib = coeffs * txn.values  # 特徵貢獻分數
        contrib_df = pd.DataFrame({
            "特徵": EXPECTED_FEATURES,
            "數值": txn.values,
            "影響分數": contrib,
            "影響方向": ["增加風險" if c > 0 else "降低風險" for c in contrib]
        }).sort_values(by="影響分數", key=lambda x: abs(x), ascending=False)

        st.write(f"💡 交易索引 {selected_index} 的風險分數: {out.loc[selected_index,'fraud_prob']:.4f}")
        st.table(contrib_df.head(10))

else:
    st.info("⬆️ 請上傳交易資料 CSV 開始分析（必須包含 Time, V1-V28, Amount 欄位）")