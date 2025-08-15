import streamlit as st
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import os

# ====== Cấu hình đường dẫn LOCAL tới model/tokenizer ======
# Sửa 2 dòng này theo máy của bạn:
MODEL_DIR = r"d:/Atrix/finbert-mlm"          # thư mục có config.json + weights
TOKENIZER_DIR = r"d:/Atrix/finbert-mlm"       # hoặc r"d:/Atrix/finlstm_tokenizer" nếu bạn tách riêng
FINLSTM_WEIGHTS = r"d:/Atrix/finlstm_model.pth"  # đường dẫn file .pth

# ====== Device ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== Kiểm tra thư mục model/tokenizer ======
def _assert_exist(path, what):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{what} không tồn tại: {path}")

_assert_exist(MODEL_DIR, "Thư mục MODEL_DIR")
_assert_exist(TOKENIZER_DIR, "Thư mục TOKENIZER_DIR")
_assert_exist(FINLSTM_WEIGHTS, "File FINLSTM_WEIGHTS (.pth)")

# ====== Định nghĩa model ======
class FinLSTM(torch.nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        # Quan trọng: dùng đường dẫn LOCAL thay vì tên repo
        self.bert = BertModel.from_pretrained(MODEL_DIR)
        self.lstm = torch.nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(768 + hidden_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2)
        )

    def forward(self, input_ids, attention_mask, price_seq):
        # BERT không cập nhật (đã freeze)
        with torch.no_grad():
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = bert_out.last_hidden_state[:, 0, :]  # CLS

        lstm_out, _ = self.lstm(price_seq.unsqueeze(-1))  # (B, T, 1) -> (B, T, H)
        lstm_feat = lstm_out[:, -1, :]  # lấy last step
        x = torch.cat([pooled_output, lstm_feat], dim=1)
        out = self.fc(x)
        return out[:, 0], out[:, 1]  # (logit_class, pip_reg)

@st.cache_resource
def load_model():
    model = FinLSTM().to(device)
    # load an toàn
    state = torch.load(FINLSTM_WEIGHTS, map_location=device)
    try:
        model.load_state_dict(state, strict=False)
    except Exception as e:
        raise RuntimeError(f"Lỗi load_state_dict: {e}")

    model.eval()

    # Tokenizer từ thư mục local
    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_DIR)
    return model, tokenizer

model, tokenizer = load_model()

def convert_to_float(x: str) -> float:
    x = x.replace(',', '').strip()
    if x.endswith('%'): return float(x[:-1]) / 100.0
    if x.endswith(('K','k')): return float(x[:-1]) * 1_000
    if x.endswith(('M','m')): return float(x[:-1]) * 1_000_000
    if x.endswith(('B','b')): return float(x[:-1]) * 1_000_000_000
    return float(x)

def predict_event_multi(titles, actuals, forecasts, price_open):
    # Chuỗi giá 16 bước (ví dụ giả định giữ nguyên giá)
    price_seq = [price_open] * 16  # độ dài đúng 16
    texts = []

    for t, a, f in zip(titles, actuals, forecasts):
        a_val = convert_to_float(a)
        f_val = convert_to_float(f)
        surprise = a_val - f_val
        text = f"{t} | Actual: {a_val:.4f} | Forecast: {f_val:.4f} | Surprise: {surprise:.4f}"
        texts.append(text)

    combined = " [SEP] ".join(texts)

    encoded = tokenizer(
        combined,
        padding='max_length',
        truncation=True,
        max_length=64,
        return_tensors='pt'
    )

    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    # Quan trọng: KHÔNG cộng thêm price_open nữa để tránh 17 bước
    price_tensor = torch.tensor(price_seq, dtype=torch.float32).unsqueeze(0).to(device)  # (1,16)

    with torch.no_grad():
        out_class, out_pip = model(input_ids, attention_mask, price_tensor)
        prob = torch.sigmoid(out_class).item()
        label = "BUY" if prob > 0.5 else "SELL"
        pip_change = float(out_pip.item())

    return label, prob, pip_change

# ====== UI ======
st.title("📊 Dự đoán BUY/SELL từ nhiều tin tức")

if "news_rows" not in st.session_state:
    st.session_state.news_rows = [{"title": "", "actual": "", "forecast": ""}]

st.markdown("### Nhập từng tin tức")

new_rows = []
for idx, row in enumerate(st.session_state.news_rows):
    cols = st.columns([4, 2, 2, 1])
    title = cols[0].text_input(f"Tiêu đề #{idx+1}", row["title"], key=f"title_{idx}")
    actual = cols[1].text_input("Actual", row["actual"], key=f"actual_{idx}")
    forecast = cols[2].text_input("Forecast", row["forecast"], key=f"forecast_{idx}")
    delete = cols[3].button("❌", key=f"delete_{idx}")
    if not delete:
        new_rows.append({"title": title, "actual": actual, "forecast": forecast})

st.session_state.news_rows = new_rows

if st.button("➕ Thêm tin mới"):
    st.session_state.news_rows.append({"title": "", "actual": "", "forecast": ""})

price_open_str = st.text_input("💰 Giá mở cửa", value="")

if st.button("🔮 Dự đoán"):
    titles = [r["title"] for r in st.session_state.news_rows if r["title"].strip()]
    actuals = [r["actual"] for r in st.session_state.news_rows if r["actual"].strip()]
    forecasts = [r["forecast"] for r in st.session_state.news_rows if r["forecast"].strip()]

    if len(titles) != len(actuals) or len(actuals) != len(forecasts):
        st.error("❌ Mỗi dòng phải đủ Title, Actual, Forecast.")
    else:
        try:
            if price_open_str.strip() == "":
                st.error("❌ Vui lòng nhập Giá mở cửa.")
            else:
                price_open_val = float(price_open_str)
                label, prob, pip = predict_event_multi(titles, actuals, forecasts, price_open_val)
                st.success(f"📈 Kết quả: {label} ({prob:.2%})\n📉 Pip thay đổi: {pip:.2f}")
        except FileNotFoundError as e:
            st.error(f"❌ Thiếu file/thư mục: {e}")
        except ValueError as e:
            st.error(f"❌ Giá trị không hợp lệ (số, %, K/M/B): {e}")
        except Exception as e:
            st.error(f"❌ Lỗi: {str(e)}")
