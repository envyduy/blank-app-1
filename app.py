import streamlit as st
import torch
from transformers import BertTokenizer, BertModel
import numpy as np

# ==== Load model/tokenizer ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FinLSTM(torch.nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.bert = BertModel.from_pretrained("finbert-mlm")
        self.lstm = torch.nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(768 + hidden_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2)
        )

    def forward(self, input_ids, attention_mask, price_seq):
        with torch.no_grad():
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = bert_out.last_hidden_state[:, 0, :]
        lstm_out, _ = self.lstm(price_seq.unsqueeze(-1))
        lstm_feat = lstm_out[:, -1, :]
        x = torch.cat([pooled_output, lstm_feat], dim=1)
        out = self.fc(x)
        return out[:, 0], out[:, 1]

@st.cache_resource
def load_model():
    model = FinLSTM().to(device)
    model.load_state_dict(torch.load("finlstm_model.pth", map_location=device))
    model.eval()
    tokenizer = BertTokenizer.from_pretrained("finlstm_tokenizer")
    return model, tokenizer

model, tokenizer = load_model()

def convert_to_float(x):
    x = x.replace(',', '').strip()
    if x.endswith('%'): return float(x.replace('%', '')) / 100
    elif x.endswith('K'): return float(x.replace('K', '')) * 1_000
    elif x.endswith('M'): return float(x.replace('M', '')) * 1_000_000
    elif x.endswith('B'): return float(x.replace('B', '')) * 1_000_000_000
    return float(x)

def predict_event_multi(titles, actuals, forecasts, price_open):
    texts = []
    price_seq = [price_open] * 16
    for t, a, f in zip(titles, actuals, forecasts):
        a_val = convert_to_float(a)
        f_val = convert_to_float(f)
        surprise = a_val - f_val
        text = f"{t} | Actual: {a_val:.4f} | Forecast: {f_val:.4f} | Surprise: {surprise:.4f}"
        texts.append(text)

    combined = " [SEP] ".join(texts)
    encoded = tokenizer(combined, padding='max_length', truncation=True, max_length=64, return_tensors='pt')
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    price_tensor = torch.tensor([price_open] + price_seq, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        out_class, out_pip = model(input_ids, attention_mask, price_tensor)
        prob = torch.sigmoid(out_class).item()
        label = "BUY" if prob > 0.5 else "SELL"
        pip_change = out_pip.item()
    return label, prob, pip_change

# ==== Giao diện ====
st.title("📊 Dự đoán BUY/SELL từ nhiều tin tức")

# Khởi tạo session state
if "news_rows" not in st.session_state:
    st.session_state.news_rows = [{"title": "", "actual": "", "forecast": ""}]

st.markdown("### Nhập từng tin tức")

# Tạo danh sách tạm thời
new_rows = []

for idx, row in enumerate(st.session_state.news_rows):
    cols = st.columns([4, 2, 2, 1])
    title = cols[0].text_input(f"Tiêu đề #{idx+1}", row["title"], key=f"title_{idx}")
    actual = cols[1].text_input("Actual", row["actual"], key=f"actual_{idx}")
    forecast = cols[2].text_input("Forecast", row["forecast"], key=f"forecast_{idx}")
    delete = cols[3].button("❌", key=f"delete_{idx}")
    
    if not delete:
        new_rows.append({"title": title, "actual": actual, "forecast": forecast})

# Cập nhật lại state nếu có dòng bị xoá
if len(new_rows) != len(st.session_state.news_rows):
    st.session_state.news_rows = new_rows
else:
    st.session_state.news_rows = new_rows

# Nút thêm tin
if st.button("➕ Thêm tin mới"):
    st.session_state.news_rows.append({"title": "", "actual": "", "forecast": ""})

# Nhập giá
price_open = st.text_input("💰 Giá mở cửa", value="")

# Dự đoán
if st.button("🔮 Dự đoán"):
    titles = [r["title"] for r in st.session_state.news_rows if r["title"].strip()]
    actuals = [r["actual"] for r in st.session_state.news_rows if r["actual"].strip()]
    forecasts = [r["forecast"] for r in st.session_state.news_rows if r["forecast"].strip()]

    if len(titles) != len(actuals) or len(actuals) != len(forecasts):
        st.error("❌ Mỗi dòng phải đủ Title, Actual, Forecast.")
    else:
        try:
            label, prob, pip = predict_event_multi(titles, actuals, forecasts, float(price_open))
            st.success(f"📈 Kết quả: {label} ({prob:.2%})\n📉 Pip thay đổi: {pip:.2f}")
        except Exception as e:
            st.error(f"❌ Lỗi: {str(e)}")
