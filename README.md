# CSC14005 — Lab 01: Linear Regression

Ước tính lượng phát thải CO₂ (g/km) của xe hạng nhẹ từ thông số thiết kế bằng hồi quy tuyến tính, dưới lăng kính Frequentist và Bayesian. Đồ án môn Nhập môn học máy (CSC14005), Khoa Công nghệ Thông tin, Trường ĐH Khoa học Tự nhiên, ĐHQG-HCM.

- Yêu cầu đề bài: [docs/requirements.md](docs/requirements.md)
- Quyết định đã có và việc còn mở: [docs/decisions.md](docs/decisions.md)
- Dữ liệu: [data/README.md](data/README.md)
- Hướng dẫn cho AI agent và quy ước code: [AGENTS.md](AGENTS.md)
- Moodboard của nhóm: [Mission LR-01](https://claude.ai/artifact/95knfAAW2ceeRQWnin9HKj)

## Thành viên

GVHD: Thầy Bùi Tiến Lên, Thầy Võ Nhật Tân.

| MSSV | Họ tên | Mô hình phụ trách |
|------|--------|-------------------|
| 24127020 | Nguyễn Thành Đạt | |
| 24127039 | Phan Quang Hiệp | |
| 24127183 | Trần Duy Khang | |
| 24127500 | Nguyễn Đại Phúc | |
| 24127508 | Lê Nho Duy Phước | |

## Cài đặt

```bash
git clone https://github.com/WonDeBuk/CSC14005-Lab01-Linear-Regression.git
cd CSC14005-Lab01-Linear-Regression
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q
```

## Sử dụng

```bash
python -m linreg.train                 # huấn luyện 5 mô hình, ghi models/metrics.csv và hệ số
streamlit run app/streamlit_app.py     # ứng dụng dự đoán CO₂
```

Kiểm định hệ số (Frequentist, sai số chuẩn HC3):

```python
from linreg.data import load_frame
from linreg.inference import fit_ols
print(fit_ols(load_frame(), "m3h_ols").summary())
```

## Mô hình

Định nghĩa trong [`src/linreg/models.py`](src/linreg/models.py). Năm thành viên nên có năm mô hình, mọi mô hình tuyến tính theo tham số. Kết quả dưới đây từ `python -m linreg.train` (chia 854/213, seed 42, CV 10-fold trên tập train).

| Mã | Mô hình | RMSE CV (g/km) | RMSE test (g/km) | R² test |
|----|---------|---------------:|-----------------:|--------:|
| `m8b_interactions` | M8b: M3h + dung tích × (bán tải, van, hybrid) | 22,05 ± 1,68 | 20,35 | 0,898 |
| `m3h_ols` | M3h: OLS trên động cơ, xi-lanh, nhiên liệu, thân xe, hộp số, hybrid | 23,07 ± 1,72 | 20,95 | 0,892 |
| `m3h_ridge` | Ridge trên biến M3h (MAP với prior Gauss) | 23,07 ± 1,72 | 20,97 | 0,892 |
| `m3h_bayes` | Hồi quy Bayesian trên biến M3h (evidence maximization) | 23,08 ± 1,72 | 20,97 | 0,892 |
| `m1_engine` | M1: OLS chỉ với dung tích động cơ (mốc) | 30,71 ± 2,37 | 31,17 | 0,761 |

Số liệu gần với moodboard nhưng chưa khớp từng số; xem [docs/decisions.md](docs/decisions.md).

## Cấu trúc thư mục

```
├── app/streamlit_app.py   # ứng dụng dự đoán CO₂
├── data/raw/              # FuelConsumptionCo2.csv (1.067 xe, 2014)
├── docs/                  # đề bài, quyết định của nhóm
├── figures/               # hình cho báo cáo
├── models/                # mô hình đã huấn luyện (không commit)
├── notebooks/             # EDA và thí nghiệm
├── src/linreg/            # config, features, data, models, inference, evaluate, train
├── tests/                 # kiểm tra pipeline
├── main.tex               # báo cáo (XeLaTeX)
└── references.bib
```

## Làm việc nhóm với AI

Repo được thiết kế để các thành viên dùng AI coding agent (Claude Code, Codex, Copilot, Cursor, …). Agent đọc [`AGENTS.md`](AGENTS.md) (Claude Code đọc qua `CLAUDE.md`), trong đó có bối cảnh đồ án và các quy tắc của đề bài về sử dụng AI. Tóm tắt:

- AI hỗ trợ viết code và soạn nháp; **thành viên nhóm kiểm tra và viết lại** phần phân tích, diễn giải, kết luận.
- Không bịa số liệu hay trích dẫn.
- Mỗi thay đổi qua pull request, có người review.
