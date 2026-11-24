# CSC14005 — Lab 01: Linear Regression

Đồ án môn Nhập môn học máy (CSC14005), Trường ĐH Khoa học Tự nhiên, ĐHQG-HCM: chọn một bài toán hồi quy trên dữ liệu thật, xây dựng và so sánh các mô hình Linear Regression, và làm ứng dụng dự đoán.

- Yêu cầu đề bài: [docs/requirements.md](docs/requirements.md)
- Dữ liệu: [data/README.md](data/README.md)
- Hướng dẫn cho AI agent và quy ước code: [AGENTS.md](AGENTS.md)

## Cài đặt

```bash
git clone https://github.com/WonDeBuk/csc14005-lab01-linear-regression.git
cd csc14005-lab01-linear-regression
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q
```

## Sử dụng

1. Tải dataset về `data/raw/dataset.csv` (xem [data/README.md](data/README.md)).
2. Khai báo biến mục tiêu trong `src/linreg/config.py` (`DATASET["target"]`).
3. Huấn luyện tất cả mô hình: `python -m linreg.train` → ghi `models/metrics.csv`, hệ số và file mô hình.
4. Chạy ứng dụng: `streamlit run app/streamlit_app.py`.

## Mô hình

Định nghĩa trong [`src/linreg/models.py`](src/linreg/models.py). Đề bài yêu cầu số mô hình ≥ số thành viên.

| Mã | Mô hình | Mục đích |
|----|---------|----------|
| `ols` | Linear Regression (OLS) | Baseline |
| `ridge` | Ridge (L2, chọn α bằng CV) | Ổn định hệ số khi đặc trưng tương quan |
| `lasso` | Lasso (L1, chọn α bằng CV) | Chọn đặc trưng |
| `elasticnet` | Elastic Net | Kết hợp L1 và L2 |
| `poly2_ridge` | Đa thức bậc 2 + Ridge | Quan hệ phi tuyến, vẫn tuyến tính theo tham số |
| `log_target_ols` | OLS trên log(1 + y) | Biến mục tiêu lệch phải, dương |
| `huber` | Huber Regressor | Giảm ảnh hưởng của ngoại lệ |

## Cấu trúc thư mục

```
├── app/streamlit_app.py   # ứng dụng dự đoán
├── data/                  # dataset (không commit, xem data/README.md)
├── docs/requirements.md   # tóm tắt đề bài
├── figures/               # hình cho báo cáo
├── models/                # mô hình đã huấn luyện (không commit)
├── notebooks/             # EDA và thí nghiệm
├── src/linreg/            # code: config, data, models, evaluate, train
├── tests/                 # smoke test
├── main.tex               # báo cáo (XeLaTeX)
└── references.bib
```

## Làm việc nhóm với AI

Repo được thiết kế để các thành viên dùng AI coding agent (Claude Code, Codex, Copilot, Cursor, …). Agent đọc [`AGENTS.md`](AGENTS.md) (Claude Code đọc qua `CLAUDE.md`), trong đó có các quy tắc của đề bài về sử dụng AI. Tóm tắt:

- AI hỗ trợ viết code và soạn nháp; **thành viên nhóm kiểm tra và viết lại** phần phân tích, diễn giải, kết luận.
- Không bịa số liệu hay trích dẫn.
- Mỗi thay đổi qua pull request, có người review.

## Thành viên

| Họ tên | MSSV | Phụ trách |
|--------|------|-----------|
| | | |
