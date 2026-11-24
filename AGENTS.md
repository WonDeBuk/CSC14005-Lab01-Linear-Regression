# Hướng dẫn cho AI agent

File này dành cho mọi AI coding agent (Claude Code, Codex, Copilot, Cursor, …) làm việc trong repo. `CLAUDE.md` import file này.

## Bối cảnh

Đồ án **CSC14005 Lab 01 — Linear Regression** (môn Nhập môn học máy). Yêu cầu đầy đủ: [docs/requirements.md](docs/requirements.md). Sản phẩm cần nộp: mã nguồn, dữ liệu (hoặc link), `requirements.txt`, ứng dụng dự đoán, video demo, báo cáo PDF 10–20 trang (`main.tex`).

## Quy tắc bắt buộc (từ đề bài)

1. **AI chỉ hỗ trợ, người quyết định.** Đề bài trừ tới 50% điểm nếu lạm dụng AI hoặc nội dung sai. Agent không tự viết hoàn chỉnh phần phân tích, kết luận hay diễn giải trong báo cáo: chỉ soạn nháp, đánh dấu `% TODO(review)` trong LaTeX để thành viên nhóm đọc lại và viết lại bằng lời của mình.
2. **Diễn giải chỉ dựa trên kết quả mô hình.** Không thêm kiến thức miền bên ngoài (ví dụ "giá nhà ở trung tâm cao hơn vì…") nếu số liệu của mô hình không cho thấy điều đó.
3. **Không bịa số liệu.** Mọi con số trong báo cáo phải lấy từ `models/metrics.csv`, `models/*_coefficients.csv` hoặc output notebook đã chạy. Không bịa trích dẫn: mọi tài liệu tham khảo phải có thật và nằm trong `references.bib`.
4. **Số mô hình ≥ số thành viên.** Mô hình chính là Linear Regression và các biến thể tuyến tính theo tham số (`src/linreg/models.py`).
5. **Repo phải public và link dữ liệu phải truy cập được** (link hỏng = trừ 50%). Không commit dữ liệu lớn; ghi link vào `data/README.md`.

## Cấu trúc

| Đường dẫn | Nội dung |
|-----------|----------|
| `src/linreg/config.py` | Đường dẫn, seed, cấu hình dataset (biến mục tiêu, đặc trưng) |
| `src/linreg/data.py` | Đọc CSV, chia train/test |
| `src/linreg/models.py` | `MODELS`: danh sách mô hình, mỗi mô hình là một `Pipeline` sklearn |
| `src/linreg/evaluate.py` | MAE, MSE, RMSE, R², CV, bảng hệ số |
| `src/linreg/train.py` | Huấn luyện tất cả, ghi `models/` |
| `app/streamlit_app.py` | Ứng dụng dự đoán và trực quan hoá |
| `notebooks/` | EDA và thí nghiệm, đánh số `01_eda.ipynb`, `02_…` |
| `figures/` | Hình xuất cho báo cáo (PDF/PNG) |
| `main.tex`, `references.bib` | Báo cáo (XeLaTeX, tiếng Việt) |

## Lệnh

```bash
pip install -r requirements.txt        # cài phụ thuộc + package linreg
pytest -q                              # smoke test trên dữ liệu tổng hợp
python -m linreg.train                 # huấn luyện, ghi models/
streamlit run app/streamlit_app.py     # chạy ứng dụng
xelatex main.tex                       # biên dịch báo cáo (hoặc dùng Oleafly/Overleaf)
```

## Quy ước

- Python ≥ 3.10, seed cố định `config.RANDOM_STATE` cho mọi thứ ngẫu nhiên.
- Tiền xử lý nằm **trong** `Pipeline` để tránh rò rỉ dữ liệu từ tập test.
- Thêm mô hình: thêm một entry vào `MODELS` kèm comment một dòng giải thích lý do; chạy lại `pytest`.
- Hình cho báo cáo: lưu vào `figures/` bằng code (notebook hoặc script), không chỉnh tay.
- Báo cáo viết bằng tiếng Việt; tên biến, code, commit message bằng tiếng Anh.
- Mỗi thay đổi đi qua branch + pull request; ít nhất một thành viên khác review trước khi merge vào `main`.
- Không commit: dữ liệu thô, file `.joblib`, `.env`, file build LaTeX, thư mục `.oleafly/`.
