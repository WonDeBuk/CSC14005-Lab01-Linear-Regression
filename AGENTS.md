# Hướng dẫn cho AI agent

File này dành cho mọi AI coding agent (Claude Code, Codex, Copilot, Cursor, …) làm việc trong repo. `CLAUDE.md` import file này.

## Bối cảnh

Đồ án **CSC14005 Lab 01 — Linear Regression** (Nhập môn học máy, Khoa CNTT, Trường ĐH Khoa học Tự nhiên, ĐHQG-HCM). GVHD: Thầy Bùi Tiến Lên, Thầy Võ Nhật Tân. Nhóm 5 thành viên (xem `README.md`).

- **Bài toán:** ước tính lượng phát thải CO₂ (g/km) của xe hạng nhẹ chỉ từ thông số thiết kế (động cơ, xi-lanh, nhiên liệu, nhóm thân xe, hộp số, hybrid), tức những thứ có trước khi xe chạy thử.
- **Mục tiêu:** cực tiểu RMSE ngoài mẫu (CV 10-fold) mà mô hình vẫn diễn giải được, và trả lời có kiểm định câu hỏi "quyết định thiết kế nào đổi thành bao nhiêu gam CO₂".
- **Dữ liệu:** Fuel Consumption Ratings 2014, Canada, 1.067 xe. Chi tiết: [data/README.md](data/README.md).
- **Lăng kính thống kê:** thầy yêu cầu trình bày Linear Regression dưới cả hai trường phái **Frequentist** (OLS, kiểm định t/F, khoảng tin cậy) và **Bayesian** (prior, posterior, khoảng đáng tin). Báo cáo phải **đặt và kiểm định câu hỏi thống kê**, không chỉ xây mô hình rồi so điểm. Tài liệu chính: sách "Phân tích dữ liệu" (book-stat) của thầy Lên, chương 5–8, 10; khoá [PSU STAT 501](https://online.stat.psu.edu/stat501/).
- **Moodboard (nguồn phân tích gốc):** [Mission LR-01](https://claude.ai/artifact/95knfAAW2ceeRQWnin9HKj). Repo tái hiện đúng từng số của moodboard; `tests/test_pipeline.py` khoá các số này. Thay đổi làm lệch số phải được ghi lý do.
- **Đề bài:** [docs/requirements.md](docs/requirements.md). **Các quyết định đã có và việc còn mở:** [docs/decisions.md](docs/decisions.md).

## Quy tắc bắt buộc

1. **AI chỉ hỗ trợ, người quyết định.** Đề bài trừ tới 50% điểm nếu lạm dụng AI hoặc nội dung sai. Agent không tự viết hoàn chỉnh phần phân tích, kết luận hay diễn giải trong báo cáo: chỉ soạn nháp, đánh dấu `% TODO(review)` trong LaTeX để thành viên đọc lại và viết lại bằng lời của mình.
2. **Diễn giải chỉ dựa trên kết quả mô hình**, không đem kiến thức miền bên ngoài vào nếu số liệu không cho thấy điều đó.
3. **Không bịa số liệu hay trích dẫn.** Mọi số trong báo cáo phải dùng macro từ `tables/numbers.tex` (sinh bởi `analysis/analyze.py`), không gõ tay vào `.tex`. Tài liệu tham khảo phải có thật và nằm trong `references.bib`.
4. **Không dùng các cột `FUELCONSUMPTION_*`** làm đầu vào: CO₂ được tính thẳng từ chúng (rò rỉ dữ liệu, R² ≈ 1).
5. **Mọi mô hình tuyến tính theo tham số.** Mô hình phi tuyến (Kernel Ridge, Gaussian Process, …) chỉ được dùng làm mốc so sánh, không tính vào "số mô hình ≥ số thành viên".
6. **Mỗi bước xử lý và mỗi thí nghiệm trả lời What / Why / How:** điều gì đang xảy ra với dữ liệu, tại sao, xử lý bằng kiến thức nào. Kèm biểu đồ trước/sau.
7. **Kiểm định và diễn giải đúng cách:**
   - Phần dư có phương sai thay đổi (Breusch–Pagan p < 10⁻¹⁴): dùng sai số chuẩn **HC3** hoặc bootstrap cho kiểm định hệ số.
   - So sánh mô hình trên các fold CV: dùng t-test hiệu chỉnh **Nadeau–Bengio**, không dùng t ghép cặp thường.
   - Tương tác hay biến tìm bằng chọn biến trên tập train: ghi rõ p-value bị lạc quan.
   - Không xoá các xe có phần dư lớn: chúng là xe thật (Bentley, Audi R8, …), không phải lỗi nhập liệu.
   - Dùng ngôn ngữ liên hệ thống kê ("gắn với mức thấp hơn / cao hơn khi giữ các biến khác cố định"), tránh khẳng định nhân quả ("làm giảm / làm tăng").
8. **Repo phải public.** Không commit secrets, `.joblib`, file build LaTeX, `main.pdf`, `analysis/out/`, thư mục `.oleafly/`, state Terraform (`*.tfstate`), `node_modules/`, `web/dist/`.
9. **Đổi mô hình thì chạy lại `python -m linreg.export_web`**, để số trên ứng dụng khớp với Python.

## Cấu trúc

| Đường dẫn | Nội dung |
|-----------|----------|
| `src/linreg/config.py` | Đường dẫn, seed 42, chia 854/213, CV 10-fold, cột cần bỏ |
| `src/linreg/features.py` | Các bước P5–P8: nhóm thân xe, hộp số, cờ hybrid, biến tương tác |
| `src/linreg/data.py` | Đọc CSV, thêm đặc trưng, chia train/test |
| `src/linreg/models.py` | `MODELS`: M1–M6, M8b, đặt tên theo mục 09 của moodboard |
| `src/linreg/bayes.py` | Hồi quy Bayesian liên hợp Normal-Inverse-Gamma (M6): khoảng đáng tin, khoảng dự đoán |
| `src/linreg/inference.py` | OLS statsmodels (HC3), F từng phần, Breusch–Pagan, Jarque–Bera |
| `src/linreg/evaluate.py` | MAE, MSE, RMSE, R², CV, t-test Nadeau–Bengio, bảng hệ số |
| `src/linreg/train.py` | Huấn luyện tất cả, ghi `models/` |
| `src/linreg/export_web.py` | Xuất hệ số M3h, M8b, (XᵀX)⁻¹ và σ² ra `web/public/model.json` |
| `web/` | CO₂ Studio: React + TypeScript + three.js; `src/model.ts` tính dự đoán và khoảng, `src/scene/Studio.ts` là cảnh 3D |
| `infra/` | Terraform: S3 riêng tư + CloudFront (OAC), upload `web/dist` |
| `notebooks/` | EDA và thí nghiệm, đánh số `01_eda.ipynb`, `02_…` |
| `analysis/` | `analyze.py` (tái lập kiểm định, sinh `tables/*.tex`, `tables/numbers.tex`), `plot_data.py` (xuất `figures/data/*`) |
| `sections/` | 8 chương báo cáo (`01-gioi-thieu.tex` … `08-ket-luan.tex`) |
| `tables/` | Các bảng LaTeX và `numbers.tex` (toàn bộ sinh tự động từ `analysis/analyze.py`) |
| `figures/` | Biểu đồ pgfplots trong `figures/tikz/`, dữ liệu trong `figures/data/`, ảnh `logo-hcmus.png` và `app-desktop.png` |
| `main.tex`, `references.bib` | Báo cáo (pdfLaTeX, 11pt, 20–30 trang, tiếng Việt bằng vntex T5; không dùng fontspec) |

## Lệnh

```bash
pip install -r requirements.txt        # cài phụ thuộc + package linreg
pytest -q                              # kiểm tra pipeline, khớp số với moodboard
python -m linreg.train                 # huấn luyện, ghi models/
python -m linreg.export_web            # cập nhật model.json cho ứng dụng web
python analysis/analyze.py             # chạy phân tích đầy đủ, sinh tables/*.tex và numbers.tex
python -m analysis.plot_data           # xuất dữ liệu biểu đồ pgfplots vào figures/data/
cd web && npm run dev                  # chạy ứng dụng web
cd infra && terraform apply            # triển khai (sau khi npm run build)
latexmk main.tex                       # biên dịch báo cáo bằng pdfLaTeX (latexmkrc)
```

## Quy ước

- Python ≥ 3.10; mọi thứ ngẫu nhiên dùng `config.RANDOM_STATE`.
- Tiền xử lý nằm **trong** `Pipeline` để không rò rỉ từ tập test.
- Thêm mô hình: thêm một `ModelSpec` vào `MODELS` (kèm mô tả lý do); công thức statsmodels được sinh tự động nếu `ols=True`. Chạy lại `pytest`.
- Biểu đồ trong báo cáo vẽ bằng pgfplots (`figures/tikz/`); dữ liệu sinh bằng `python -m analysis.plot_data` vào `figures/data/`, không chỉnh tay.
- Báo cáo PDF giữ trong khoảng **20–30 trang** (theo yêu cầu mới của nhóm để trình bày thoáng các bảng và biểu đồ trước/sau), biên dịch sạch không có `Overfull \hbox` > 5pt hay tham chiếu `??`; mọi chú giải (legend) đặt ngoài vùng dữ liệu để không che điểm/đường.
- Báo cáo bằng tiếng Việt; tên biến, code, commit message bằng tiếng Anh.
- Mỗi thay đổi đi qua branch + pull request, có ít nhất một thành viên khác review.
