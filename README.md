# CSC14005 — Lab 01: Linear Regression

Ước tính lượng phát thải CO₂ (g/km) của xe hạng nhẹ từ thông số thiết kế bằng hồi quy tuyến tính, dưới lăng kính Frequentist và Bayesian. Đồ án môn Nhập môn học máy (CSC14005), Khoa Công nghệ Thông tin, Trường ĐH Khoa học Tự nhiên, ĐHQG-HCM.

- Yêu cầu đề bài: [docs/requirements.md](docs/requirements.md)
- Quyết định đã có và việc còn mở: [docs/decisions.md](docs/decisions.md)
- Dữ liệu: [data/README.md](data/README.md)
- Hướng dẫn cho AI agent và quy ước code: [AGENTS.md](AGENTS.md)
- Moodboard của nhóm: [Mission LR-01](https://claude.ai/artifact/95knfAAW2ceeRQWnin9HKj)
- **Ứng dụng đang chạy: [CO₂ Studio](https://d1dts9wd5j3er9.cloudfront.net)**

## Tech stack

| Phần | Công nghệ |
|------|-----------|
| Phân tích, mô hình, kiểm định | Python 3.12, pandas, scikit-learn, statsmodels, SciPy, pytest |
| Ứng dụng web | React 19, TypeScript, Vite, three.js (RenderPixelatedPass), mẫu xe Kenney Car Kit (CC0) |
| Hạ tầng | Terraform, AWS S3 (riêng tư) + CloudFront (Origin Access Control, HTTPS) |
| Báo cáo | LaTeX (pdfLaTeX qua latexmk, tiếng Việt bằng vntex) |
| CI | GitHub Actions: pytest, build web, `terraform validate` |

Mô hình chạy ngay trên trình duyệt từ `web/public/model.json`, nên ứng dụng không cần máy chủ hay API.

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
python -m linreg.train        # huấn luyện 9 mô hình, ghi models/metrics.csv và hệ số
python -m linreg.export_web   # xuất hệ số M3h, M8b cho ứng dụng web
```

Kiểm định (Frequentist, sai số chuẩn HC3) trên tập train:

```python
import pandas as pd
from linreg.data import load_dataset
from linreg.inference import fit_ols, partial_f_test, residual_diagnostics

ds = load_dataset()
train = pd.concat([ds.X_train, ds.y_train], axis=1)
print(fit_ols(train, "M3h").summary())        # t-test từng hệ số, KTC 95%
print(partial_f_test(train, "M3", "M3h"))     # F từng phần cho mô hình lồng nhau
```

## Ứng dụng web: CO₂ Studio

Xưởng xe 3D pixel (React + TypeScript + three.js) trong [`web/`](web/). Chọn thân xe, động cơ, nhiên liệu, hộp số và hybrid; mô hình M3h hoặc M8b tính CO₂ ngay trên trình duyệt, kèm khoảng tin cậy và khoảng dự đoán 95%, vị trí trong đội xe 2014 và phần đóng góp của từng yếu tố. Mẫu xe: [Kenney Car Kit](https://kenney.nl/assets/car-kit) (CC0).

```bash
cd web && npm install && npm run dev
```

### Triển khai lên AWS

Hạ tầng viết bằng Terraform trong [`infra/`](infra/): bucket S3 riêng tư, CloudFront với Origin Access Control, header bảo mật, và upload toàn bộ `web/dist`. File JS/CSS có hash trong tên nên được cache một năm; các file còn lại luôn kiểm tra lại, nên không cần xoá cache CloudFront sau mỗi lần cập nhật.

```bash
aws login --region ap-southeast-1             # hoặc aws configure sso
python -m linreg.export_web && (cd web && npm ci && npm run build)
cd infra && terraform init && terraform apply
terraform destroy                             # xoá toàn bộ khi không dùng nữa
```

State của Terraform nằm cục bộ trong `infra/` (không commit). Chỉ người triển khai giữ state; muốn cả nhóm cùng triển khai thì chuyển sang backend S3.

## Mô hình

Tên mô hình theo mục 09 của moodboard. Định nghĩa trong [`src/linreg/models.py`](src/linreg/models.py); mọi mô hình tuyến tính theo tham số. M7 (rò rỉ qua mức tiêu thụ nhiên liệu) bị loại có chủ đích. Kết quả từ `python -m linreg.train` (chia 854/213 bằng `default_rng(42)`, CV 10-fold trên tập train) khớp từng số với moodboard; `pytest` khoá các số này.

| Mã | Mô hình | Số tham số | RMSE CV (g/km) | RMSE test (g/km) | R² test |
|----|---------|-----------:|---------------:|-----------------:|--------:|
| M1 | Hồi quy đơn theo dung tích động cơ (mốc) | 2 | 30,16 ± 2,75 | 33,01 | 0,758 |
| M2 | M1 + số xi-lanh | 3 | 29,63 ± 2,69 | 32,46 | 0,766 |
| M3 | M2 + nhiên liệu, nhóm thân xe, hộp số | 16 | 23,80 ± 2,08 | 25,29 | 0,858 |
| M3h | M3 + cờ HYBRID (đã sửa nhãn, tức M3h′ ở mục 12) | 17 | 22,56 ± 2,29 | 24,08 | 0,871 |
| M4 | M3 + dung tích², dung tích × (diesel, E85) | 19 | 23,80 ± 2,12 | 25,43 | 0,856 |
| M4log | M4 trên log(CO₂), đổi ngược bằng smearing của Duan | 19 | 23,76 ± 2,15 | 25,37 | 0,857 |
| M5 | Ridge trên biến M4, λ = 3 (cầu nối: MAP với prior Gauss) | 19 | 23,80 ± 2,10 | 25,38 | 0,857 |
| M6 | Bayesian Normal-Inverse-Gamma trên biến M3h, prior yếu | 17 | 22,56 ± 2,29 | 24,08 | 0,871 |
| M8b | M3h + dung tích × (bán tải, van, hybrid) | 20 | 21,61 ± 2,29 | 22,61 | 0,886 |

## Cấu trúc thư mục

```
├── analysis/              # analyze.py (bảng + macro số liệu), plot_data.py (dữ liệu pgfplots)
├── web/                   # CO₂ Studio: React + TypeScript + three.js
├── infra/                 # Terraform: S3 + CloudFront
├── data/raw/              # FuelConsumptionCo2.csv (1.067 xe, 2014)
├── docs/                  # đề bài, quyết định của nhóm
├── figures/               # biểu đồ pgfplots (tikz/, data/) và ảnh báo cáo
├── sections/              # 8 chương báo cáo LaTeX
├── tables/                # bảng LaTeX và numbers.tex sinh tự động
├── models/                # mô hình đã huấn luyện (không commit)
├── notebooks/             # EDA và thí nghiệm
├── src/linreg/            # config, features, data, models, bayes, inference, evaluate, train
├── tests/                 # kiểm tra pipeline
├── main.tex               # báo cáo (pdfLaTeX: latexmk main.tex, ≤ 20 trang)
└── references.bib
```

## Làm việc nhóm với AI

Repo được thiết kế để các thành viên dùng AI coding agent (Claude Code, Codex, Copilot, Cursor, …). Agent đọc [`AGENTS.md`](AGENTS.md) (Claude Code đọc qua `CLAUDE.md`), trong đó có bối cảnh đồ án và các quy tắc của đề bài về sử dụng AI. Tóm tắt:

- AI hỗ trợ viết code và soạn nháp; **thành viên nhóm kiểm tra và viết lại** phần phân tích, diễn giải, kết luận.
- Không bịa số liệu hay trích dẫn.
- Mỗi thay đổi qua pull request, có người review.
