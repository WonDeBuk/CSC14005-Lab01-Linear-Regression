# Order cho Neon: thu gọn báo cáo Lab 01 xuống 10–20 trang

- **Người giao:** Phan Quang Hiệp (WonDeBuk)
- **Ngày giao:** 24/11/2026
- **Đầu vào:** `main.tex`, `sections/*.tex`, `tables/*.tex`, `figures/tikz/*.tex` trên `main` (bản 43 trang, đã chuyển toàn bộ biểu đồ sang pgfplots)
- **Đầu ra:** một branch `report/condense` + pull request, `main.pdf` ≤ 20 trang, `pytest -q` vẫn qua

Trước khi làm, đọc `AGENTS.md` (quy tắc bắt buộc) và `docs/requirements.md` (đề bài). Order này không thay thế hai file đó.

## 1. Bối cảnh

Đề bài khuyến nghị báo cáo PDF **10–20 trang**. Bản hiện tại có **43 trang**:

| Phần | Trang hiện tại | Ghi chú |
|------|---------------:|---------|
| Bìa + Mục lục + Danh mục bảng + Danh mục hình + Bảng thuật ngữ | 7 | mỗi danh mục một trang riêng |
| 1. Giới thiệu | 2,5 | có mục "Cấu trúc báo cáo" liệt kê 8 chương |
| 2. Cơ sở lý thuyết | 5 | có cả chứng minh Gauss–Markov |
| 3. Dữ liệu, EDA, tiền xử lý | 6 | 4 hình, 4 bảng, 1 sơ đồ TikZ |
| 4. Xây dựng mô hình | 3 | mô tả từng mô hình bằng văn xuôi |
| 5. Đánh giá | 7 | 5 hình, 6 bảng |
| 6. Suy diễn và Bayes | 5 | 2 hình, 2 bảng |
| 7. Ứng dụng | 2 | 3 ảnh chụp màn hình |
| 8. Kết luận | 3 | |
| Tài liệu tham khảo | 1,5 | |

Cỡ chữ 12pt, giãn dòng 1,25 và `\parskip` 5pt cũng làm báo cáo dài thêm khoảng 25–30%.

## 2. Mục tiêu và tiêu chí nghiệm thu

1. `latexmk main.tex` biên dịch không lỗi, không có `Overfull \hbox` > 5pt, không có tham chiếu `??`.
2. **Tổng số trang PDF ≤ 20**, tính cả bìa, mục lục và tài liệu tham khảo. Phần thân (chương 1 đến hết Kết luận) nhắm **≤ 16 trang**.
3. Vẫn đủ các mục đề bài yêu cầu (`docs/requirements.md`): bài toán và động lực; dữ liệu (nguồn, tiền xử lý, trực quan hoá); thiết kế mô hình; đánh giá và so sánh; kết luận; ứng dụng; tài liệu tham khảo.
4. Vẫn giữ đủ **9 mô hình tuyến tính** (≥ 5 thành viên), cả hai lăng kính **Frequentist và Bayesian**, và các câu hỏi thống kê **có kiểm định** (bảng Q1–Q10).
5. Không đổi bất kỳ con số nào. Mọi số trong văn bản lấy từ macro của `tables/numbers.tex` (xem mục 5).
6. `pytest -q` qua, và không sửa `src/linreg/` hay `tests/`.

## 3. Ngân sách trang đề xuất

| Phần | Mục tiêu | Cách làm chính |
|------|---------:|----------------|
| Bìa | 1 | giữ nguyên |
| Mục lục | 1 | bỏ danh mục bảng, danh mục hình; mục lục chỉ tới `subsection` (`\setcounter{tocdepth}{2}`) |
| 1. Giới thiệu | 1 | bỏ mục 1.3 "Cấu trúc báo cáo"; gộp động lực + mục tiêu; giữ bảng phân công (rút gọn) |
| 2. Cơ sở lý thuyết | 2 | xem 4.2 |
| 3. Dữ liệu | 3 | xem 4.3 |
| 4. Mô hình | 1,5 | thay văn xuôi M1–M8b bằng một bảng "thang mô hình" |
| 5. Đánh giá | 3,5 | xem 4.5 |
| 6. Suy diễn và Bayes | 3 | xem 4.6 |
| 7. Ứng dụng | 1 | xem 4.7 |
| 8. Kết luận | 1 | |
| Tài liệu tham khảo | 1 | |
| **Tổng** | **≈ 19** | |

## 4. Việc cần làm

### 4.1 Bố cục chung (`main.tex`)

- `\documentclass[11pt,a4paper]{article}`, lề 2,0 cm, `\setstretch{1.1}`, `\parskip` 3pt.
- Bỏ `\listoftables`, `\listoffigures` và các `\clearpage` giữa chúng.
- Bỏ trang Bảng thuật ngữ (`sections/00-thuat-ngu.tex`). Thuật ngữ nào cần thì ghi tiếng Anh trong ngoặc ở lần xuất hiện đầu.
- Đặt `\captionsetup{font=footnotesize}` và bớt khoảng trắng quanh float (`\setlength{\textfloatsep}{8pt plus 2pt}`, `\intextsep` tương tự).
- Đo lại số trang sau bước này rồi mới cắt nội dung: chỉ riêng bố cục đã có thể giảm khoảng 10 trang.

### 4.2 Chương 2: Cơ sở lý thuyết (5 → 2 trang)

- **Giữ:** mô hình $y = X\beta + \varepsilon$ và nghiệm OLS; thống kê $t$ và $F$ từng phần; công thức CI/PI; HC3; t-test Nadeau–Bengio; Ridge ↔ MAP; tiên nghiệm và hậu nghiệm NIG; bảng đối chiếu Frequentist–Bayesian (`tab:freq-bayes`).
- **Bỏ:** chứng minh Gauss–Markov (chỉ phát biểu 4 giả thiết G1–G4 trong một câu, vì G3, G4 được nhắc lại ở chương 5 và phần hạn chế); công thức riêng của MSE/MAE/MedAE/$R^2_\text{adj}$/BIC (gộp thành một dòng); công thức Jarque–Bera và Cook (giữ tên kèm trích dẫn); phân rã Bias–Variance (giữ một câu).
- WLS và log–smearing: mỗi thứ tối đa 2 câu.

### 4.3 Chương 3: Dữ liệu (6 → 3 trang)

- Gộp `tab:vars` và `tab:fe` thành **một bảng**: cột gốc → xử lý → lý do.
- Bỏ `tab:desc` (`tables/desc-stats.tex`); đưa 4–5 con số chính vào câu văn bằng macro.
- **Hình 1** (phân phối CO₂, `fig01`): bỏ hình, giữ một câu kèm số độ lệch và JB.
- Giữ **Hình 2** (rò rỉ), **Hình 3** (tương quan + hộp theo thân xe), **Hình 4** (sửa cờ hybrid: đây là bước "trước/sau" mà `AGENTS.md` quy tắc 6 yêu cầu).
- Sơ đồ pipeline TikZ (`fig:pipeline`): thu nhỏ thành một hàng 4 ô, hoặc bỏ và thay bằng một câu.
- `tab:eda-groups`: giữ, nhưng chỉ còn các cột trung bình CO₂ và dung tích trung bình (bằng chứng sớm cho nghịch lý Simpson).

### 4.4 Chương 4: Mô hình (3 → 1,5 trang)

- Thay danh sách văn xuôi M1–M8b bằng **một bảng "thang mô hình"**: Mô hình | Thêm gì so với mô hình trước | $p$ | Lý do (một dòng). Các số $F$, RMSE đã có ở `tab:models` và `tab:ftests` nên không lặp lại.
- Giữ đoạn cảnh báo p-value lạc quan của M8b (quy tắc 7) và đoạn mốc Gaussian Process (ghi rõ đây là mốc phi tuyến, không tính vào số mô hình).
- Có thể chuyển **Hình 9** (tương tác M8b) sang chương 6, cạnh phần diễn giải độ dốc.

### 4.5 Chương 5: Đánh giá (7 → 3,5 trang)

- Giữ `tab:models` (bảng chính).
- Gộp `tab:ftests` và `tab:nb` thành **một bảng kiểm định so sánh mô hình** ($F$ | NB $p$ | Wilcoxon $p$).
- **Hình 5:** chỉ giữ panel (a) hoặc (b). Gợi ý: giữ (a) vì đã có cả CV, test và mốc GP.
- **Hình 6 và Hình 7:** gộp thành một hình hai panel: (a) đường cong học tập, (b) Bias²–Variance. Bỏ panel đường chọn λ của Ridge, thay bằng một câu ($\lambda^*$ và RMSE CV từ macro).
- Bỏ `tab:fit-diag`; giữ 1–2 con số tỉ lệ phương sai qua macro `\bvVarShare*`.
- Giữ **Hình 8** (chẩn đoán phần dư 2×2): đây là căn cứ để dùng HC3.
- Độ bền vững: bỏ **Hình 12**, `tab:split`, `tab:noise`; thay bằng 2–3 câu (thứ hạng không đổi giữa các tỉ lệ chia; RMSE dưới nhiễu σ = 0,5).
- Đoạn "khả năng tái lập": rút còn một câu.

### 4.6 Chương 6: Suy diễn và Bayes (5 → 3 trang)

- Giữ `tab:coef`, **Hình 10** (Simpson/E85), **Hình 11** (Frequentist vs Bayes), `tab:q-summary`.
- Văn xuôi ở mục 6.1 đang lặp lại số đã có trong `tab:coef` và `tab:q-summary`: chỉ giữ phần diễn giải (vì sao hệ số dung tích giảm từ M1 sang M3h, ý nghĩa của "không bác bỏ $H_0$").
- Mục 6.2–6.3: mỗi mục tối đa 2 đoạn.

### 4.7 Chương 7: Ứng dụng (2 → 1 trang)

- Giữ ảnh `app-desktop.png` (chiều rộng 0,8\textwidth), bỏ hình `fig:app2` (hai ảnh x-ray và cards).
- Danh sách tính năng: 4 gạch đầu dòng, mỗi dòng ≤ 2 dòng in.
- Giữ link ứng dụng; chừa chỗ cho link video demo (`% TODO(team)` đã có).

### 4.8 Chương 8: Kết luận (3 → 1 trang)

- 4 kết quả chính: mỗi ý ≤ 3 dòng, không lặp lại đủ bộ số đã có ở chương 5–6.
- Hạn chế: giữ 5 ý, mỗi ý 1 dòng. Hướng phát triển: 3 ý, mỗi ý 1 dòng.

## 5. Lỗi phát hiện khi review: sửa trong lúc thu gọn

Các lỗi dưới đây phải được sửa trong PR này (hoặc biến mất vì đoạn chứa nó bị cắt).

**a. Số viết tay thay vì macro (vi phạm quy tắc 3).** Thêm macro vào `analysis/analyze.py` rồi chạy lại để sinh `tables/numbers.tex`, không gõ số vào `.tex`:

| Vị trí | Số viết tay |
|--------|-------------|
| `04-mo-hinh.tex` (M3h) | "10 mẫu bị sót" → đã có `\nHybridMissed` |
| `04-mo-hinh.tex` (GP), `08-ket-luan.tex` | "0,45 g/km" (M8b − GP) |
| `05-danh-gia.tex` (tầng hybrid) | "1,21 g/km" (M3 − M3h) |
| `05-danh-gia.tex` (tầng cơ sở) | "$R^2 \approx 0{,}76$" |
| `05-danh-gia.tex` (kiểm định) | "$p > 0{,}87$" |
| `05-danh-gia.tex` (đường cong học tập) | "1,2–1,9 g/km", "hơn 97%" (cũng có ở `08-ket-luan.tex`), "4–15 ms", "$n/p \approx 45$" |
| `05-danh-gia.tex` (PI) | "200/213 xe", "≥ 294 g/km" |
| `05-danh-gia.tex` (độ bền) | "$R^2 \in [0{,}884; 0{,}887]$", "$[0{,}869; 0{,}871]$" |
| `05-danh-gia.tex` (Cook) | thông số Bentley, Audi R8, Lancer Evolution (6,8 L, 437 g/km, +93,0, …) |
| `05-danh-gia.tex` (tái lập) | "241 macro" |
| `06-dien-giai.tex` (E85) | "50,0 g/km" (\meanE − \meanX), bậc tự do $F(1, 851)$, $F(13, 838)$… |
| `06-dien-giai.tex` (bảng Q10) | "khoảng 18–19 g/km" |

**b. Ngôn ngữ nhân quả mâu thuẫn với phần hạn chế.** Phần 8.2 nói rõ hệ số là *liên hệ thống kê, không phải nhân quả*, nhưng chương 6 và bảng Q viết "E85 **làm giảm** … g/km", "hybrid … **mang lại** mức giảm \COtwo{} lớn nhất", "Q4: Hệ truyền động hybrid **làm giảm** CO₂?". Đổi thành "gắn với mức thấp hơn …" hoặc "chênh … so với xăng thường khi giữ các biến khác cố định".

**c. Nhận định không có trong dữ liệu (vi phạm quy tắc 2).**
- `04-mo-hinh.tex`, đoạn mở M8b: nói Hình 2(b) cho thấy "xe hybrid và bán tải có độ dốc phẳng hơn", nhưng Hình 2(b) không tách xe hybrid. Sửa lại lập luận (ví dụ dựa vào phần dư M3h theo nhóm) và ghi rõ tương tác được chọn trên tập train.
- `05-danh-gia.tex`, đoạn nhiễu Gaussian: "không xảy ra sụp đổ bất thường như các mô hình đa thức bậc cao". Nhóm không thử mô hình đa thức bậc cao nào, nên bỏ vế so sánh này.
- `06-dien-giai.tex`, mục 6.1: "thuộc nhóm thân xe **nặng hơn**". Dữ liệu không có khối lượng xe; chỉ nói "thuộc nhóm SUV, bán tải, van" (có số trong `tab:eda-groups`).
- `01-gioi-thieu.tex`, câu mở đầu "CO₂ từ giao thông đường bộ là một trong những nguồn phát thải chính": cần trích dẫn có thật trong `references.bib` hoặc bỏ.

**d. Đánh dấu cho người đọc lại (quy tắc 1).** Mọi đoạn diễn giải hoặc kết luận bị viết lại phải có `% TODO(review)` ngay phía trên. Không xoá các `TODO(review)` / `TODO(team)` hiện có (phân công, link video).

## 6. Ràng buộc

- **Không tự viết mới phần phân tích hay kết luận.** Việc của Neon là cắt, gộp, rút gọn câu chữ đã có. Không thêm phát hiện, nhận định hay trích dẫn mới.
- Không đổi số, không chạy lại mô hình với tham số khác. Nếu cần số mới (macro mới ở mục 5a) thì thêm vào `analysis/analyze.py`, chạy `.venv/bin/python analysis/analyze.py`, và commit cả `tables/numbers.tex`.
- **Biểu đồ:** tất cả 12 biểu đồ là pgfplots trong `figures/tikz/fig*.tex`, đọc dữ liệu từ `figures/data/` (sinh bằng `.venv/bin/python -m analysis.plot_data`). Muốn bỏ panel thì sửa file `.tex` của hình đó (xoá `\nextgroupplot` tương ứng và đổi `group size`). Không chỉnh tay `figures/data/*`. Muốn gộp Hình 6 và Hình 7 thì tạo file mới `figures/tikz/fig06-learning-bias-variance.tex` dùng lại `fig07-learning.dat` và `fig06-bv.dat`.
- Nếu một hình bị bỏ khỏi báo cáo, xoá file `figures/tikz/` tương ứng và bỏ hàm export tương ứng trong `analysis/plot_data.py` để repo không còn code chết.
- Giữ pdfLaTeX + vntex T5; không dùng fontspec.
- Không commit file build LaTeX, `.oleafly/`, `main.pdf` (xem `.gitignore`, quy tắc 8).
- Commit message bằng tiếng Anh; nội dung báo cáo bằng tiếng Việt.

## 7. Bàn giao

Trong mô tả PR, ghi:

1. Số trang trước và sau (tổng và phần thân), lấy bằng `pdfinfo main.pdf`.
2. Bảng "đã bỏ / đã gộp" theo từng hình và bảng (số cũ → số mới).
3. Danh sách macro mới thêm vào `analysis/analyze.py`.
4. Danh sách các đoạn đã gắn `% TODO(review)` để thành viên phụ trách đọc lại (xem bảng phân công trong `sections/01-gioi-thieu.tex`).

Ít nhất một thành viên khác review PR trước khi merge.
