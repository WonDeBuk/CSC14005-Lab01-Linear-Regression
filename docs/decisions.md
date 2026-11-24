# Quyết định và việc còn mở

Tổng hợp từ moodboard của nhóm ("Mission LR-01", [artifact](https://claude.ai/artifact/95knfAAW2ceeRQWnin9HKj), bản `lr01-moodboard.html` ngày 24/11/2026) và hai phiên chat với Claude (22–23/11/2026). Moodboard là nơi phân tích chi tiết; repo này tái hiện từng con số của nó bằng code và test.

## Đã quyết

| Chủ đề | Quyết định | Mục moodboard |
|--------|-----------|---------------|
| Dataset | CO₂ Canada 2014, 1.067 xe (chọn từ 12 bộ đã khảo sát) | 02–04, 07 |
| Mục tiêu | Ước tính CO₂ (g/km) chỉ từ thông số thiết kế; cực tiểu RMSE ngoài mẫu mà vẫn diễn giải được; trả lời có kiểm định "quyết định thiết kế nào đổi thành bao nhiêu gam CO₂" | 07 |
| Loại cột | `MODELYEAR` (hằng), `FUELCONSUMPTION_*` (rò rỉ: CO₂/COMB = 23 xăng, 27 diesel, 16 E85) | 08 P1–P2 |
| Xe flex-fuel | Giữ cả hai dòng của 78 cặp xe; dùng làm kiểm định t ghép cặp cho E85; ghi hạn chế về tính độc lập | 08 P3 |
| Biến mục tiêu | Giữ CO₂ gốc (log không cải thiện, xem M4log) | 08 P4, 09 |
| Nhóm thân xe | 5 nhóm, tham chiếu Xe nhỏ | 08 P5 |
| Hộp số | Kiểu (tham chiếu A) + số cấp + cờ NOGEAR cho xe vô cấp | 08 P6 |
| Đa cộng tuyến | Giữ cả dung tích và xi-lanh (VIF 9,7 và 9,6 < 10), diễn giải thận trọng | 08 P7 |
| Cờ hybrid | Sửa nhãn cho 10 xe bị sót (Lexus "…h", Prius, CR-Z): hệ số −44,5 → −48,3 g/km, BIC giảm 35 | 08 P8, 12.8 |
| Chia dữ liệu | `default_rng(42).permutation`, train 854, test 213; chọn mô hình bằng CV, mở test một lần | 08 P9 |
| Tên mô hình | Theo mục 09 (M1–M7), mục 16 cũng đề xuất dùng bộ này | 09, 16 |
| Phương sai thay đổi | Breusch–Pagan bác bỏ; dùng SE HC3 và bootstrap; WLS và log không sửa được | 09, 12 |
| So sánh CV | t-test hiệu chỉnh Nadeau–Bengio | 10.2 |
| Mô hình phi tuyến | Gaussian Process chỉ là mốc trần, không tính vào số mô hình | 12 |
| Khung báo cáo | 8 chương, khoảng 19 trang (đã đưa vào `main.tex`) | 16 |
| Ứng dụng | CO₂ Studio (React + three.js, xưởng xe pixel), mô hình chạy trên trình duyệt; bỏ bản Streamlit. Giao diện theo phong cách "Orbital Drift" của moodboard (nền vũ trụ tối, panel HUD, Space Grotesk, Be Vietnam Pro, JetBrains Mono) | 01, 14 |
| Hosting | S3 + CloudFront, viết bằng Terraform; không dùng EC2 hay Lambda vì không có phần máy chủ | — |

## Việc còn mở

- [ ] **Mô hình cuối: M3h hay M8b.** M8b có RMSE test 22,6 và BIC thấp nhất nhưng các tương tác được tìm trên chính tập train (p = 0,048 so với M3h cờ cũ, lạc quan). Ứng dụng web mặc định M8b và cho chọn M3h (`WEB_MODELS` trong `src/linreg/export_web.py`); moodboard mục 14 vẫn chạy M3h cờ cũ.
- [ ] **Phân công 5 kênh mô hình** (mục 06 và bảng phân công mục 16): M1, M2, M3 (gồm M3h, M4, M4log), Ridge (M5), Bayesian (M6). Điền vào cột "Mô hình phụ trách" trong `README.md`.
- [ ] **Khoảng dự đoán 95% quá hẹp ở nhóm xe thải nhiều** (phủ 93,9% tập test nói chung, chỉ 87% ở nhóm cao nhất). Ứng dụng web tính khoảng dự đoán OLS đầy đủ (σ̂·√(1 + h)) nhưng vẫn giả định phương sai không đổi, nên vẫn hẹp ở nhóm này.
- [ ] **Mức Bayesian thầy mong đợi.** Đề Lab 01 không nhắc Bayesian; cần xác nhận với thầy.
- [ ] **Chuyển phần còn lại của moodboard thành code:** kiểm định ghép cặp 78 xe flex-fuel, bootstrap hệ số, đường cong học, bias–variance, độ bền theo tỉ lệ chia và nhiễu, 20 câu hỏi kiểm định (mục 10, 11, 13).
- [ ] **Video demo** quay trên [bản đang chạy](https://d1dts9wd5j3er9.cloudfront.net).
