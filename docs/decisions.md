# Quyết định và việc còn mở

Tổng hợp từ moodboard của nhóm ("Mission LR-01", [artifact](https://claude.ai/artifact/95knfAAW2ceeRQWnin9HKj)) và hai phiên chat với Claude (22–23/11/2026). Moodboard là nơi phân tích chi tiết; repo này là nơi chạy lại và tái lập.

## Đã quyết

| Chủ đề | Quyết định |
|--------|-----------|
| Dataset | CO₂ Canada 2014, 1.067 xe (chọn từ 12 bộ đã khảo sát; dự phòng: Medical Cost, Ames Housing) |
| Biến mục tiêu | `CO2EMISSIONS` (g/km) |
| Loại bỏ | Cột tiêu thụ nhiên liệu (rò rỉ: tỉ số CO₂/nhiên liệu cố định 23 cho xăng, 27 cho diesel, 16 cho E85) |
| Cờ hybrid | Sửa nhãn cho 10 xe bị sót (Lexus "…h", Prius, CR-Z): hệ số hybrid −44,5 → −48,3 g/km, BIC giảm 35 |
| Mô hình chính | M3h: động cơ, xi-lanh, nhiên liệu, nhóm thân xe, hộp số, hybrid |
| Mô hình mở rộng | M8b = M3h + dung tích × (bán tải, van, hybrid); p = 0,048 so với M3h, lạc quan vì tương tác được tìm trên chính tập train |
| Hai trường phái | OLS (Frequentist) đặt cạnh Ridge và hồi quy Bayesian; với 854 xe cả hai gần trùng OLS, prior chỉ ảnh hưởng rõ trên mẫu con 40 xe |
| Kiểm định | HC3 và bootstrap cho hệ số; Nadeau–Bengio cho so sánh CV |
| Phát hiện chính | Nghịch lý Simpson: E85 có CO₂ trung bình thô cao nhất, nhưng khi giữ các yếu tố khác không đổi thì thải ít hơn ≈ 18 g/km (xác nhận bằng hồi quy, Bayesian, và kiểm định ghép cặp trên 78 xe flex-fuel) |
| Ứng dụng | Xưởng xe 3D pixel (three.js, Kenney Car Kit CC0) trên moodboard; repo có thêm bản Streamlit |

## Việc còn mở

- [ ] **Thống nhất tên mô hình.** Mục 06 của moodboard gọi Ridge là M4 và Bayesian là M5; mục 09 gọi là M5 và M6. Repo dùng tên mã (`m1_engine`, `m3h_ols`, `m3h_ridge`, `m3h_bayes`, `m8b_interactions`) cho đến khi nhóm chốt.
- [ ] **Đối chiếu nhóm thân xe và phép chia với moodboard.** Repo gom 16 hạng xe thành car/wagon/suv/pickup/van và chia bằng `train_test_split(test_size=213, random_state=42)`. Kết quả gần với moodboard nhưng chưa khớp từng số (ví dụ CV RMSE M3h 23,07 so với 23,09–23,26), nên cách gom nhóm hoặc cách chia có thể khác. Cần chép script `analysis/` từ phiên moodboard vào repo để khớp số.
- [ ] **Ứng dụng dùng M3h hay M8b.** Moodboard vẫn chạy M3h với cờ hybrid cũ.
- [ ] **Khoảng dự đoán 95% quá hẹp ở nhóm xe thải nhiều** (phủ 93,9% tập test nói chung, chỉ 87% ở nhóm cao nhất).
- [ ] **Mức Bayesian thầy mong đợi.** Đề Lab 01 không nhắc Bayesian; cần xác nhận với thầy.
- [ ] **Phân công:** mỗi thành viên phụ trách một mô hình (bảng trong `README.md`).
