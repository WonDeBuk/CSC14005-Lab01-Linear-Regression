# Dữ liệu

| Mục | Giá trị |
|-----|---------|
| Tên | Fuel Consumption Ratings 2014 (bản rút gọn `FuelConsumptionCo2.csv`) |
| Đơn vị công bố | Natural Resources Canada — [Open Government Portal](https://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64) |
| Giấy phép | [Open Government Licence – Canada](https://open.canada.ca/en/open-government-licence-canada) (được phân phối lại, phải ghi nguồn) |
| Bản dùng trong repo | [IBM Skills Network, ML0101EN](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv), đã đổi ký tự xuống dòng `\r` thành `\n` |
| Kích thước | 1.067 xe × 13 cột, chỉ năm 2014 |
| SHA-256 | `04431e34d62e95e5a1c481c7b8b2b988c95a8ce5c099189a39740fc86260ffa3` |
| Biến mục tiêu | `CO2EMISSIONS` (g/km) |

File đã được commit tại `data/raw/FuelConsumptionCo2.csv` để link dữ liệu luôn truy cập được cùng repo (đề bài trừ 50% nếu link hỏng).

## Các cột

| Cột | Ý nghĩa | Dùng làm đầu vào |
|-----|---------|------------------|
| `MODELYEAR`, `MAKE`, `MODEL` | Năm, hãng, mẫu xe | Không (chỉ dùng để gắn cờ hybrid) |
| `VEHICLECLASS` | 16 hạng xe → gom thành 5 nhóm `BODY`: car, wagon, suv, pickup, van | Có, qua `BODY` |
| `ENGINESIZE` | Dung tích động cơ (L) | Có |
| `CYLINDERS` | Số xi-lanh | Có |
| `TRANSMISSION` | Ví dụ `AS6` → `TRANS_TYPE` = AS, `GEARS` = 6 | Có, qua `TRANS_TYPE` |
| `FUELTYPE` | X xăng thường, Z xăng cao cấp, D diesel, E ethanol E85 | Có |
| `FUELCONSUMPTION_*` (4 cột) | Mức tiêu thụ nhiên liệu | **Không: rò rỉ dữ liệu.** CO₂ được quy đổi trực tiếp từ mức tiêu thụ theo tỉ lệ cố định cho từng loại nhiên liệu |
| `CO2EMISSIONS` | Phát thải CO₂ (g/km) | Biến mục tiêu |

Cờ `HYBRID` (34 xe) gắn khi tên mẫu xe chứa "HYBRID", hoặc là Prius, CR-Z, hoặc Lexus có đuôi "h" (CT 200h, ES 300h, GS 450h, LS 600h L, RX 450h). Mã: `src/linreg/features.py`.

## Hạn chế

Chỉ có năm 2014. Muốn kiểm tra kết luận trên dữ liệu mới hơn, nhóm tải bản nhiều năm từ Open Government Portal và chạy lại pipeline.
