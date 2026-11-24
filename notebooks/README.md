# Notebooks

Đánh số theo thứ tự chạy, import code từ package `linreg` thay vì chép lại:

| Notebook | Nội dung |
|----------|----------|
| `01_eda.ipynb` | Phân phối biến mục tiêu, thiếu dữ liệu, tương quan, ngoại lệ, đa cộng tuyến (VIF) |
| `02_models.ipynb` | Huấn luyện, so sánh mô hình, phân tích phần dư, hệ số |
| `03_figures.ipynb` | Xuất hình cho báo cáo vào `figures/` |

Xoá output lớn trước khi commit (`jupyter nbconvert --clear-output --inplace notebooks/*.ipynb`) nếu output chứa dữ liệu thô.
