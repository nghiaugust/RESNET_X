Dataset hiện tại dùng annotation:

- `0 = no_x`
- `1 = x_cancel`
- `2 = x_mark`

Ảnh dataset là `640x640`, nên cấu hình đã đổi `input_size` từ `[128, 512]` sang `[640, 640]`.

Mặc định pipeline dùng:

- CNN backbone: `ResNet18` pretrained ImageNet
- Số lớp output CNN: 3
- Feature cho SVM: 512 chiều
- SVM train trên feature của `train + val`
- Test riêng trên split `test`

Chạy từ thư mục `train_model/`:

```powershell
# Tạo virtual environment
python -m venv .venv

# Cho phép activate trong PowerShell hiện tại
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Kích hoạt môi trường
.\.venv\Scripts\Activate.ps1

# Cập nhật pip và cài thư viện
python -m pip install --upgrade pip
pip install -r requirements.txt

# Kiểm tra dataset visualization
python visualize_dataset.py --config config.yaml

# Chạy pipeline train ResNet18 + SVM
python run_pipeline.py --config config.yaml
```

Các config so sánh cũng đã cập nhật theo dataset 3 lớp 640x640:

```powershell
python run_pipeline.py --config config_resnet50.yaml
python run_pipeline.py --config config_convnext_tiny.yaml
```

Ghi chú feature:

- ResNet18 sinh feature 512 chiều
- ResNet50 sinh feature 2048 chiều
- ConvNeXt-Tiny sinh feature 768 chiều
