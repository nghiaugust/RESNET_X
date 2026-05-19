# ResNet18 + SVM cho dataset X-mark

Pipeline này dùng dataset 3 nhãn:

- `0`: `no_x`
- `1`: `x_cancel`
- `2`: `x_mark`

Ảnh dataset hiện tại là `640x640`, nên cấu hình mặc định dùng `input_size: [640, 640]`. Loader vẫn dùng resize-padding giữ tỉ lệ; với ảnh vuông 640x640 thì ảnh gần như được đưa vào đúng kích thước gốc.

Pipeline gồm 2 giai đoạn:

1. Fine-tune `ResNet18` pretrained ImageNet với Cross-Entropy.
2. Cắt lớp FC cuối, trích feature 512 chiều, rồi train `SVC(kernel="rbf")` bằng Grid Search trên `C` và `gamma`.

Nên chạy các lệnh bên dưới từ thư mục `train_model/` vì `config.yaml` đang dùng `dataset.root: dataset`.

## Cài đặt

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### GPU / CUDA setup

Config mac dinh hien tai dung `device: cuda`, nen training se dung NVIDIA GPU. Neu PyTorch dang la ban CPU-only, chuong trinh se dung lai va bao loi ro rang.

Kiem tra PyTorch co thay GPU khong:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

Neu ket qua la `False` hoac version co `+cpu`, cai lai PyTorch CUDA truoc khi cai cac package con lai:

```powershell
pip uninstall -y torch torchvision
pip install -r requirements-gpu-cu128.txt
pip install -r requirements.txt
```

Train ep GPU:

```powershell
python train_cnn.py --config config.yaml --device cuda
```

Neu muon chon GPU cu the:

```powershell
python train_cnn.py --config config.yaml --device cuda:0
```

## Xem tổng quan dataset

```powershell
python visualize_dataset.py --config config.yaml
```

Kết quả nằm trong `runs/dataset_overview/`: phân bố nhãn, phân bố kích thước ảnh, và lưới ảnh mẫu.

## Train ResNet18 CNN

```powershell
python train_cnn.py --config config.yaml
```

Output chính trong `runs/cnn_resnet18/`:

- `best_cnn.pt`, `last_cnn.pt`
- `history.csv`
- `training_curves.png`
- `cnn_val_confusion.png`, `cnn_test_confusion.png`
- `cnn_*_classification_report.txt`
- `cnn_*_roc_curve.png`, `cnn_*_pr_curve.png`
- `tensorboard/`

Xem TensorBoard:

```powershell
tensorboard --logdir runs/cnn_resnet18/tensorboard
```

## Train ResNet18 + SVM

Sau khi có `runs/cnn_resnet18/best_cnn.pt`:

```powershell
python train_svm.py --config config.yaml
```

Output chính trong `runs/svm_resnet18/`:

- `features.npz`
- `svm_model.joblib`
- `best_params.json`
- `grid_search_results.csv`
- `svm_grid_heatmap.png`
- `svm_test_confusion.png`
- `svm_test_roc_curve.png`
- `svm_test_pr_curve.png`
- `svm_test_classification_report.txt`
- `svm_test_predictions.csv`
- `svm_test_misclassified.png`

## Chạy toàn bộ pipeline

```powershell
python run_pipeline.py --config config.yaml
```

Nếu đã train CNN và chỉ muốn train lại SVM:

```powershell
python run_pipeline.py --config config.yaml --skip-cnn --force-extract
```

## Đánh giá lại model

CNN:

```powershell
python evaluate.py --config config.yaml --model-type cnn --split test --checkpoint runs/cnn_resnet18/best_cnn.pt
```

SVM:

```powershell
python evaluate.py --config config.yaml --model-type svm --split test --svm-model runs/svm_resnet18/svm_model.joblib --feature-cache runs/svm_resnet18/features.npz
```

## Config khác

Mặc định `config.yaml` dùng `model.name: resnet18`, output vào `runs/cnn_resnet18` và `runs/svm_resnet18`.

Các config so sánh vẫn đã được cập nhật cho dataset 3 lớp 640x640:

```powershell
python run_pipeline.py --config config_resnet50.yaml
python run_pipeline.py --config config_convnext_tiny.yaml
```

## Deploy ResNet18

Gói deploy nằm trong `deploy_resnet18/`, hỗ trợ:

- CNN thuần: `best_cnn.pt`
- CNN + SVM: `best_cnn.pt` và `svm_model.joblib`

Sau khi train lại với dataset 3 lớp, copy weights mới:

```powershell
Copy-Item runs\cnn_resnet18\best_cnn.pt deploy_resnet18\weights\best_cnn.pt
Copy-Item runs\svm_resnet18\svm_model.joblib deploy_resnet18\weights\svm_model.joblib
```

Chạy ResNet18 + SVM:

```powershell
python deploy_resnet18\predict.py --mode svm --input C:\path\to\image.jpg
```
