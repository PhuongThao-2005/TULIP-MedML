# TULIP-MedML

Dự án phân loại đa nhãn ảnh X-quang ngực (Chest X-ray) trên CheXpert, tập trung vào:
- mô hình GCN để học quan hệ giữa các bệnh lý,
- so sánh với baseline (CheXNet, ADD-GCN),
- xử lý nhãn không chắc chắn (`-1`) bằng hàm mất mát `UA-ASL`.

---

## 1) Mục tiêu và phạm vi dự án

### Mục tiêu chính
- Huấn luyện mô hình multi-label cho 14 nhãn bệnh của CheXpert.
- Cải thiện chất lượng bằng 3 hướng:
  - backbone mạnh hơn (ResNet -> Swin),
  - khởi tạo semantic tốt hơn (GloVe/BioMedCLIP),
  - loss phù hợp dữ liệu y khoa có uncertain label (`UA-ASL`).

### Nguồn tham khảo và baseline gốc
- Baseline gốc của nhánh GCN trong dự án này là **ML-GCN**.
- Source code tham khảo chính: [ML-GCN (megvii-research)](https://github.com/megvii-research/ML-GCN).
- Các phần tham khảo gồm:
  - ý tưởng kiến trúc multi-label dùng Graph Convolution để mô hình hóa quan hệ giữa nhãn,
  - cách biểu diễn classifier theo embedding nhãn và tính logits theo hướng image feature x label node feature,
  - quy trình huấn luyện/evaluate cho bài toán multi-label theo tinh thần ML-GCN.
- Code trong repo này đã được điều chỉnh để phù hợp bài toán X-quang ngực (CheXpert), thêm baseline đối chiếu (CheXNet, ADD-GCN), mở rộng backbone (Swin) và thêm loss `UA-ASL` cho uncertain labels.

---

## 2) Kiến trúc tổng quan (nhánh GCN)

Luồng chính:
1. Ảnh X-ray đi qua backbone (`resnet101`, `swin_t`, `swin_b`) để lấy vector ảnh.
2. Word embedding của 14 nhãn bệnh đi qua GCN để học quan hệ đồng xuất hiện.
3. Dot product giữa feature ảnh và classifier vector của từng nhãn -> logits đa nhãn.
4. Tối ưu bằng `BCEWithLogitsLoss` hoặc `UncertaintyAwareASL`.

Ý tưởng trọng tâm: không coi các nhãn là độc lập tuyệt đối; GCN giúp mô hình tận dụng tương quan bệnh lý (ví dụ các dấu hiệu thường xuất hiện cùng nhau).

---

## 3) Cấu trúc repository và chức năng từng file

```text
TULIP-MedML/
├── src/                             # Toàn bộ mã nguồn chính
│   ├── train.py                     # Entry point train GCN (C1-C5), auto-resume checkpoint
│   ├── engine.py                    # Train/val loop, hook callbacks, save best model, logging
│   ├── evaluate.py                  # Tính mAP, mean AUC, uncertain AUC + in bang per-class
│   ├── util.py                      # Các hàm util (transform, meter, cac helper cho GCN)
│   ├── grid_search_c5.py            # Grid search siêu tham số cho cấu hình C5
│   │
│   ├── models/                      # Định nghĩa các model
│   │   ├── gcn.py                   # Model GCN chính: gcn_resnet101 / gcn_swin_t / gcn_swin_b
│   │   ├── backbone.py              # Backbone Swin + helper tạo backbone
│   │   ├── chexnet.py               # Baseline CheXNet (DenseNet-style classifier)
│   │   └── addgcn.py                # Baseline ADD-GCN
│   │
│   ├── data/                        # Dataset và script tạo dữ liệu phụ trợ
│   │   ├── chexpert.py              # Dataset CheXpert: đọc CSV, load ảnh, xử lý uncertain label
│   │   └── gen_chexpert_data.py     # Tạo word_vec (GloVe/BioMedCLIP) va adjacency matrix
│   │
│   ├── loss/
│   │   └── ua_asl.py                # Uncertainty-Aware ASL loss cho nhãn uncertain (-1)
│   │
│   ├── baselines/                   # Script train baseline để đối chiếu
│   │   ├── train_chexnet.py         # Train/eval baseline CheXNet
│   │   └── train_addgcn.py          # Train/eval baseline ADD-GCN
│   │
│   └── configs/                     # Các cấu hình huấn luyện
│       ├── c1.yaml                  # C1: Baseline ML-GCN
│       ├── c2.yaml                  # C2: Thay backbone (Swin)
│       ├── c3.yaml                  # C3: Node init BioMedCLIP
│       ├── c4.yaml                  # C4: Thay loss UA-ASL
│       ├── c5.yaml                  # C5: Kết hợp cải tiến
│       ├── c5_tulip.yaml            # Biến thể C5 dùng cho grid-search
│       ├── chexnet_baseline.yaml    # Config baseline CheXNet
│       └── addgcn_baseline.yaml     # Config baseline ADD-GCN
│
├── data/                            # CSV split + các file .npy/.pkl dùng cho train
├── notebooks/
│   └── tulip-medml-train.ipynb      # Notebook chính sử dụng để train các mô hình
│   └── c5-grid-search.ipynb         # Notebook hỗ trợ thử nghiệm grid-search
|   └── test_all_models_c1_c5_addgcn.ipynb # Notebook chạy test set cho tất cả model
├── requirements.txt                
└── README.md                        
```

---

## 4) Data yêu cầu và file đầu vào

Dataset nguồn:
- [CheXpert trên Kaggle](https://www.kaggle.com/datasets/ashery/chexpert)

Đường dẫn phổ biến trong config hiện tại:
- Root dataset: `/kaggle/input/datasets/ashery/chexpert`
- Root repo khi clone trên Kaggle: `/kaggle/working/TULIP-MedML`

Các file cần có trong `data/` cho nhánh GCN:
- `chexpert_adj.pkl` (ma trận quan hệ nhãn)
- `chexpert_glove_word2vec.npy` (cho C1/C2/C4)
- `chexpert_biomedclip_vec.npy` (cho C3/C5)
- CSV split:
  - `train_small_v2.csv` hoặc `train_small_v3.csv`
  - `valid.csv`
  - `val_uncertain.csv`

---

## 5) Hướng dẫn chạy chi tiết trên Kaggle Notebook

## 5.1 Tạo notebook mới
- Mở Kaggle Notebook mới.
- Bật GPU: `Notebook settings -> Accelerator -> GPU`.

## 5.2 Clone repo và cài thư viện

```bash
!git clone https://github.com/PhuongThao-2005/TULIP-MedML.git
%cd /kaggle/working/TULIP-MedML
!pip install -q -r requirements.txt
```

## 5.3 Chạy train chính thức (GCN)

Ví dụ:

```bash
!python src/train.py --config src/configs/c1.yaml
!python src/train.py --config src/configs/c5.yaml
```

Nếu muốn override đường dẫn dataset:

```bash
!python src/train.py \
  --config src/configs/c1.yaml \
  --data_root /kaggle/input/datasets/ashery/chexpert
```

Lưu ý:
- Script sẽ auto-resume từ checkpoint mới nhất trong `output.save_dir`.
- Checkpoint và log mặc định lưu ở `/kaggle/working/checkpoints/...` và `/kaggle/working/logs/...`.

## 5.4 Chạy baseline để so sánh

CheXNet:

```bash
!python src/baselines/train_chexnet.py \
  --config src/configs/chexnet_baseline.yaml
```

ADD-GCN:

```bash
!python src/baselines/train_addgcn.py \
  --config src/configs/addgcn_baseline.yaml
```

---

## 6) Ý nghĩa các config C1 -> C5

- `c1.yaml`: baseline GCN với `resnet101` + GloVe + BCE.
- `c2.yaml`: thay backbone sang Swin để cải thiện biểu diễn ảnh.
- `c3.yaml`: dùng BioMedCLIP vector cho node/class embeddings.
- `c4.yaml`: chuyển loss sang `ua_asl` để xử lý uncertain tốt hơn.
- `c5.yaml`: kết hợp các cải tiến chính thành phiên bản mạnh hơn.
- `c5_tulip.yaml`: biến thể C5 mới hơn, có thêm metadata hỗ trợ grid-search.
- `test.yaml`: cấu hình nhẹ để smoke test nhanh.

---

## 7) Metrics và cách đọc kết quả

Các metric được tính trong `src/evaluate.py`:

- `mAP`: trung bình Average Precision theo từng class  
  -> metric chính cho multi-label.
- `mean AUC`: trung bình ROC-AUC theo class đủ điều kiện.
- `uncertain AUC`: AUC trên tập có uncertain label  
  -> dùng để đánh giá trực tiếp khả năng xử lý `-1`.

Kết quả in ra gồm:
- bảng per-class (`AP`, `AUC`, `Unc_AUC`),
- dòng `Mean` tổng hợp.

---

## 8) Hướng dẫn cách chạy

1. Xác nhận có đầy đủ file trong `data/`.
2. Chạy config chính (C1/C5).
3. Theo dõi checkpoint và log sau mỗi epoch.
4. Chạy baseline để có mốc đối chiếu.
5. Tổng hợp metric cuối cùng bằng output của `evaluate.py`.

---

## 9) Tóm tắt nhanh

  1) clone repo + `pip install -r requirements.txt`,
  2) chạy `src/train.py --config src/configs/c5.yaml`.

Chỉ cần theo đúng 2 bước trên là có thể train và lấy metric.

Ngoài ra, TULIP-MedML có web xây dựng để visualize kết quả đầu ra của baseline và TULIP-MedML: [TULIP-web](https://github.com/vandimmi/TULIPMedML-web)