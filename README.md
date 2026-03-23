# TULIP-MedML

Multi-label chest X-ray classification trên CheXpert (14 nhãn bệnh).  
Dự án nghiên cứu theo kế hoạch 6 tuần — 5 config cải tiến dần từ baseline đến mô hình đầy đủ.

---

## Tổng quan kiến trúc

```
Ảnh X-ray (448×448)
      │
      ▼
  ResNet-101 / Swin-T          ← backbone trích xuất features
  GlobalPool → (B, 2048)
      │
      │         Word embeddings (14, 300)   ← GloVe hoặc BiomedCLIP
      │                │
      │           GraphConvolution          ← học quan hệ giữa 14 nhãn
      │           300→1024→2048
      │                │
      └──── dot product ──── logits (B, 14)
                             │
                          BCELoss / UA-ASL
```

**Ý tưởng cốt lõi:** GCN học ma trận quan hệ giữa 14 bệnh (ví dụ Pneumonia thường đi kèm Consolidation) từ co-occurrence statistics, rồi dùng quan hệ đó để tạo ra 14 classifier vectors có "hiểu biết về nhau". Mỗi classifier vector dot product với image feature cho ra 1 logit.

---

## Cấu trúc repo

```
TULIP-MedML/
├── src/
│   ├── data/
│   │   ├── chexpert.py        Dataset class, xử lý label uncertain {-1,0,1}
│   │   └── gen_chexpert_data.py  Tạo word_vec.npy + adj.pkl (chạy 1 lần)
│   ├── models/
│   │   ├── gcn.py             GCNResnet: backbone + GCN + dot product
│   │   └── backbone.py        SwinT backbone (Tuần 2)
│   ├── loss/
│   │   └── ua_asl.py          UncertaintyAwareASL loss (Tuần 3)
│   ├── configs/
│   │   ├── test.yaml          Smoke test — 1 epoch, 2000 ảnh
│   │   ├── c1.yaml            C1: ResNet-101 + GloVe + BCE
│   │   ├── c2.yaml            C2: Swin-T + GloVe + BCE
│   │   ├── c3.yaml            C3: ResNet + BiomedCLIP init + BCE
│   │   ├── c4.yaml            C4: ResNet + GloVe + UA-ASL
│   │   └── c5.yaml            C5: Swin-T + BiomedCLIP + UA-ASL (TULIP-MedML)
│   ├── util.py                Gen_A, gen_adj, AveragePrecisionMeter, transforms
│   ├── engine.py              Training loop, callback pattern
│   ├── train.py               Entry point, đọc config yaml
│   └── evaluate.py            mAP, AUC-ROC, GradCAM
├── data/                      ← KHÔNG commit CSV; .npy và .pkl nhỏ thì commit
│   ├── chexpert_glove_word2vec.npy   Word vectors (14×300), random nếu không có GloVe
│   ├── chexpert_adj.pkl              Co-occurrence matrix (14×14)
│   └── README_data.md                Hướng dẫn tạo lại + link Google Drive
├── notebooks/
│   └── tulip-medml-train.ipynb     Notebook chạy trên Kaggle GPU T4
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Cài đặt

```bash
git clone https://github.com/PhuongThao-2005/TULIP-MedML.git
cd TULIP-MedML
pip install -r requirements.txt
```

---

## Data

Dataset CheXpert: https://www.kaggle.com/datasets/ashery/chexpert

**Tạo auxiliary files (chạy 1 lần):**
```bash
# Gen word vectors + adjacency matrix
python src/data/gen_chexpert_data.py \
    --csv /path/to/CheXpert/train.csv \
    --out_dir data/

# Stratified split 90/10
python src/data/make_splits.py \
    --root /path/to/CheXpert \
    --out_dir data/
```

File tạo ra:
- `data/chexpert_glove_word2vec.npy` — word embeddings (14, 300)
- `data/chexpert_adj.pkl` — co-occurrence matrix (14, 14)
- `data/train_small.csv` — 20% data đã stratified, dùng để thử nghiệm nhanh
- `data/val_small.csv`

**Dùng GloVe thật** (tốt hơn random vectors):
```bash
# Download glove.840B.300d.txt từ https://nlp.stanford.edu/projects/glove/
python src/data/gen_chexpert_data.py \
    --csv /path/to/train.csv \
    --glove /path/to/glove.840B.300d.txt \
    --out_dir data/
```

---

## Chạy trên Kaggle

Xem `tulip-medml-train.ipynb`

Tóm tắt:
```python
# Cell 1: Clone repo
!git clone https://github.com/PhuongThao-2005/TULIP-MedML.git
!pip install pyyaml timm -q
import sys; sys.path.insert(0, '/kaggle/working/TULIP-MedML')

# Cell 2: Train
!cd /kaggle/working/TULIP-MedML && python src/train.py \
    --config src/configs/c1.yaml
```

---

## Train local

```bash
# Smoke test nhanh — 2000 ảnh, 1 epoch (~2 phút)
python src/train.py --config src/configs/test.yaml --subset 2000

# Train C1 đầy đủ
python src/train.py --config src/configs/c1.yaml

# Resume từ checkpoint
python src/train.py --config src/configs/c1.yaml \
    --resume checkpoints/c1/checkpoint.pth.tar
```

---

## 5 config thực nghiệm

| Config | Backbone | Node init | Loss | Mục tiêu |
|--------|----------|-----------|------|-----------|
| C1 | ResNet-101 | GloVe random | BCE | Baseline |
| C2 | **Swin-T** | GloVe random | BCE | Visual feature tốt hơn |
| C3 | ResNet-101 | **BiomedCLIP** | BCE | Semantic init tốt hơn |
| C4 | ResNet-101 | GloVe random | **UA-ASL** | Xử lý uncertain đúng |
| C5 | Swin-T | BiomedCLIP | UA-ASL | Tất cả cải tiến cộng dồn |

**Ablation table mục tiêu:**

| Config | mAP | mean AUC | AUC-uncertain |
|--------|-----|----------|---------------|
| C1 | ? | ? | ? |
| C2 | ? | ? | ? |
| C3 | ? | ? | ? |
| C4 | ? | ? | ? |
| C5 | ? | ? | ? |

---

## Xử lý nhãn uncertain

CheXpert có 3 giá trị nhãn: `1` (positive), `0` (negative), `-1` (uncertain).

Uncertain nghĩa là bác sĩ không chắc chắn — không nên coi là negative cũng không phải positive.

**3 chiến lược đơn giản** (dùng cho C1–C3):
```yaml
uncertain: zeros   # -1 → negative (U-Zeros, default)
uncertain: ones    # -1 → positive (U-Ones)
uncertain: ignore  # -1 → 0, engine mask khỏi loss
```

**UA-ASL** (dùng cho C4–C5): giữ nguyên `-1`, loss function xử lý riêng:
```yaml
uncertain: ignore  # gán 0 để UA-ASL nhận diện được
loss:
  type:       ua_asl
  gamma_pos:  0      # ít penalize false negative
  gamma_neg:  4      # penalize mạnh false positive
  lambda_unc: 0.5    # weight cho uncertain (nhỏ hơn negative thuần)
```

---

## Implement C2: Swin-T backbone (Tuần 2)

Sửa `src/models/backbone.py`:
```python
import timm
import torch.nn as nn

class SwinTBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=pretrained,
            num_classes=0)           # bỏ head classification
        self.proj = nn.Linear(768, 2048)   # align với GCN output dim

        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std  = [0.229, 0.224, 0.225]

    def forward(self, x):
        feat = self.model(x)         # (B, 768)
        return self.proj(feat)       # (B, 2048)
```

Tạo `src/configs/c2.yaml` — chỉ thay `backbone: swin_t`, giữ nguyên phần còn lại.

---

## Implement C3: BiomedCLIP node init (Tuần 2)

```python
# src/data/gen_chexpert_data.py — thêm hàm này
import open_clip

LABEL_DESCRIPTIONS = {
    'No Finding':                  'no disease or abnormality found in chest x-ray',
    'Enlarged Cardiomediastinum':  'enlarged cardiac silhouette or widened mediastinum',
    'Cardiomegaly':                'enlarged heart cardiomegaly on chest x-ray',
    'Lung Opacity':                'opacity or haziness in the lung fields',
    'Lung Lesion':                 'focal lung lesion or nodule in the lung',
    'Edema':                       'pulmonary edema fluid in the lungs',
    'Consolidation':               'lung consolidation airspace disease',
    'Pneumonia':                   'pneumonia infection in the lung',
    'Atelectasis':                 'atelectasis collapsed lung or partial collapse',
    'Pneumothorax':                'pneumothorax air in the pleural space',
    'Pleural Effusion':            'pleural effusion fluid around the lung',
    'Pleural Other':               'other pleural abnormality or thickening',
    'Fracture':                    'rib fracture or bone fracture on chest x-ray',
    'Support Devices':             'medical support devices tubes lines on x-ray',
}

def build_biomedclip_vectors(out_path='data/chexpert_biomedclip_vec.npy'):
    model, _, _ = open_clip.create_model_and_transforms(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    tokenizer = open_clip.get_tokenizer(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    model.eval()

    texts  = [LABEL_DESCRIPTIONS[cls] for cls in CHEXPERT_CLASSES]
    tokens = tokenizer(texts)
    with torch.no_grad():
        vecs = model.encode_text(tokens)   # (14, 512)
    vecs = vecs.cpu().numpy().astype(np.float32)
    np.save(out_path, vecs)
    print(f'BiomedCLIP vectors saved: {out_path}  shape={vecs.shape}')
    return vecs
```

Config C3: thêm `gcn_in: 512` và `word_vec: data/chexpert_biomedclip_vec.npy`.  
Thêm `Linear(512, 300)` projection trước GCN trong `models/gcn.py`.

---

## Implement C4: UA-ASL loss (Tuần 3)

Xem `src/loss/ua_asl.py`. Công thức:

```
y = 1  → L_pos = -(1-p)^γ⁺ · log(p)
y = -1 → L_neg = -p^γ⁻ · log(1-p)
y = 0  → L_unc = -λu · p^γ⁻ · log(1-p)
```

Uncertain được penalize nhẹ hơn negative thuần (`λu < 1`) vì uncertain có thể thật sự là positive.

Grid search tốt nhất: `γ⁺∈{0,1,2}`, `γ⁻∈{2,3,4}`, `λu∈{0.3,0.5,0.7}` — chạy trên 20% data.

---

## Metrics

- **mAP** (mean Average Precision): metric chính, tính AUC của precision-recall curve mỗi class rồi average
- **mean AUC** (AUC-ROC): metric phụ, dùng để so sánh với paper khác
- **AUC-uncertain**: AUC tính riêng trên các sample có label uncertain — đánh giá trực tiếp hiệu quả xử lý uncertain

Xem `src/evaluate.py` để tính đầy đủ sau khi train xong.

---

## Phân công

| Thành viên | Trách nhiệm |
|---|---|
| A | backbone.py (C2), ablation analysis, GradCAM |
| B | ua_asl.py (C4), BiomedCLIP init (C3), grid search |
| C | train.py, evaluate.py, báo cáo kết quả |

---

## Ghi chú

- `data/` và `checkpoints/` không commit lên GitHub — dùng Google Drive chia sẻ trong nhóm
- Mỗi config C1–C5 làm trên branch riêng, merge vào `main` khi xong
- Checkpoint sau mỗi session Kaggle → download → upload Drive
- Word vectors hiện dùng random — thay GloVe thật sẽ cải thiện đáng kể C1–C2
