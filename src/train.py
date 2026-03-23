import argparse, yaml, os, sys, torch, torch.nn as nn

# Thêm root vào path để import src.*
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.chexpert import CheXpert, NUM_CLASSES
from src.models.gcn   import gcn_resnet101
from src.engine       import GCNMultiLabelMAPEngine

def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config',  required=True)
    p.add_argument('--subset',  type=int, default=None,
                   help='N ảnh đầu để test nhanh')
    p.add_argument('--data_root', default=None,
                   help='Override data root (dùng khi path Kaggle khác local)')
    args = p.parse_args()

    cfg = load_cfg(args.config)
    print(f"Config: {cfg['name']}")

    # Cho phép override root từ command line
    # → chạy local: --data_root /Users/me/chexpert
    # → chạy Kaggle: --data_root /kaggle/input/datasets/ashery/chexpert
    root = args.data_root or cfg['data']['root']

    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])

    train_ds = CheXpert(root=root,
                        csv_file=cfg['data']['train_csv'],
                        inp_name=cfg['data']['word_vec'],
                        uncertain=cfg['data']['uncertain'])
    val_ds   = CheXpert(root=root,
                        csv_file=cfg['data']['val_csv'],
                        inp_name=cfg['data']['word_vec'],
                        uncertain=cfg['data']['uncertain'])

    if args.subset:
        train_ds.df = train_ds.df.head(args.subset).reset_index(drop=True)
        val_ds.df   = val_ds.df.head(max(50, args.subset//9)).reset_index(drop=True)
        print(f'Subset: {len(train_ds)} train / {len(val_ds)} val')

    model = gcn_resnet101(
        num_classes=NUM_CLASSES, t=cfg['model']['t'],
        pretrained=cfg['model']['pretrained'],
        adj_file=cfg['data']['adj'], in_channel=cfg['model']['gcn_in'],
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(
        model.get_config_optim(cfg['train']['lr'], cfg['train']['lrp']),
        lr=cfg['train']['lr'],
        momentum=cfg['train']['momentum'],
        weight_decay=cfg['train']['weight_decay'],
    )

    os.makedirs(cfg['output']['save_dir'], exist_ok=True)
    state = {
        'batch_size'      : cfg['train']['batch_size'],
        'image_size'      : cfg['data']['img_size'],
        'max_epochs'      : cfg['train']['epochs'],
        'workers'         : cfg['train']['workers'],
        'epoch_step'      : cfg['train']['epoch_step'],
        'save_model_path' : cfg['output']['save_dir'],
        'print_freq'      : 100,
        'use_pb'          : True,
        'difficult_examples': False,
    }

    engine = GCNMultiLabelMAPEngine(state)
    best   = engine.learning(model, criterion, train_ds, val_ds, optimizer)
    print(f'\n[{cfg["name"]}] Best mAP = {best:.2f}')

if __name__ == '__main__':
    main()