import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from data import PairedSliceDataset
from model import MDASGT, combined_loss


def metrics(logits, target):
    p, y = torch.sigmoid(logits) > .5, target > .5
    dims = (1, 2, 3); eps = 1e-6
    inter = (p & y).sum(dims).float(); union = (p | y).sum(dims).float()
    dice = ((2 * inter + eps) / (p.sum(dims) + y.sum(dims) + eps)).mean()
    iou = ((inter + eps) / (union + eps)).mean()
    return dice.item(), iou.item()


def main(a):
    torch.manual_seed(a.seed)
    ds = PairedSliceDataset(a.data, a.size, augment=True)
    n_val = max(1, int(len(ds) * a.val_fraction)); train_ds, val_ds = random_split(ds, [len(ds)-n_val, n_val])
    train_loader = DataLoader(train_ds, a.batch, shuffle=True); val_loader = DataLoader(val_ds, a.batch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MDASGT(base=a.base).to(device); opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    best = -1.; Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, a.epochs + 1):
        model.train(); total = 0.
        for x, y in train_loader:
            x, y = x.to(device), y.to(device); opt.zero_grad(set_to_none=True)
            loss = combined_loss(model(x), y); loss.backward(); opt.step(); total += loss.item()
        model.eval(); scores = []
        with torch.no_grad():
            for x, y in val_loader: scores.append(metrics(model(x.to(device)), y.to(device)))
        dice = sum(v[0] for v in scores) / len(scores); iou = sum(v[1] for v in scores) / len(scores)
        print(f"epoch={epoch:03d} loss={total/len(train_loader):.4f} val_dice={dice:.4f} val_iou={iou:.4f}")
        if dice > best:
            best = dice; torch.save({"model": model.state_dict(), "base": a.base, "size": a.size}, a.output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--data", required=True); p.add_argument("--output", default="checkpoints/best.pt")
    p.add_argument("--epochs", type=int, default=50); p.add_argument("--batch", type=int, default=8)
    p.add_argument("--size", type=int, default=256); p.add_argument("--base", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--val-fraction", type=float, default=.2); p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())

