import argparse
import os
import time

import torch
import torch.nn as nnx
from torchvision.models import efficientnet_b0
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.nn import BCEWithLogitsLoss
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
import tqdm

# Custom Dataset for paired image and label txt files
class PairedImageLabelDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_filename = self.image_files[idx]
        image_path = os.path.join(self.image_dir, img_filename)
        label_filename = os.path.splitext(img_filename)[0] + ".txt"
        label_path = os.path.join(self.label_dir, label_filename)

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        with open(label_path, 'r') as f:
            label = float(f.readline().strip())

        return image, torch.tensor([label], dtype=torch.float32)


def build_model():
    model = efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(in_features=1280, out_features=1)
    return model


def train(model, train_loader, valid_loader, device, args):
    criterion = BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    model.to(device)

    with open('loss_log.txt', 'w') as f:
        f.write('epoch,batch,loss,accuracy\n')

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(tqdm.tqdm(train_loader, desc=f"Train Epoch {epoch+1}")):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            batch_correct = (preds == labels).float().sum().item()
            batch_total = labels.numel()
            batch_acc = batch_correct / batch_total
            correct += batch_correct
            total += batch_total

            with open('loss_log.txt', 'a') as f:
                f.write(f"{epoch+1},{batch_idx+1},{loss.item():.6f},{batch_acc:.4f}\n")

        train_acc = correct / total
        avg_train_loss = total_loss / len(train_loader)
        scheduler.step()

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in tqdm.tqdm(valid_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == labels).float().sum().item()
                val_total += labels.numel()

        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(valid_loader)

        print(f"Epoch {epoch+1} Summary\ttrain_loss={avg_train_loss:.6f}\ttrain_acc={train_acc:.4f}\tval_loss={avg_val_loss:.6f}\tval_acc={val_acc:.4f}")

        with open('loss_log.txt', 'a') as f:
            f.write(f"Epoch {epoch+1} Summary\ttrain_loss={avg_train_loss:.6f}\ttrain_acc={train_acc:.4f}\tval_loss={avg_val_loss:.6f}\tval_acc={val_acc:.4f}\n")

        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/efficientnet_epoch{epoch+1}.pth")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_image_dir', type=str, required=True)
    parser.add_argument('--train_label_dir', type=str, required=True)
    parser.add_argument('--valid_image_dir', type=str, required=True)
    parser.add_argument('--valid_label_dir', type=str, required=True)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor()
    ])

    train_dataset = PairedImageLabelDataset(args.train_image_dir, args.train_label_dir, transform=transform)
    valid_dataset = PairedImageLabelDataset(args.valid_image_dir, args.valid_label_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model()
    train(model, train_loader, valid_loader, device, args)


if __name__ == '__main__':
    main()
