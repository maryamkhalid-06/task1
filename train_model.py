"""
Train SegNet model on Heart data
Results saved to data/training_metrics.json and model to data/model_best.pth
"""
import sys
import os
import json
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
sys.path.insert(0, '.')
from segnet_model import get_model

print('='*50)
print('TRAINING SEGNET FOR HEART SEGMENTATION')
print('='*50)

# Load data
img_path = 'data/heart/images/heart.nii'
msk_path = 'data/heart/masks/heartseg.nii'

print(f'Loading {img_path}...')
image = nib.load(img_path).get_fdata().astype(np.float32)
mask = nib.load(msk_path).get_fdata().astype(np.int64)
image = (image - image.min()) / (image.max() - image.min() + 1e-7)
mask = (mask > 0).astype(np.int64)
print(f'Image: {image.shape}, Mask unique: {np.unique(mask)}')

# Create model
model = get_model('segnet', 1, 2)
print(f'Model: SegNet with {model.get_num_params():,} params')

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training settings
epochs = 30
patches_per_epoch = 100
patch_size = 48
results = []
best_dice = 0

print(f'Training for {epochs} epochs...')
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    
    for _ in range(patches_per_epoch):
        # Get foreground-biased patch
        fg = np.where(mask > 0)
        if len(fg[0]) > 0 and np.random.random() > 0.3:
            idx = np.random.randint(len(fg[0]))
            cz, cy, cx = fg[0][idx], fg[1][idx], fg[2][idx]
            z = max(0, min(cz - patch_size//2, image.shape[0] - patch_size))
            y = max(0, min(cy - patch_size//2, image.shape[1] - patch_size))
            x = max(0, min(cx - patch_size//2, image.shape[2] - patch_size))
        else:
            z = np.random.randint(0, max(1, image.shape[0] - patch_size))
            y = np.random.randint(0, max(1, image.shape[1] - patch_size))
            x = np.random.randint(0, max(1, image.shape[2] - patch_size))
        
        img = image[z:z+patch_size, y:y+patch_size, x:x+patch_size]
        msk = mask[z:z+patch_size, y:y+patch_size, x:x+patch_size]
        
        inp = torch.from_numpy(img.copy()).unsqueeze(0).unsqueeze(0).float()
        tgt = torch.from_numpy(msk.copy()).unsqueeze(0).long()
        
        optimizer.zero_grad()
        out = model(inp)
        loss = criterion(out, tgt)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    # Validation
    model.eval()
    tp, fp, fn, tn = 0, 0, 0, 0
    with torch.no_grad():
        for _ in range(15):
            z = np.random.randint(0, max(1, image.shape[0] - patch_size))
            y = np.random.randint(0, max(1, image.shape[1] - patch_size))
            x = np.random.randint(0, max(1, image.shape[2] - patch_size))
            
            inp = torch.from_numpy(image[z:z+patch_size, y:y+patch_size, x:x+patch_size].copy()).unsqueeze(0).unsqueeze(0).float()
            tgt = mask[z:z+patch_size, y:y+patch_size, x:x+patch_size]
            
            out = model(inp)
            pred = (torch.argmax(out, dim=1).numpy()[0] > 0)
            
            tp += np.sum(pred & (tgt > 0))
            fp += np.sum(pred & (tgt == 0))
            fn += np.sum((~pred) & (tgt > 0))
            tn += np.sum((~pred) & (tgt == 0))
    
    dice = float((2*tp) / (2*tp + fp + fn + 1e-7))
    iou = float(tp / (tp + fp + fn + 1e-7))
    acc = float((tp + tn) / (tp + tn + fp + fn + 1e-7))
    avg_loss = epoch_loss / patches_per_epoch
    
    results.append({
        'epoch': epoch + 1,
        'loss': avg_loss,
        'dice': dice,
        'iou': iou,
        'accuracy': acc
    })
    
    # Save best model
    if dice > best_dice:
        best_dice = dice
        torch.save(model.state_dict(), 'data/model_best.pth')
    
    print(f'Epoch {epoch+1:2d}/{epochs}: Loss={avg_loss:.4f} Dice={dice:.4f} IoU={iou:.4f} Acc={acc:.4f}')

# Save metrics
with open('data/training_metrics.json', 'w') as f:
    json.dump(results, f, indent=2)

# Generate and save predictions
print('\nGenerating predictions for visualization...')
d, h, w = image.shape
predictions = np.zeros((d, h, w), dtype=np.int64)
model.eval()

with torch.no_grad():
    for z in range(0, d, 32):
        for y in range(0, h, 32):
            for x in range(0, w, 32):
                z1, z2 = z, min(z + 48, d)
                y1, y2 = y, min(y + 48, h)
                x1, x2 = x, min(x + 48, w)
                
                pz, py, px = z2-z1, y2-y1, x2-x1
                if pz < 16 or py < 16 or px < 16:
                    continue
                
                padded = np.zeros((48, 48, 48), dtype=np.float32)
                padded[:pz, :py, :px] = image[z1:z2, y1:y2, x1:x2]
                
                inp = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0).float()
                out = model(inp)
                pred = torch.argmax(out, dim=1).numpy()[0]
                
                predictions[z1:z2, y1:y2, x1:x2] = np.maximum(
                    predictions[z1:z2, y1:y2, x1:x2],
                    pred[:pz, :py, :px]
                )

# Save predictions as NIfTI
pred_nii = nib.Nifti1Image(predictions.astype(np.float32), np.eye(4))
nib.save(pred_nii, 'data/predictions.nii.gz')

print('='*50)
print('TRAINING COMPLETE!')
print(f'Best Dice: {best_dice:.4f}')
print(f'Model: data/model_best.pth')
print(f'Metrics: data/training_metrics.json')
print(f'Predictions: data/predictions.nii.gz')
print('='*50)
