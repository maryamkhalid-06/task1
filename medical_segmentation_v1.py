"""
Medical Image Segmentation with SegNet3D
=========================================
Fast and efficient 3D segmentation using encoder-decoder architecture with pooling indices.

Features:
- SegNet3D architecture (faster than UNet)
- Local NIfTI data loading (no downloads required)
- Comprehensive evaluation metrics (Dice, IoU, Accuracy, Sensitivity, Specificity)
- CPU-optimized training with progress logging
- Model checkpointing and easy reuse
"""

import os
import sys
import subprocess
from typing import List, Dict, Optional, Tuple
import json

# --- Environment Check & Dependency Installation ---
def is_venv():
    return (hasattr(sys, 'real_prefix') or 
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))

def install_dependencies():
    if not is_venv():
        print("⚠️  WARNING: You are NOT running in a Virtual Environment.")
        print("Recommendation: Run '.\\setup_venv.ps1' first to create a dedicated environment.")
    
    packages = [
        "torch",
        "nibabel",
        "numpy",
        "scikit-image",
        "pyvista",
        "matplotlib",
        "tqdm"
    ]
    print("📦 Checking/Installing dependencies...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-q", package],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"⚠️  Failed to install {package}: {e}")

# Call installation
install_dependencies()

# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pyvista as pv
from skimage import measure

# Import our SegNet model
from segnet_model import SegNet3D, LightSegNet3D


# =============================================================================
# Evaluation Metrics Class
# =============================================================================
class SegmentationMetrics:
    """
    Comprehensive evaluation metrics for medical image segmentation.
    
    Computes:
    - Dice Score (DSC) - Primary segmentation metric
    - IoU / Jaccard Index - Overlap measure
    - Accuracy - Overall pixel accuracy  
    - Sensitivity (Recall) - True positive rate
    - Specificity - True negative rate
    - Precision - Positive predictive value
    """
    
    def __init__(self, num_classes: int = 2, include_background: bool = False):
        self.num_classes = num_classes
        self.include_background = include_background
        self.reset()
    
    def reset(self):
        """Reset all accumulators"""
        self.tp = np.zeros(self.num_classes)  # True positives
        self.fp = np.zeros(self.num_classes)  # False positives
        self.fn = np.zeros(self.num_classes)  # False negatives
        self.tn = np.zeros(self.num_classes)  # True negatives
    
    def update(self, pred: np.ndarray, target: np.ndarray):
        """
        Update metrics with new prediction/target pair.
        
        Args:
            pred: Predicted labels (H, W, D) or (B, H, W, D)
            target: Ground truth labels (H, W, D) or (B, H, W, D)
        """
        pred = pred.flatten()
        target = target.flatten()
        
        for c in range(self.num_classes):
            pred_c = (pred == c)
            target_c = (target == c)
            
            self.tp[c] += np.sum(pred_c & target_c)
            self.fp[c] += np.sum(pred_c & ~target_c)
            self.fn[c] += np.sum(~pred_c & target_c)
            self.tn[c] += np.sum(~pred_c & ~target_c)
    
    def dice_score(self) -> float:
        """Compute mean Dice Score across classes"""
        start_idx = 0 if self.include_background else 1
        dice_per_class = (2 * self.tp + 1e-7) / (2 * self.tp + self.fp + self.fn + 1e-7)
        return float(np.mean(dice_per_class[start_idx:]))
    
    def iou_score(self) -> float:
        """Compute mean IoU (Jaccard Index) across classes"""
        start_idx = 0 if self.include_background else 1
        iou_per_class = (self.tp + 1e-7) / (self.tp + self.fp + self.fn + 1e-7)
        return float(np.mean(iou_per_class[start_idx:]))
    
    def accuracy(self) -> float:
        """Compute overall accuracy"""
        total_correct = np.sum(self.tp)
        total_samples = np.sum(self.tp + self.fp + self.fn + self.tn) / self.num_classes
        return float(total_correct / (total_samples + 1e-7))
    
    def sensitivity(self) -> float:
        """Compute mean sensitivity (recall / true positive rate)"""
        start_idx = 0 if self.include_background else 1
        sens_per_class = (self.tp + 1e-7) / (self.tp + self.fn + 1e-7)
        return float(np.mean(sens_per_class[start_idx:]))
    
    def specificity(self) -> float:
        """Compute mean specificity (true negative rate)"""
        start_idx = 0 if self.include_background else 1
        spec_per_class = (self.tn + 1e-7) / (self.tn + self.fp + 1e-7)
        return float(np.mean(spec_per_class[start_idx:]))
    
    def precision(self) -> float:
        """Compute mean precision (positive predictive value)"""
        start_idx = 0 if self.include_background else 1
        prec_per_class = (self.tp + 1e-7) / (self.tp + self.fp + 1e-7)
        return float(np.mean(prec_per_class[start_idx:]))
    
    def get_all_metrics(self) -> Dict[str, float]:
        """Return all metrics as a dictionary"""
        return {
            "dice": self.dice_score(),
            "iou": self.iou_score(),
            "accuracy": self.accuracy(),
            "sensitivity": self.sensitivity(),
            "specificity": self.specificity(),
            "precision": self.precision()
        }
    
    def print_metrics(self, prefix: str = ""):
        """Print all metrics in a formatted way"""
        metrics = self.get_all_metrics()
        print(f"\n{'='*50}")
        print(f"{prefix} Evaluation Metrics")
        print(f"{'='*50}")
        print(f"  📊 Dice Score:   {metrics['dice']:.4f}")
        print(f"  📊 IoU (Jaccard):{metrics['iou']:.4f}")
        print(f"  📊 Accuracy:     {metrics['accuracy']:.4f}")
        print(f"  📊 Sensitivity:  {metrics['sensitivity']:.4f}")
        print(f"  📊 Specificity:  {metrics['specificity']:.4f}")
        print(f"  📊 Precision:    {metrics['precision']:.4f}")
        print(f"{'='*50}\n")


# =============================================================================
# Dataset for Local NIfTI Files
# =============================================================================
class NIfTIDataset(Dataset):
    """
    Dataset for loading local NIfTI files.
    
    Args:
        image_path: Path to CT/MRI volume (.nii or .nii.gz)
        mask_path: Path to segmentation mask
        patch_size: Size of 3D patches to extract
        num_patches: Number of patches per epoch
        augment: Whether to apply data augmentation
    """
    
    def __init__(self, 
                 image_path: str,
                 mask_path: str,
                 patch_size: Tuple[int, int, int] = (64, 64, 64),
                 num_patches: int = 20,
                 augment: bool = True):
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.augment = augment
        
        # Load NIfTI files
        print(f"📂 Loading image: {image_path}")
        img_nii = nib.load(image_path)
        self.image = img_nii.get_fdata().astype(np.float32)
        
        print(f"📂 Loading mask: {mask_path}")
        mask_nii = nib.load(mask_path)
        self.mask = mask_nii.get_fdata().astype(np.int64)
        
        # Normalize image to [0, 1]
        self.image = (self.image - self.image.min()) / (self.image.max() - self.image.min() + 1e-7)
        
        # Convert mask to binary if it has multiple labels
        unique_labels = np.unique(self.mask)
        print(f"  📌 Image shape: {self.image.shape}")
        print(f"  📌 Mask labels: {unique_labels}")
        
        # If mask has values > 1, convert to binary (foreground/background)
        if len(unique_labels) > 2:
            self.mask = (self.mask > 0).astype(np.int64)
            self.num_classes = 2
        else:
            self.num_classes = len(unique_labels)
        
        print(f"  ✓ Dataset ready with {self.num_classes} classes")
    
    def __len__(self):
        return self.num_patches
    
    def _random_crop(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract a random patch from the volume"""
        d, h, w = self.image.shape
        pd, ph, pw = self.patch_size
        
        # Ensure we can extract a patch
        pd, ph, pw = min(pd, d), min(ph, h), min(pw, w)
        
        # Random starting point
        z = np.random.randint(0, max(1, d - pd + 1))
        y = np.random.randint(0, max(1, h - ph + 1))
        x = np.random.randint(0, max(1, w - pw + 1))
        
        img_patch = self.image[z:z+pd, y:y+ph, x:x+pw]
        mask_patch = self.mask[z:z+pd, y:y+ph, x:x+pw]
        
        return img_patch, mask_patch
    
    def _augment(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random augmentations"""
        # Random flip along each axis
        for axis in range(3):
            if np.random.random() > 0.5:
                img = np.flip(img, axis=axis)
                mask = np.flip(mask, axis=axis)
        
        # Random intensity shift
        if np.random.random() > 0.5:
            shift = np.random.uniform(-0.1, 0.1)
            img = np.clip(img + shift, 0, 1)
        
        return img.copy(), mask.copy()
    
    def __getitem__(self, idx):
        img_patch, mask_patch = self._random_crop()
        
        if self.augment:
            img_patch, mask_patch = self._augment(img_patch, mask_patch)
        
        # Add channel dimension and convert to tensor
        img_tensor = torch.from_numpy(img_patch).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask_patch).long()
        
        return {"image": img_tensor, "label": mask_tensor}


# =============================================================================
# Main Application Class
# =============================================================================
class MedicalSegmentationApp:
    """
    Medical Image Segmentation Application with SegNet3D
    
    Features:
    - Fast SegNet3D architecture
    - Local NIfTI data loading
    - Comprehensive evaluation metrics
    - 3D visualization
    """
    
    def __init__(self, 
                 root_dir: str = "./data",
                 model_type: str = "light",  # "segnet" or "light"
                 num_classes: int = 2):
        self.root_dir = root_dir
        self.model_type = model_type
        self.num_classes = num_classes
        self.device = torch.device("cpu")
        
        print(f"\n{'='*60}")
        print("🏥 Medical Image Segmentation with SegNet3D")
        print(f"{'='*60}")
        print(f"📁 Data directory: {root_dir}")
        print(f"🖥️  Device: {self.device}")
        print(f"🧠 Model type: {model_type}")
        
        # Paths
        self.checkpoint_path = os.path.join(root_dir, "segnet_checkpoint.pth")
        self.metrics_path = os.path.join(root_dir, "training_metrics.json")
        
        # Initialize metrics
        self.metrics = SegmentationMetrics(num_classes=num_classes)
    
    def prepare_data(self, 
                     image_path: Optional[str] = None,
                     mask_path: Optional[str] = None,
                     organ: str = "heart"):
        """
        Prepare data loaders from local NIfTI files.
        
        Args:
            image_path: Path to image volume (optional, auto-detected if None)
            mask_path: Path to segmentation mask (optional, auto-detected if None)
            organ: Organ name for auto-detection (default: "heart")
        """
        # Auto-detect paths if not provided
        if image_path is None:
            # Try heart data first
            heart_img = os.path.join(self.root_dir, "heart", "images", "heart.nii")
            ct_img = os.path.join(self.root_dir, "ct.nii.gz")
            
            if os.path.exists(heart_img):
                image_path = heart_img
            elif os.path.exists(ct_img):
                image_path = ct_img
            else:
                raise FileNotFoundError("No image file found! Please provide image_path.")
        
        if mask_path is None:
            # Try heart mask first
            heart_mask = os.path.join(self.root_dir, "heart", "masks", "heartseg.nii")
            organ_mask = os.path.join(self.root_dir, "segmentations", f"{organ}.nii.gz")
            
            if os.path.exists(heart_mask):
                mask_path = heart_mask
            elif os.path.exists(organ_mask):
                mask_path = organ_mask
            else:
                raise FileNotFoundError(f"No mask file found for {organ}!")
        
        print(f"\n📊 Loading Dataset")
        print(f"  Image: {image_path}")
        print(f"  Mask: {mask_path}")
        
        # Create datasets (80% train, 20% validation via different patches)
        self.train_ds = NIfTIDataset(
            image_path=image_path,
            mask_path=mask_path,
            patch_size=(64, 64, 64),
            num_patches=40,  # Training patches per epoch
            augment=True
        )
        
        self.val_ds = NIfTIDataset(
            image_path=image_path,
            mask_path=mask_path,
            patch_size=(64, 64, 64),
            num_patches=10,  # Validation patches
            augment=False
        )
        
        self.train_loader = DataLoader(self.train_ds, batch_size=2, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(self.val_ds, batch_size=1, shuffle=False, num_workers=0)
        
        # Store full volumes for visualization
        self.full_image = self.val_ds.image
        self.full_mask = self.val_ds.mask
    
    def get_model(self) -> nn.Module:
        """Create and return the SegNet model"""
        if self.model_type == "light":
            model = LightSegNet3D(
                in_channels=1,
                out_channels=self.num_classes,
                init_features=16
            )
        else:
            model = SegNet3D(
                in_channels=1,
                out_channels=self.num_classes,
                init_features=32
            )
        
        model = model.to(self.device)
        print(f"\n🧠 Model: {model.__class__.__name__}")
        print(f"   Parameters: {model.get_num_params():,}")
        
        return model
    
    def train_and_evaluate(self, 
                           epochs: int = 5,
                           learning_rate: float = 1e-3,
                           save_checkpoint: bool = True) -> nn.Module:
        """
        Train the model and evaluate with comprehensive metrics.
        
        Args:
            epochs: Number of training epochs
            learning_rate: Initial learning rate  
            save_checkpoint: Whether to save model checkpoint
            
        Returns:
            Trained model
        """
        model = self.get_model()
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )
        
        best_dice = 0.0
        training_history = []
        
        print(f"\n🚀 Starting Training for {epochs} epochs...")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Batch size: 2")
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            epoch_loss = 0.0
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                inputs = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            avg_loss = epoch_loss / len(self.train_loader)
            
            # Validation phase
            model.eval()
            self.metrics.reset()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in self.val_loader:
                    inputs = batch["image"].to(self.device)
                    labels = batch["label"].to(self.device)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    # Get predictions
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    targets = labels.cpu().numpy()
                    
                    self.metrics.update(preds, targets)
            
            avg_val_loss = val_loss / len(self.val_loader)
            scheduler.step(avg_val_loss)
            
            # Get all metrics
            metrics_dict = self.metrics.get_all_metrics()
            
            # Log progress
            print(f"\n📈 Epoch {epoch+1}/{epochs}")
            print(f"   Train Loss: {avg_loss:.4f}")
            print(f"   Val Loss:   {avg_val_loss:.4f}")
            print(f"   Dice:       {metrics_dict['dice']:.4f}")
            print(f"   IoU:        {metrics_dict['iou']:.4f}")
            print(f"   Accuracy:   {metrics_dict['accuracy']:.4f}")
            
            # Save training history
            training_history.append({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "val_loss": avg_val_loss,
                **metrics_dict
            })
            
            # Save best model
            if metrics_dict['dice'] > best_dice:
                best_dice = metrics_dict['dice']
                if save_checkpoint:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'dice': best_dice,
                        'metrics': metrics_dict
                    }, self.checkpoint_path)
                    print(f"   ✓ Saved best model (Dice: {best_dice:.4f})")
        
        # Final evaluation
        self.metrics.print_metrics("Final Validation")
        
        # Save training history
        with open(self.metrics_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        print(f"📊 Training history saved to: {self.metrics_path}")
        
        # Generate full volume prediction for visualization
        self._generate_full_prediction(model)
        
        return model
    
    def _generate_full_prediction(self, model: nn.Module):
        """Generate prediction on full volume for visualization"""
        print("\n🎨 Generating full volume prediction...")
        
        model.eval()
        
        # Use sliding window for memory efficiency
        d, h, w = self.full_image.shape
        patch_size = 64
        stride = 48  # Overlap for smoother predictions
        
        # Initialize prediction volume
        pred_volume = np.zeros((self.num_classes, d, h, w), dtype=np.float32)
        count_volume = np.zeros((d, h, w), dtype=np.float32)
        
        with torch.no_grad():
            for z in range(0, d - patch_size + 1, stride):
                for y in range(0, h - patch_size + 1, stride):
                    for x in range(0, w - patch_size + 1, stride):
                        # Extract patch
                        patch = self.full_image[z:z+patch_size, y:y+patch_size, x:x+patch_size]
                        patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float()
                        patch_tensor = patch_tensor.to(self.device)
                        
                        # Predict
                        output = model(patch_tensor)
                        output = F.softmax(output, dim=1)
                        output_np = output.cpu().numpy()[0]
                        
                        # Accumulate predictions
                        pred_volume[:, z:z+patch_size, y:y+patch_size, x:x+patch_size] += output_np
                        count_volume[z:z+patch_size, y:y+patch_size, x:x+patch_size] += 1
        
        # Average overlapping predictions
        count_volume = np.maximum(count_volume, 1)
        for c in range(self.num_classes):
            pred_volume[c] /= count_volume
        
        # Get final prediction
        self.viz_pred = np.argmax(pred_volume, axis=0)
        self.viz_label = self.full_mask
        self.viz_image = self.full_image
        
        print("   ✓ Full volume prediction complete")
    
    def load_checkpoint(self, model: nn.Module) -> nn.Module:
        """Load model from checkpoint"""
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded checkpoint (Dice: {checkpoint.get('dice', 'N/A'):.4f})")
        return model
    
    def launch_3d_viewer(self):
        """Launch interactive 3D visualization"""
        print("\n🎨 Launching 3D Viewer...")
        
        plotter = pv.Plotter(title="SegNet3D - Medical Segmentation Viewer")
        plotter.set_background("black")
        
        def create_mesh(volume, label_id, color, opacity, name):
            binary_vol = (volume == label_id).astype(np.float32)
            if np.sum(binary_vol) < 100:
                print(f"  ⚠️  Skipping {name} (too few voxels)")
                return None
            
            try:
                verts, faces, normals, values = measure.marching_cubes(binary_vol, level=0.5)
                pv_faces = np.column_stack((np.full(len(faces), 3), faces)).flatten()
                mesh = pv.PolyData(verts, pv_faces)
                actor = plotter.add_mesh(mesh, color=color, opacity=opacity, label=name, smooth_shading=True)
                return actor
            except Exception as e:
                print(f"  ⚠️  Could not create mesh for {name}: {e}")
                return None
        
        # Create meshes for prediction and ground truth
        actors = {}
        
        # Ground truth
        gt_actor = create_mesh(self.viz_label, 1, "blue", 0.4, "Ground Truth")
        if gt_actor:
            actors["Ground Truth"] = gt_actor
        
        # Prediction
        pred_actor = create_mesh(self.viz_pred, 1, "red", 0.6, "Prediction")
        if pred_actor:
            actors["Prediction"] = pred_actor
        
        # Add toggle buttons
        y_pos = 10
        for name, actor in actors.items():
            def make_toggle(n):
                return lambda flag: actors[n].SetVisibility(flag)
            
            color = "blue" if "Truth" in name else "red"
            plotter.add_checkbox_button_widget(make_toggle(name), value=True, color_on=color, position=(10, y_pos))
            plotter.add_text(f"Toggle {name}", position=(60, y_pos), font_size=10)
            y_pos += 40
        
        plotter.add_legend()
        plotter.show()


# =============================================================================
# Main Entry Point
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🏥 SegNet3D Medical Image Segmentation")
    print("="*60)
    
    # Create application
    app = MedicalSegmentationApp(
        root_dir="./data",
        model_type="light",  # Use lightweight model for CPU
        num_classes=2
    )
    
    # Prepare data (auto-detects heart data or CT scan)
    try:
        app.prepare_data()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nPlease ensure your data folder contains:")
        print("  - data/ct.nii.gz (CT volume)")
        print("  - data/segmentations/<organ>.nii.gz (segmentation mask)")
        print("  OR")
        print("  - data/heart/images/heart.nii")
        print("  - data/heart/masks/heartseg.nii")
        sys.exit(1)
    
    # Train and evaluate
    model = app.train_and_evaluate(epochs=5)
    
    # Launch 3D visualization
    app.launch_3d_viewer()
