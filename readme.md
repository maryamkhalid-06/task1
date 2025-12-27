SegNet Medical Image Segmentation Implementation
Replacing the current UNet model with a SegNet architecture for faster training, memory efficiency, and high evaluation metrics on your CT scan data.

Data Analysis
Your 
data/
 folder contains:

ct.nii.gz
 - Main CT volume (23.5 MB)
segmentations/ - 117 organ masks including heart, liver, spleen, kidneys, lungs, vertebrae, etc.
heart/images/heart.nii
 and 
heart/masks/heartseg.nii
 - Dedicated heart dataset
Why SegNet?
Feature	SegNet	UNet
Speed	✅ Faster (fewer parameters)	Standard
Memory	✅ Lower (pooling indices)	Higher (skip connections)
Accuracy	High	High
Best For	Real-time, resource-limited	Maximum accuracy
SegNet uses max pooling indices for upsampling instead of transposed convolutions, making it faster and more memory-efficient.

Proposed Changes
Component 1: SegNet Model Architecture
[NEW] 
segnet_model.py
Create a new SegNet3D model implementation:

Encoder: 5 convolutional blocks with VGG-style stacking
Pooling Indices: Store max pooling locations for decoder
Decoder: 5 blocks using unpooling with stored indices
Batch Normalization: After each conv for faster convergence
Lightweight channels: (32, 64, 128, 256, 512) for CPU efficiency
Component 2: Data Loading
[MODIFY] 
medical_segmentation_v1.py
Replace DecathlonDataset with local NIfTI data loading:

Load 
ct.nii.gz
 as the input image
Load desired segmentation masks from segmentations/ folder
Support multi-organ selection (heart, liver, spleen, etc.)
Add train/validation split functionality
Component 3: Evaluation Metrics
[MODIFY] 
medical_segmentation_v1.py
Add comprehensive evaluation metrics class:

Dice Score (DSC) - Primary metric
IoU / Jaccard - Overlap measure
Accuracy - Overall pixel accuracy
Sensitivity (Recall) - True positive rate
Specificity - True negative rate
Precision - Positive predictive value
Hausdorff Distance - Boundary accuracy
Component 4: Training Optimization
[MODIFY] 
medical_segmentation_v1.py
Add learning rate scheduler (ReduceLROnPlateau)
Add early stopping to prevent overfitting
Add model checkpointing to save best model
Add training progress logging with metrics per epoch
Optimize for CPU execution with smaller batch sizes
Verification Plan
Automated Testing
# Run from project directory with activated venv
cd "d:\task1_the seconed time"
.\segmentation_venv\Scripts\Activate.ps1
# Test 1: Verify imports and model creation
python -c "from segnet_model import SegNet3D; import torch; model = SegNet3D(1, 2); x = torch.randn(1, 1, 32, 32, 32); y = model(x); print(f'Output shape: {y.shape}')"
# Test 2: Run short training (1 epoch) to verify pipeline
python medical_segmentation_v1.py
Manual Verification
Check training output - Verify loss decreases and metrics are printed
Verify 3D visualization - Confirm PyVista viewer shows segmented organs
Compare metrics - Dice should be > 0.5 even after 1 epoch with good data
Expected Results
After implementation you will have:

✅ Faster training (30-50% speedup vs UNet)
✅ Lower memory usage (suitable for CPU)
✅ Comprehensive evaluation metrics dashboard
✅ Easy-to-use single-file solution
✅ Works with your local NIfTI data