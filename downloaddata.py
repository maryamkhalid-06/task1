import os
import tarfile
import requests
import nibabel as nib
import numpy as np
from pathlib import Path

# -----------------------------
# 1. Define datasets and URLs
# -----------------------------
datasets = {
    "Heart": "https://s3.amazonaws.com/medicaldecathlon/tasks/Task02_Heart.tar",
    "Liver": "https://s3.amazonaws.com/medicaldecathlon/tasks/Task03_Liver.tar",
    "Lung":  "https://s3.amazonaws.com/medicaldecathlon/tasks/Task06_Lung.tar"
}

download_dir = Path("./MSD_Datasets")
download_dir.mkdir(exist_ok=True)

# -----------------------------
# 2. Download datasets
# -----------------------------
def download_file(url, output_path):
    if output_path.exists() and output_path.stat().st_size > 1000:
        print(f"{output_path} already exists and looks valid. Skipping download.")
        return True
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        if response.status_code != 200:
            print(f"Error downloading {url}: Status {response.status_code}")
            return False
            
        # Check if it's an XML error (common with S3 Access Denied)
        content_type = response.headers.get('Content-Type', '')
        if 'xml' in content_type.lower() or 'html' in content_type.lower():
            print(f"Error: {url} returned {content_type} instead of a tar file.")
            return False

        with open(output_path, 'wb') as file:
            for data in response.iter_content(chunk_size=1024*1024):
                file.write(data)
        
        # Immediate validation
        if not tarfile.is_tarfile(output_path):
            print(f"Error: {output_path} is not a valid tar file.")
            output_path.unlink() # Delete corrupt file
            return False

        print(f"Downloaded {output_path}")
        return True
    except Exception as e:
        print(f"Exception during download of {url}: {e}")
        return False

for name, url in datasets.items():
    tar_path = download_dir / f"{name}.tar"
    print(f"--- Processing {name} ---")
    
    success = download_file(url, tar_path)
    
    if not success:
        print(f"Failed to download {name}. Checking for fallback...")
        if name == "Heart":
            # Check for local heart.gz and heartseg.gz in root
            img_src = Path("heart.gz")
            seg_src = Path("heartseg.gz")
            if img_src.exists() and seg_src.exists():
                print("Found local fallback for Heart dataset.")
                extract_path = download_dir / name
                (extract_path / "imagesTr").mkdir(parents=True, exist_ok=True)
                (extract_path / "labelsTr").mkdir(parents=True, exist_ok=True)
                
                import shutil
                shutil.copy(img_src, extract_path / "imagesTr" / "la_003.nii.gz")
                shutil.copy(seg_src, extract_path / "labelsTr" / "la_003.nii.gz")
                print(f"Copied local heart data to {extract_path}")
                continue
        print(f"Skipping {name} due to download failure and no fallback.")
        continue

    # -----------------------------
    # 3. Extract dataset
    # -----------------------------
    extract_path = download_dir / name
    if not (extract_path / "imagesTr").exists():
        extract_path.mkdir(exist_ok=True)
        print(f"Extracting {tar_path} ...")
        try:
            with tarfile.open(tar_path) as tar:
                tar.extractall(path=extract_path)
            print(f"Extracted to {extract_path}")
        except Exception as e:
            print(f"Failed to extract {tar_path}: {e}")

# -----------------------------
# 4. Optional: Convert 3D NIfTI to 2D slices
# -----------------------------
def nii_to_slices(nii_dir, output_slices_dir):
    output_slices_dir.mkdir(parents=True, exist_ok=True)
    nii_files = list(Path(nii_dir).glob("*.nii*"))
    for nii_file in nii_files:
        img = nib.load(str(nii_file))
        data = img.get_fdata()
        # Save each slice as a .npy file (can convert to .png later)
        for i in range(data.shape[2]):  # assuming axial slices
            slice_2d = data[:, :, i]
            slice_file = output_slices_dir / f"{nii_file.stem}_slice{i:03d}.npy"
            np.save(slice_file, slice_2d)
    print(f"Converted {len(nii_files)} NIfTIs to 2D slices at {output_slices_dir}")

# Example: Convert imagesTr for each dataset
for name in datasets.keys():
    nii_image_dir = download_dir / name / f"imagesTr"
    output_slices_dir = download_dir / name / "slices"
    if nii_image_dir.exists():
        print(f"Converting {name} images to slices...")
        nii_to_slices(nii_image_dir, output_slices_dir)
