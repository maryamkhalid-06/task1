import os
import sys
import numpy as np

# Add local dir to path to import gui modules if needed
sys.path.append(os.getcwd())

def verify_integration():
    print("--- Verifying Heart Data Integration ---")
    
    # 1. Check file existence
    img_path = "data/heart/images/heart.nii"
    mask_path = "data/heart/masks/heartseg.nii"
    
    if not os.path.exists(img_path):
        print(f"FAILED: {img_path} not found")
        return
    if not os.path.exists(mask_path):
        print(f"FAILED: {mask_path} not found")
        return
    print("SUCCESS: Heart files found in data directory.")

    # 2. Try loading with nibabel
    try:
        import nibabel as nib
        img = nib.load(img_path)
        mask = nib.load(mask_path)
        print(f"SUCCESS: Loaded NIfTI files.")
        print(f"Image shape: {img.shape}")
        print(f"Mask shape: {mask.shape}")
    except ImportError:
        print("SKIP: nibabel not yet installed in this environment.")
    except Exception as e:
        print(f"FAILED: Error loading NIfTI: {e}")

    # 3. Test DatasetHandler (from gui.py)
    try:
        from gui import DatasetHandler
        handler = DatasetHandler()
        loaded, count = handler.load_from_directory("data/heart")
        if loaded:
            print(f"SUCCESS: DatasetHandler loaded {count} volume(s).")
            img_slice, mask_slice = handler.get_current_slice()
            if img_slice is not None:
                print(f"SUCCESS: Extracted slice of shape {img_slice.shape}")
            else:
                print("FAILED: get_current_slice returned None")
        else:
            print("FAILED: DatasetHandler failed to load 'data/heart'")
    except ImportError:
        print("SKIP: gui dependencies (customtkinter, etc) not yet installed.")
    except Exception as e:
        print(f"FAILED: DatasetHandler test crashed: {e}")

if __name__ == "__main__":
    verify_integration()
