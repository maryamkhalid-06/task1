
import nibabel as nib
import numpy as np
import os
import shutil

def process_s0011_data():
    base_dir = r'd:\task1_the seconed time\data\s0011'
    output_dir = r'd:\task1_the seconed time\data\training_s0011'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Processing S0011 data from {base_dir} to {output_dir}...")
    
    # 1. Copy CT Volume
    ct_src = os.path.join(base_dir, 'ct.nii.gz')
    ct_dst = os.path.join(output_dir, 'ct.nii.gz')
    if os.path.exists(ct_src):
        shutil.copy2(ct_src, ct_dst)
        print(f"  Copied CT: {ct_dst}")
    else:
        print(f"  ERROR: CT not found at {ct_src}")
        return

    # 2. Process Liver
    liver_src = os.path.join(base_dir, 'segmentations', 'liver.nii.gz')
    liver_dst = os.path.join(output_dir, 'liver.nii.gz')
    if os.path.exists(liver_src):
        shutil.copy2(liver_src, liver_dst)
        print(f"  Copied Liver: {liver_dst}")
    else:
        print(f"  WARNING: Liver mask not found.")

    # 3. Process Lungs
    lung_lobes = [
        'lung_lower_lobe_left.nii.gz', 'lung_lower_lobe_right.nii.gz',
        'lung_middle_lobe_right.nii.gz', 'lung_upper_lobe_left.nii.gz',
        'lung_upper_lobe_right.nii.gz'
    ]
    for lobe in lung_lobes:
        src = os.path.join(base_dir, 'segmentations', lobe)
        dst = os.path.join(output_dir, lobe)
        if os.path.exists(src): shutil.copy2(src, dst)

    # 4. Process Other Organs (Heart, Kidneys, Pancreas, Stomach)
    other_organs = [
        'heart.nii.gz', 'kidney_left.nii.gz', 'kidney_right.nii.gz',
        'pancreas.nii.gz', 'stomach.nii.gz', 'spleen.nii.gz', 'aorta.nii.gz',
        'kidney_cyst_left.nii.gz', 'kidney_cyst_right.nii.gz'
    ]
    for org in other_organs:
        src = os.path.join(base_dir, 'segmentations', org)
        dst = os.path.join(output_dir, org)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Copied {org}")
        else:
            print(f"  WARNING: Missing {org}")

    print("S0011 Data Preparation Complete.")

if __name__ == "__main__":
    process_s0011_data()
