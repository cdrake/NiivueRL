"""Pick a held-out AOMIC val subject, save its brain volume to public/, and
print landmark centroids in TS-ready format for src/lib/landmarks.ts.

We replicate train_goal_net.py's split (seed=0, val_frac=0.2) to ensure the
chosen subject was NOT in training. The first val subject (after the seeded
shuffle) is used.
"""
import json
from pathlib import Path
import numpy as np
import nibabel as nib

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "aomic_id1000"

LANDMARKS = {
    "Lateral-Ventricle":       (4, 43),
    "Inf-Lat-Vent":            (5, 44),
    "Cerebellum-White-Matter": (7, 46),
    "Cerebellum-Cortex":       (8, 47),
    "Thalamus":                (10, 49),
    "Caudate":                 (11, 50),
    "Putamen":                 (12, 51),
    "Pallidum":                (13, 52),
    "3rd-Ventricle":           (14,),
    "4th-Ventricle":           (15,),
    "Brain-Stem":              (16,),
    "Hippocampus":             (17, 53),
    "Amygdala":                (18, 54),
    "Accumbens-area":          (26, 58),
    "VentralDC":               (28, 60),
}
# src/lib/landmarks.ts uses these spellings; map any differences here.
TS_NAME = {
    "Inf-Lat-Vent": "Inferior-Lateral-Ventricle",
}

# Replicate the training split.
sub_dirs = sorted(p for p in DATA.glob("sub-*") if (p / "aseg.mgz").exists())
rng = np.random.default_rng(0)
rng.shuffle(sub_dirs)
n_val = max(1, int(0.2 * len(sub_dirs)))
val_dirs = sub_dirs[:n_val]
train_dirs = sub_dirs[n_val:]
print(f"split: {len(train_dirs)} train + {len(val_dirs)} val (seed=0, val_frac=0.2)")
chosen = val_dirs[0]
print(f"chosen held-out subject: {chosen.name}")

# Load + save volume.
brain_img = nib.load(chosen / "brain.mgz")
brain = np.asarray(brain_img.dataobj)
out_nii = REPO / "public" / "aomic_test.nii.gz"
nib.save(nib.Nifti1Image(brain.astype(np.int16), brain_img.affine), out_nii)
print(f"wrote {out_nii} (shape={brain.shape}, max={brain.max()})")

# Centroids.
aseg = np.asarray(nib.load(chosen / "aseg.mgz").dataobj).astype(np.int32)
centroids: dict[str, list[int]] = {}
for name, labels in LANDMARKS.items():
    mask = np.isin(aseg, labels)
    if not mask.any():
        print(f"  WARN: {name} absent from aseg")
        continue
    c = np.argwhere(mask).mean(axis=0).round().astype(int).tolist()
    centroids[TS_NAME.get(name, name)] = c

print()
print("=== TS-ready coords (paste into src/lib/landmarks.ts mniCoords) ===")
# Match the existing key order of src/lib/landmarks.ts.
order = ["Thalamus", "Caudate", "Putamen", "Pallidum", "Hippocampus", "Amygdala",
         "Brain-Stem", "Cerebellum-White-Matter", "Cerebellum-Cortex",
         "Lateral-Ventricle", "Inferior-Lateral-Ventricle",
         "3rd-Ventricle", "4th-Ventricle", "Accumbens-area", "VentralDC"]
for k in order:
    if k in centroids:
        x, y, z = centroids[k]
        # Pad strings to roughly match the existing column alignment.
        padded = f"'{k}':"
        print(f"  {padded:<33} [{x}, {y}, {z}],")

# Also dump JSON for reference.
out_json = REPO / "public" / "aomic_test_landmarks.json"
out_json.write_text(json.dumps({"subject": chosen.name, "centroids": centroids}, indent=2))
print(f"\nwrote {out_json}")
