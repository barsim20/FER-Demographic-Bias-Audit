"""
One-off script to turn the raw FairFace val.pt pickle into a plain folder of
jpgs plus a labels csv, the same shape the notebook expects to find on disk
(fairface_labels.csv with image_path, age, gender, race columns).

Uses the full FairFace validation split (close to 11k images).
"""

import pickle
from pathlib import Path

here = Path(__file__).parent
with open(here / 'val.pt', 'rb') as f:
    examples = pickle.load(f)

print(f'loaded {len(examples)} examples from val.pt')

img_dir = here / 'images'
img_dir.mkdir(exist_ok=True)

rows = ['image_path,age,gender,race']
for ex in examples:
    img_id = ex['_id']
    fname = f'{img_id}.jpg'
    with open(img_dir / fname, 'wb') as out:
        out.write(ex['img_bytes'])
    rows.append(f"images/{fname},{ex['age']},{ex['gender']},{ex['race']}")

with open(here / 'fairface_labels.csv', 'w') as f:
    f.write('\n'.join(rows) + '\n')

print(f'wrote {len(examples)} images and fairface_labels.csv')
