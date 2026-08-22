"""
One-off script to turn the raw FairFace val.pt pickle into a plain folder of
jpgs plus a labels csv, the same shape the notebook expects to find on disk
(fairface_labels.csv with image_path, age, gender, race columns).

We only keep a random sample of the validation set (not the full ~11k
images) so that the notebook stays runnable end to end in a reasonable
amount of time, including the pretrained FER inference pass.
"""

import pickle
import random
from pathlib import Path

SAMPLE_SIZE = 1500
SEED = 42

here = Path(__file__).parent
with open(here / 'val.pt', 'rb') as f:
    examples = pickle.load(f)

print(f'loaded {len(examples)} examples from val.pt')

random.seed(SEED)
sample = random.sample(examples, SAMPLE_SIZE)

img_dir = here / 'images'
img_dir.mkdir(exist_ok=True)

rows = ['image_path,age,gender,race']
for ex in sample:
    img_id = ex['_id']
    fname = f'{img_id}.jpg'
    with open(img_dir / fname, 'wb') as out:
        out.write(ex['img_bytes'])
    rows.append(f"images/{fname},{ex['age']},{ex['gender']},{ex['race']}")

with open(here / 'fairface_labels.csv', 'w') as f:
    f.write('\n'.join(rows) + '\n')

print(f'wrote {len(sample)} images and fairface_labels.csv')
