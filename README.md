# FER Demographic Bias Audit

Audits a pretrained facial expression recognition (FER) model for demographic bias, using FairFace for age, gender and race labels and `trpakov/vit-face-expression` for expression pseudo-labels.

## Layout

- `data/prepare_dataset.py` - pulls a sample of the FairFace validation set (`nateraw/fairface` on Hugging Face) and writes it out as plain jpgs plus `fairface_labels.csv`.
- `data/fairface_labels.csv`, `data/images/` - the sampled dataset used by the notebook.
- `data/fer_predictions.csv` - cached expression pseudo-labels from the ViT model, so the notebook doesn't have to rerun inference on every kernel restart.
- `notebooks/FER_Demographic_Bias_Audit.ipynb` - the full audit, from data loading through the subgroup bias analysis and report write-up.

## Running it

```
pip install pandas scikit-learn scipy seaborn matplotlib pillow tensorflow-cpu transformers torch
jupyter notebook notebooks/FER_Demographic_Bias_Audit.ipynb
```

`data/fairface_labels.csv` and `data/fer_predictions.csv` are already included, so the notebook runs without needing to redo the data pull or the ViT inference pass. Delete `data/fer_predictions.csv` first if you want to regenerate the pseudo-labels from scratch.
