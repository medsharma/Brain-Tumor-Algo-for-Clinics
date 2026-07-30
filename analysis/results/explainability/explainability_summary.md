# Explainability — Grad-CAM (ResNet-50) & Attention Rollout (ViT-B/16)

Run: `20260703_155524`

## resnet50 (Grad-CAM)

- **glioma**: pool=80, misclassified_in_pool=6, 4 example(s) saved
- **meningioma**: pool=80, misclassified_in_pool=6, 5 example(s) saved
- **pituitary**: pool=80, misclassified_in_pool=2, 4 example(s) saved
- **notumor**: pool=80, misclassified_in_pool=3, 4 example(s) saved

## vit (Attention Rollout)

- **glioma**: pool=80, misclassified_in_pool=5, 6 example(s) saved
- **meningioma**: pool=80, misclassified_in_pool=2, 5 example(s) saved
- **pituitary**: pool=80, misclassified_in_pool=2, 5 example(s) saved
- **notumor**: pool=80, misclassified_in_pool=1, 5 example(s) saved

Heatmap PNGs saved under `analysis/results/explainability/{resnet50,vit}/`.