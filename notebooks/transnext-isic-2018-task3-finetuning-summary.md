# TransNeXt Fine-tuning Summary (ISIC 2018 Task 3)

This notebook trains and evaluates a TransNeXt-Base classifier for ISIC 2018 Task 3 (7 skin lesion classes).

## What was done
- Set up a Kaggle training environment (Python 3.8 + PyTorch 2.0.1 + mmcv/timm) and fixed CUDA runtime issues needed by the `swattention` extension.
- Prepared ISIC 2018 Task 3 data from official CSV labels into `ImageFolder` format (`train/val/test` with class subfolders).
- Loaded pretrained `transnext_base_384_1k_ft_1k.pth`, replaced the classification head for 7 classes, and fine-tuned end-to-end.
- Trained for 30 epochs with a low learning rate (`1e-5`), warmup (`5` epochs), gradient clipping, and medical-safe augmentation choices (`mixup=0`, `cutmix=0`).
- Evaluated the trained checkpoint on the official test split and compared inference-time resolution settings.

Best result (test set, 1,512 images): **Acc@1 = 88.29%**, **Acc@5 = 99.27%**, **Loss = 0.476**, using `input-size=512` with `pretrain-size=384`.

## Leaderboard context (ISIC 2018 Task 3)
- This run is **no external data** (challenge data only).
- On the ISIC 2018 leaderboard screenshot, the primary metric is **Balanced Accuracy** with top scores around **0.885** (overall) and **0.845** (best shown among entries without external data).
- Our reported **Acc@1 = 88.29%** is standard top-1 accuracy, so it is useful as a practical reference but not a strict apples-to-apples comparison to leaderboard Balanced Accuracy.

Training notebook: https://www.kaggle.com/code/kydinhnhat/transnext-isic-2018-task3-finetuning  
Model weights: https://www.kaggle.com/models/kydinhnhat/transnext-base-384-isic-2018-finetuning
