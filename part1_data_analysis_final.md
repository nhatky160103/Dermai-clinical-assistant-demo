# Part 1: Data Exploration and Analysis
> ISIC 2018 — Mole Network Structure Analysis | Belle AI Technical Test

---

## Q1. Describe the Dataset

The **ISIC 2018 Challenge Dataset** was released by the International Skin Imaging Collaboration for the 2018 MICCAI Challenge on Skin Lesion Analysis Toward Melanoma Detection. It is sourced from two collections: the HAM10000 dataset (ViDIR Group, Medical University of Vienna) and the MSK Dataset (Memorial Sloan Kettering Cancer Center), acquired from multiple clinical centers using different dermoscope types.

The challenge attracted 900 registered teams and 299 total submissions across 3 tasks.

### Tasks

| Task | Description | Annotation |
|------|-------------|------------|
| Task 1 | Lesion Boundary Segmentation | 1 binary mask per image |
| **Task 2** | **Dermoscopic Attribute Detection** | **5 binary masks per image** |
| Task 3 | Disease Classification | 1 label per image (7 classes) |

Tasks 1 and 2 share the same input images. Task 1 answers *"Where is the lesion?"* and Task 2 answers *"What dermoscopic structures are present?"*. They are evaluated independently.

### Dataset Size (Task 1 & 2)

| Split | Images | Task 1 Masks | Task 2 Masks (×5) |
|-------|-------:|-------------:|------------------:|
| Training | 2,594 | 2,594 | 12,970 |
| Validation | 100 | 100 | 500 |
| Test | 1,000 | 1,000 | 5,000 |
| **Total** | **3,694** | **3,694** | **18,470** |

### Task 2 — 5 Dermoscopic Structures

| Structure | Clinical Description |
|-----------|---------------------|
| `pigment_network` | Pigmented reticular grid; primary indicator of melanocytic origin |
| `negative_network` | Hypopigmented inverse network; associated with specific melanoma subtypes |
| `streaks` | Radial projections at lesion periphery; may indicate aggressive growth |
| `milia_like_cyst` | Small white/yellow round globules; typically benign |
| `globules` | Small round brown structures; correspond to melanocyte nests |

Per the official challenge definition, attribute annotations are made within the full dermoscopic image — not constrained to the Task 1 lesion boundary, since some structures (e.g., streaks) by clinical definition can extend beyond the lesion edge.

---

## Q2. Analyze the Quality of Annotations

### 2.1 Class Distribution

**From EDA on training set (n = 2,594 images):**

| Structure | Prevalence | N Present | N Absent | Avg Coverage | Median Cov | Std Coverage |
|-----------|----------:|----------:|---------:|-------------:|-----------:|-------------:|
| `pigment_network` | **58.7%** | 1,523 | 1,071 | 6.76% | 2.94% | **9.62%** |
| `milia_like_cyst` | 26.3% | 682 | 1,912 | 0.81% | 0.44% | 1.08% |
| `globules` | 23.2% | 603 | 1,991 | 2.78% | 1.43% | 4.24% |
| `negative_network` | 7.3% | 190 | 2,404 | 3.30% | 2.00% | 3.99% |
| `streaks` | **3.9%** | 100 | 2,494 | 1.41% | 0.89% | 1.62% |

There is a **15× imbalance** between `pigment_network` (58.7%) and `streaks` (3.9%). This pattern is consistent across the full dataset (train + val + test): according to DatasetNinja, across all 3,694 Task 2 images, `pigment_network` appears in 2,244 images while `streaks` appears in only 183 images.

### 2.2 Coverage and Scale Characteristics

Even when present, mask coverage is very small for most structures:

- `milia_like_cyst`: average coverage only **0.81%** — punctiform micro-structures close to annotation resolution limits. Per DatasetNinja, average object area is 0.20%, with the smallest instances at 0.06%
- `streaks`: average **1.41%**, thin elongated shapes. Average object area 0.47%, with max extent reaching only ~10% of image area
- `pigment_network`: highest average coverage at **6.76%** but with very high std (9.62%), ranging from small patches to pervasive grids covering over 30% of the image. Maximum object area reaches 77.81%
- `globules`: average object area 0.59% per object, but with up to 4.57 objects per image on average — these are discrete multi-instance structures
- Median coverage is consistently lower than mean for all structures, confirming right-skewed distributions with few large outliers

### 2.3 Annotation Quality — Official Assessment

The challenge organizers explicitly acknowledge in Codella et al. (2019) that ground truth masks are influenced by inter-observer and intra-observer variability due to differences between human annotators and annotation software. An ideal evaluation would use multiple annotators per image, but this was deemed impractical.

For Task 1, prior work measured inter-annotator agreement between 3 experts on 100 images, yielding Jaccard values of 0.743, 0.754, and 0.861 (mean 0.786). This variability among experts, even for the simpler lesion boundary task, motivated the introduction of Thresholded Jaccard as the evaluation metric, with failure threshold T = 0.65.

For Task 2, Codella et al. (2019) note directly that poor model performance on this task may result from the fact that dermoscopic attributes tend to have poor inter-observer correlation among expert clinicians, citing Carrera et al. (JAMA Dermatology, 2016). The annotation noise is therefore not just a data collection artifact — it reflects genuine clinical disagreement about these structures.

### 2.4 Annotation Quality — Benchmark Results as Evidence

Only **26 teams** submitted to Task 2, compared to 112 for Task 1 and 141 for Task 3. The best Task 2 submission achieved an average Jaccard of only **0.473**, while the best Task 1 submission achieved **0.802**. This gap cannot be explained solely by model limitations — it reflects inherent difficulty caused by annotation ambiguity.

The challenge paper itself concludes that poor Task 2 performance may imply that the field of clinical dermoscopic attributes must mature further before machine learning can be effectively applied.

### 2.5 Annotation Consistency Flags from EDA

**Tiny masks (coverage < 0.1%)** labeled as "present" are likely borderline or ambiguous cases, annotation errors, or structures at the resolution limit of the annotation tool.

**High coverage standard deviation** — especially for `pigment_network` (std = 9.62%) — reflects genuine biological variability in how extensively structures manifest across patients, not pure annotation error.

**Many zero-structure images** — a notable fraction of training images have 0 Task 2 annotations, meaning the lesion exists (Task 1 mask present) but no specific dermoscopic attribute was annotated. This may reflect genuine absence, annotator fatigue, or subthreshold structures.

---

## Q3. How Annotation Quality May Affect the Model

### 3.1 Class Imbalance → Failure on Rare Structures

With a 15× prevalence gap, a model trained with standard Binary Cross-Entropy loss will learn to always predict "absent" for `streaks` and `negative_network`. This achieves high pixel accuracy while being clinically useless for these structures. Mitigation requires weighted loss functions such as Weighted BCE or Focal Loss, and balanced sampling strategies.

### 3.2 Irreducible Annotation Noise → Performance Ceiling

Because expert dermatologists themselves disagree on structure boundaries, the single-annotator ground truth contains noise that cannot be removed by better training. The best achievable Jaccard is bounded by inter-annotator agreement — which for Task 2 structures is known to be low. This means a Jaccard of ~0.473 represents state-of-the-art, not a model failure. Models should output confidence maps rather than hard binary predictions to communicate this inherent uncertainty to clinicians.

### 3.3 Multi-Scale Structures → Single-Scale Models Insufficient

`pigment_network` requires large receptive fields (grid patterns spanning up to 77% of image area per DatasetNinja). `milia_like_cyst` and `streaks` require detection at sub-1% area. No single convolutional scale handles both simultaneously. This directly justifies a multi-scale architecture such as FPN or UNet++.

### 3.4 Incomplete Annotations → Suppressed Recall

With single-annotator ground truth and known annotator fatigue in complex multi-structure images, structures present but unannotated create false negatives in training. The model is penalized for correct predictions, systematically suppressing recall across all classes.

### 3.5 Generalization

Codella et al. (2019) show that most Task 3 classification models overfit to internal data and perform worse on external data from unseen institutions. This generalization problem is likely more severe for Task 2 given its lower annotation consistency. A model deployed in a product like BelleLens must be validated on data from new clinical centers and device types beyond ISIC 2018.

### 3.6 Summary

| Issue | Evidence | Impact | Mitigation |
|-------|----------|--------|------------|
| 15× class imbalance | EDA: streaks 3.9% vs pigment_network 58.7% | Rare structures ignored | Weighted BCE / Focal Loss |
| Low inter-annotator agreement | Codella et al. (2019); Carrera et al. (2016) | Performance ceiling ~0.473 Jaccard | Output uncertainty maps |
| Multi-scale structures | EDA + DatasetNinja: 0.20% to 23.33% avg object area | Fine structures missed | FPN / UNet++ decoder |
| Single annotator per image | Codella et al. (2019) | False negatives penalize correct predictions | Pseudo-labeling, loss masking |
| Many 0-structure images | EDA | Background dominates training signal | Balanced sampling |
| Domain shift | Codella et al. (2019) Section 3.3 | Fails on new devices/centers | Augmentation, domain adaptation |

---

## References

1. Codella N. et al. "Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the ISIC." arXiv:1902.03368v2, 2019.
2. Tschandl P., Rosendahl C., Kittler H. "The HAM10000 dataset." Sci. Data 5, 180161 (2018).
3. Carrera C. et al. "Validity and Reliability of Dermoscopic Criteria." JAMA Dermatol. 152(7):798–806, 2016.
4. ISIC Challenge 2018. https://challenge.isic-archive.com/landing/2018/
5. DatasetNinja ISIC Challenge 2018. https://datasetninja.com/isic-challenge-2018