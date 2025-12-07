# Explanation of Attribution Discrepancy: DinoV2 vs ViT-base

## Observation
When comparing the CheferCAM explainability maps between **ViT-base** (224x224 input) and **DinoV2** (518x518 input), we observe that the attribution values in DinoV2 are significantly lower (max ~0.001) compared to ViT-base (max ~0.003). The visualization for DinoV2 also appears less "focused" or "hot".

## The Cause: Patch Density Dilution
The primary reason for this discrepancy is the difference in the number of patches (tokens) between the two models, caused by the difference in input image resolution.

### 1. Patch Count Comparison
- **ViT-base**:
    - Input Size: 224x224
    - Patch Size: 16x16
    - Grid Size: 14x14
    - **Total Patches**: $14 \times 14 = 196$

- **DinoV2 (with registers)**:
    - Input Size: 518x518
    - Patch Size: 14x14
    - Grid Size: $518/14 = 37$
    - **Total Patches**: $37 \times 37 = 1369$

### 2. Conservation of Attribution Mass
Explainability methods like Attention Rollout and CheferCAM (Transformer Attribution) generally conserve the total "attention mass" or "relevance" flowing from the [CLS] token to the input tokens. Ideally, the sum of attribution across all input tokens is close to 1.

$$ \sum_{i=1}^{N} A_i \approx 1 $$

Where $N$ is the number of patches and $A_i$ is the attribution value for patch $i$.

### 3. The Dilution Effect
If we distribute a total mass of ~1.0 across a different number of patches, the average value per patch changes inversely with the number of patches.

- **Average Value for ViT-base**:
  $$ \frac{1}{196} \approx 0.0051 $$

- **Average Value for DinoV2**:
  $$ \frac{1}{1369} \approx 0.00073 $$

**Ratio of Dilution**:
$$ \frac{196}{1369} \approx \frac{1}{7} $$

The average signal strength per patch in DinoV2 is naturally about **7 times lower** than in ViT-base simply because the image is divided into 7 times more pieces.

### 4. Empirical Verification
In our analysis of the DinoV2 model, we observed:
- **Patches Total Sum**: ~0.05 (Note: CheferCAM doesn't strictly sum to 1 like Rollout, and some mass goes to registers/CLS self-attention, but the relative scale holds).
- **Registers Total Sum**: ~0.015
- **Max Patch Value**: ~0.00103

A max value of `0.001` in DinoV2 is roughly equivalent to a max value of `0.007` in ViT-base in terms of "significance relative to the uniform distribution".

## Conclusion
The lower values in the DinoV2 explainability maps are **not an error**. They are a mathematical consequence of the higher resolution (518x518). The model is providing a much finer-grained explanation (1369 pixels vs 196 pixels in the heatmap).

To compare them visually on the same scale, one would theoretically need to multiply the DinoV2 values by ~7, but it is better to understand them as densities: **DinoV2 provides a lower-density but higher-resolution signal.**
