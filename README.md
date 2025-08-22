# Spatio-Temporal Crop Modeling Project

This README provides a **living roadmap** for building a deep learning model using satellite imagery for crop classification and monitoring.  
Progress through **phases**, mark off **checkpoints**, and choose from **alternative forks** when we hit decision points.

---

## Phase 1: Foundation & Data Acquisition

**Goal: Prepare  environment, data pipeline, and ground truth.**

### Checkpoints
- [ ] Provision cloud compute (GPU-enabled instance) and set up storage bucket.
- [ ] Initialize Python virtual environment.
- [ ] Install core dependencies:
  ```bash
  pip install torch torchvision timm einops torchgeo rasterio geopandas scikit-learn
  ```
- [ ] Initialize Git repository.
- [ ] Obtain field boundary polygons (GeoJSON or Shapefile) for your AOI.
- [ ] Acquire crop type labels for a subset of fields.

### Data Sourcing Strategy — Forks
- **Fork 1 (Recommended):** Use a hosted data platform.
  - Microsoft Planetary Computer → direct access to Sentinel/Landsat archives.
  - Google Earth Engine → scalable queries and data retrieval.
- **Fork 2 (Manual):** Build our own download pipeline.
  - Use tools like `sentinelsat` or Landsat APIs to fetch raw granules.
  - Build robust retry logic and processing error handling.
- **Fork 2 (Manual):** Satellite Fetcher
### Preprocessing Pipeline
- For each field polygon and relevant time window:
  - Query intersecting Sentinel-1, Sentinel-2, and Landsat scenes.
  - Apply cloud masking (on optical imagery) and ensure Level-2A surface reflectance.
  - Clip to polygon bounding box.
  - Resample and co-register all imagery to a common 10 m grid.
  - Save “chips” (e.g., 64×64 px) named as `{field_id}_{satellite}_{date}.tif`.

---

## Phase 2: Model Implementation & Pre-Training

**Goal: Develop and train a self-supervised spatio-temporal model on unlabeled imagery.**

### Checkpoints
- [ ] Create cloud-resident dataset of preprocessed image chips.
- [ ] Implement custom PyTorch `Dataset` to load time-series chips by `field_id`.
- [ ] Define model architecture (backbone + embeddings + transformer).
- [ ] Develop self-supervised objective and training loop.

### DataLoader Details
- Input: `field_id`
- Handles:..
  - Irregular acquisition dates.
  - Multi-modality imagery (SAR, optical, etc.)
- Output sequences of `(image, date, modality)` tuples.

### Architecture Components
- **Spatial Encoder:** Choose between:
  - ResNet-50 (via timm)
  - Vision Transformer (ViT-Base)
- **Temporal Embedding:** Sinusoidal encoding for dates.
- **Modality Embedding:** Simple `nn.Embedding` capturing sensor type.
- **Core Transformer Encoder:** Processes combined token sequences.

### Self-Supervised Learning — Forks
- **Fork 1 (Robust Path – ST-MAE):**
  - Mask ~75% of spatio-temporal tokens.
  - Train model to reconstruct masked patches.
  - Loss: MSE on reconstruction.
- **Fork 2 (Experimental Path – Contrastive Learning):**
  - Generate augmented views of time-series (e.g., drop frames, jitter).
  - Objective: NT-Xent contrastive loss between positive pairs.

### Training Setup
- Optimizer: `AdamW`
- Learning rate scheduler: Cosine annealing.
- Enable automatic checkpointing.
- Monitor reconstruction (or contrastive) loss to confirm training progress.

---

## Phase 3: Fine-Tuning & Evaluation

**Goal: Leverage pretrained embeddings for downstream tasks like classification or health monitoring.**

### Checkpoints
- [ ] Load pretrained encoder weights.
- [ ] Prepare labeled dataset (fields with ground-truth crop types).
- [ ] Attach task-specific heads (classification/regression).
- [ ] Train head(s) and evaluate performance.

### Task Heads
- **Crop Classification:** Add a linear layer + softmax.
- **Health Monitoring (e.g., NDVI):** Add a regression head (linear).

### Embedding Strategy — Forks
- **Fork 1 (End-of-Season Classification):**
  - Process full time-series → extract one embedding (e.g., [CLS] or average) → classify.
- **Fork 2 (Time-Series Monitoring):**
  - Produce per-timestep embeddings → predict health value for each time-step.

### Fine-Tuning
- Freeze encoder weights.
- Train only the lightweight head(s)—quick (minutes to hours).

### Model Evaluation
- Classification metrics: Accuracy and F1-score.
- Regression metrics: R² and RMSE.
- Qualitative error analysis: Which crops are misclassified? Temporal trends mistakes?

---

## Optional Extensions & Next Steps
- Explore **semi-supervised learning** for improving sample efficiency.
- Try **multi-task learning** (e.g., classification + NDVI regression).
- Scale up to larger AOIs or new satellite platforms.
- Integrate **active learning** to iteratively label hard examples.

---

### Thoughts on AlphaEarth Foundations & Project Inspiration

**AlphaEarth Foundations**, from Google DeepMind, embodies a compelling vision: a *“virtual satellite”* that compresses vast, multimodal Earth observation data into compact, annual 10 m × 10 m embeddings that users can quickly query and analyze.

Key highlights include:

- **Multimodal fusion**: Integrates optical imagery, radar, LiDAR, climate simulations, and more into a seamless representation.
- **Storage efficiency**: Embeddings are ~16× smaller than traditional formats, enabling scalable use.
- **Superior accuracy**: Delivers ~24% lower error compared to competing mapping systems, even when label data is sparse.
- **Continuous time modeling**: The architecture supports interpolation/extrapolation of temporal gaps—addressing cloud cover and revisit limitations.
- **Accessibility**: Made available via Google Earth Engine as "Satellite Embedding" layers (2017–2024) for a wide range of users.

**How this aligns with our approach:**

- **Self-supervised foundation**: The idea of training a spatio-temporal model to produce embeddings directly parallels AlphaEarth’s embedding field concept.
- **Efficiency & scalability**: Although we may not match AlphaEarth’s global scale, forecasting and compression concepts (e.g., masked reconstruction) are similar in spirit.
- **Time-aware modeling**: Their "Space-Time Precision" architecture inspires our temporal encoding and transformer-based modeling strategies.
- **Downstream utility**: Like how AlphaEarth supports diverse applications (crop mapping, ecosystems, etc.), our downstream heads approach the same principle of transferring general embeddings to specific tasks.

**Possible enhancements to consider:**

- Adopt multi-sensor fusion (e.g., combining optical + SAR) directly in our model, rather than just preprocessing separately.
- Introduce embedding compression or dimensionality reduction to reduce storage and speed up training/inference.
- Explore teacher-student or inpainting strategies to enhance robustness to missing data, especially during clouds or sensor gaps.
- Consider generating per-year or per-season embeddings to align with how AlphaEarth offers annual snapshots.

---


