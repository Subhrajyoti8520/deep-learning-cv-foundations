# Applied Deep Learning & Computer Vision Foundations

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAI](https://img.shields.io/badge/FastAI-2.7%2B-00A98F.svg)](https://docs.fast.ai/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A curated collection of modular, production-focused deep learning implementations and computer vision labs. This repository covers end-to-end workflows: automated data scraping, dataset verification, transfer learning backbones, pixel-level semantic segmentation, categorical tabular embeddings, collaborative filtering, error diagnosis, and standalone model serialization.

---

## Modality & Technical Coverage

| Lab / Domain | Core Architecture | Dataset / Source | Key Techniques |
| :--- | :--- | :--- | :--- |
| **01. Binary Vision** | ResNet-18 | DuckDuckGo Web Scraping | Lighting invariance, DataBlock API, Single-image inference |
| **02. Semantic Segmentation** | U-Net + ResNet-34 | CamVid Tiny | Pixel-level classification, custom argmax metric, foreground accuracy |
| **03. Tabular Classification** | MLP + Categorical Embeddings | Adult Census Income | Embedding high-cardinality data, 1-Cycle policy, F1-Score |
| **04. Recommendation Engine** | Matrix Factorization | MovieLens 100k Sample | Latent factor embeddings, Sigmoid output bounding (`y_range`) |
| **05. Systems & Foundations** | Theoretical & Design Guide | Literature & Industry Notes | Drivetrain approach, NLP factuality, CV domain drift |
| **06. Vision Pipeline & Cleaning** | ResNet-18 + Active Cleanup | Web-Scraped Multi-Class | Transform benchmarks, GUI data sanitization, standalone `.pkl` export |

---

## Repository Structure

      deep-learning-cv-foundations/
      ├── assets/                                 # Visuals, plots, and architecture diagrams
      ├── notebooks/
      │   ├── 01_binary_image_classifier_resnet18.ipynb
      │   ├── 02_semantic_segmentation_camvid_unet.ipynb
      │   ├── 03_tabular_classification_adult_census.ipynb
      │   ├── 04_collaborative_filtering_movie_recommender.ipynb
      │   ├── 05_deep_learning_foundations_and_system_design.ipynb
      │   └── 06_multiclass_bear_classifier_and_data_cleaning.ipynb
      ├── requirements.txt                        # Core dependencies
      └── README.md


---

## Notebook Index & Detailed Summaries

### [01. Binary Image Classification (ResNet-18)](notebooks/01_binary_image_classifier_resnet18.ipynb)
* **Task:** Binary Image Classification (`Bird` vs. `Forest`).
* **Backbone:** ResNet-18 (Pretrained on ImageNet).
* **Highlights:**
  * Automated image retrieval pipeline querying multi-condition search terms (`sun`, `shade`) to enforce lighting invariance.
  * FastAI `DataBlock` construction with directory-derived labeling (`parent_label`) and corrupted image pruning (`verify_images`).
  * Transfer learning fine-tuning loop and standalone single-instance inference with softmax confidence scores.

### [02. Pixel-Level Semantic Segmentation (U-Net)](notebooks/02_semantic_segmentation_camvid_unet.ipynb)
* **Task:** Pixel-Level Semantic Segmentation (CamVid Tiny).
* **Backbone:** U-Net with ResNet-34 Encoder Backbone.
* **Highlights:**
  * Lambda-based segmentation mask file association mapping input scenes to pixel-level ground truth classes.
  * Custom evaluation metric implementation (`acc_seg`) computing argmax pixel tensor matches.
  * Multi-metric performance tracking across validation loss, global pixel accuracy, and foreground-specific accuracy.

### [03. Tabular Deep Learning & Categorical Embeddings](notebooks/03_tabular_classification_adult_census.ipynb)
* **Task:** Binary Income Classification ($>\$50\text{K}$ vs. $\le\$50\text{K}$).
* **Architecture:** Tabular Neural Network with Categorical Embeddings.
* **Highlights:**
  * Automated continuous feature scaling (`Normalize`), missing value imputation (`FillMissing`), and high-cardinality discrete encoding (`Categorify`).
  * Learning rate scheduling optimized via the 1-Cycle policy (`fit_one_cycle`).
  * Balanced performance evaluation using both `accuracy` and `F1Score` (precision-recall harmonic mean).

### [04. Collaborative Filtering Recommendation Engine](notebooks/04_collaborative_filtering_movie_recommender.ipynb)
* **Task:** Continuous User-Movie Rating Prediction (MovieLens Sample).
* **Architecture:** Embedding-based Latent Factor Matrix Factorization.
* **Highlights:**
  * Ingestion and mapping of sparse user-item interaction pairs into dense continuous latent spaces.
  * Bounded output layer scaling (`y_range=(0.5, 5.5)`) using scaled sigmoids to constrain predictions to realistic rating ranges without gradient clipping.
  * Fine-tuning with discriminative learning rates to capture latent user taste vectors.

### [05. Deep Learning Foundations & System Design](notebooks/05_deep_learning_foundations_and_system_design.ipynb)
* **Focus:** Deep Learning Theory, Modality Comparison & Production System Design.
* **Highlights:**
  * **Computer Vision Taxonomy:** Recognition vs. Object Detection vs. Semantic Segmentation and handling out-of-domain distribution shift.
  * **NLP System Dynamics:** Hallucination mitigation, generator vs. detector arms races, and speed vs. accuracy trade-offs.
  * **The Drivetrain Approach:** A 4-step framework for causal data products (Objective $\rightarrow$ Levers $\rightarrow$ Data $\rightarrow$ Models) and counterfactual two-model recommendation evaluation.
  * **Framework Architecture:** Modularity breakdown of `fastai` (Learner abstractions), `fastcore` (idiomatic Python power tools), and `fastbook`.

### [06. Multi-Class Vision Pipeline & Data Cleaning](notebooks/06_multiclass_bear_classifier_and_data_cleaning.ipynb)
* **Task:** 4-Class Classification (`grizzly`, `black`, `teddy`, `polar`).
* **Backbone:** ResNet-18 with Image Cleaning and Model Export.
* **Highlights:**
  * Fault-tolerant image scraper with automated exponential backoff retries and multi-threaded parallel downloads (`n_workers=8`).
  * Systematic transform benchmarking comparing `Squish`, `Pad`, `RandomResizedCrop`, and GPU batch augmentations (`aug_transforms`).
  * Diagnostics and model interpretation via confusion matrices and top loss extraction.
  * In-notebook data cleaning with `ImageClassifierCleaner` to delete noisy samples and re-route mislabeled images on disk.
  * Production serialization (`learn.export`) and standalone artifact loading (`load_learner`).

---

## Technical Stack & Dependencies

* **Core Frameworks:** PyTorch, FastAI, Torchvision
* **Data Processing & Utilities:** NumPy, Pandas, FastCore, Pillow
* **Data Gathering & UI:** DuckDuckGo Search API (`ddgs`), `ipywidgets`

---

## Quickstart & Environment Setup

```bash
# Clone the repository
git clone https://github.com/Subhrajyoti8520/deep-learning-cv-foundations.git
cd deep-learning-cv-foundations

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install project dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Install nbdev for notebook development
pip install nbdev

```
Launch Jupyter:

```bash
jupyter lab
# or
jupyter notebook
```
---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
