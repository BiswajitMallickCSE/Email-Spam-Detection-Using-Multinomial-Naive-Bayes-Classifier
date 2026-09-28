# Email Spam Detection and Analytics Using Multinomial Naive Bayes

An end-to-end intelligent text classification and feature-engineering pipeline developed to detect and filter out spam and junk emails from textual communication networks. This system leverages probabilistic machine learning and high-quality token vectorization to achieve robust classification accuracy.

## 🚀 Key Technical Highlights
* **High-Quality Feature Extraction:** Implemented text tokenization and numerical vectorization using `CountVectorizer` to transform raw text messages into rich numerical feature matrices.
* **Probabilistic Classification:** Utilized a Custom **Multinomial Naive Bayes (MultinomialNB)** classifier, which is structurally optimized for discrete word-count distributions and text analytics.
* **Performance Optimization:** Evaluated features using log-loss metrics to minimize convergence delays and reduce misclassification risks.
* **Advanced Data Visualization:** Integrated k-fold insights into automated visualization, plotting distribution pie-charts, confusion matrix heatmaps, and classification metric reports.

---

## 📊 Model Performance & Benchmarks
The model was rigorously trained and validated on a text corpus utilizing an 80/20 train-test configuration. The probabilistic framework demonstrated outstanding robustness across all evaluation metrics:

* **Training Accuracy:** `99.44%`
* **Testing Accuracy:** `97.85%`
* **Training Log-Loss:** `0.202`
* **Testing Log-Loss:** `0.776`

### 🗺️ Confusion Matrix Diagnostics
* **True Negatives (Correctly identified Legal Mail):** `952`
* **True Positives (Correctly identified Junk Mail):** `139`
* **False Positives (Type I Error):** `13`
* **False Negatives (Type II Error):** `11`

The classification pipeline generated an overall **Spam F1-Score of 0.92**, proving its readiness for secure enterprise message filtering.

---

## 🛠️ Tech Stack & Frameworks
* **Language:** Python 3.x
* **Core Libraries:** NumPy, Pandas, Scikit-Learn
* **Visualization:** Matplotlib, Seaborn
* **Environment:** Google Colab / Jupyter Notebook

---

## 📂 Project Workflow & Architecture
1. **Data Preprocessing:** Handled encoding issues using `ISO-8859-1`, dropped redundant dimensions, and mapped text categories into binary tensors (`0` for Ham, `1` for Spam).
2. **Distribution Analysis:** Analyzed dataset balance through descriptive analytics and visual charts.
3. **Vectorization:** Transformed clean textual sequences into sparse numerical token vectors.
4. **Model Fit & Evaluation:** Trained the Multinomial Naive Bayes pipeline and extracted precision, recall, and classification report matrices.
