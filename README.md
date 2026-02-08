# Rapid Miner processes

Perform ML process by RapidMiner

## Titanic decision tree

Using decision tree for classification in Titanic dataset

## **RapidMiner** to explore unsupervised methods (outlier detection, clustering) and supervised methods (classification, stacking) on two datasets: **EmployeesSalary** and **Diabetes**

---

### Techniques Used

#### Task 1 – EmployeesSalary

- **Data construction & integration**
  - Create a synthetic record with personal student information using `Create ExampleSet` and append it to the EmployeesSalary data.
  - Duplicate the `Salary` attribute so original salary values can be preserved for later joins.

- **Data preparation**
  - Column selection to keep only relevant attributes.
  - Encoding of nominal attributes using `Nominal to Numerical` to make the data suitable for distance‑based methods.
- **Outlier detection**
  - **1‑Class SVM** to compute an outlier score for each employee, indicating how unusual the record is compared to the rest.
  - Convert the continuous outlier score into a **binary flag** (outlier vs. normal).
  - Join the outlier results back to the original dataset and keep original salary plus the new flag/score attributes.

- **Density‑based clustering**
  - **DBSCAN** clustering on the prepared EmployeesSalary data to identify dense regions and mark **noise points**.
  - Join cluster IDs and noise labels with the original data and select relevant original columns plus clustering outputs.

#### Task 2 – Diabetes Dataset

- **Data loading & preparation**
  - Import the Diabetes `.arff` file from Weka into RapidMiner’s Local Repository.
  - Set the class attribute (diabetes outcome) and prepare features, again ensuring suitability for distance‑based algorithms (e.g., scaling/encoding).

- **Handling class imbalance**
  - **Resampling** to obtain **equal numbers of instances for both classes**, creating a balanced dataset.
  - 
- **Train–test splitting and reuse**
  - Apply a **70/30 split** to the (multiplied) dataset to create training and testing sets.
  - Multiply both train and test sets so that exactly the same samples are reused for all models and for stacking.

- **Base classifiers**
  - Train and evaluate the following on the same train/test split:
    - **k‑Nearest Neighbours (kNN)**
    - **Naïve Bayes (NB)**
    - **Support Vector Machine (SVM)**
    - **Logistic Regression**

- **Stacked generalization**
  - Build a **stacked ensemble** with:
    - Base learners: kNN, Naïve Bayes, SVM.
    - Meta‑learner: Logistic Regression trained on the base learners’ predictions.
  - Evaluate the stacked model on the same test set used for the individual models.

- **Result reporting**
  - Manually tabulate performance metrics (e.g., accuracy, precision, recall, etc.) for all individual models and the stacked model in a comparison table.

---

### Key RapidMiner Operators

- `Create ExampleSet` – build a one‑row example containing student information for EmployeesSalary.
- `Nominal to Numerical` – one‑hot encode nominal attributes for distance‑based algorithms.
- `One-Class SVM` – compute outlier scores for EmployeesSalary records.
- Thresholding / generate attribute – convert SVM outlier scores to a binary outlier flag.
- `Join` – merge outlier or clustering outputs back to the original data while preserving original salary.
- `DBSCAN` – perform density‑based clustering and identify noise points.
- `Sample` / resampling operators – balance the Diabetes dataset by taking equal numbers of both classes.
- `Split Data` – apply a 70/30 train–test split.
- Classification operators: `k-NN`, `Naive Bayes`, `SVM`, `Logistic Regression`.
- Stacking / ensemble operators – configure kNN, NB, SVM as base learners and Logistic Regression as stacker.

---

### Expected Challenges and Notes

- **Data preparation for distance‑based methods**  
  Ensuring all relevant features are numerical and appropriately scaled/encoded is critical; incorrect encoding or missing scaling can degrade SVM, DBSCAN, and kNN performance.

- **Interpreting outlier scores and flags**  
  Choosing a threshold for converting 1‑class SVM scores into a meaningful outlier flag may require experimentation and domain intuition.

- **DBSCAN parameter tuning**  
  Selecting `eps` and `minPts` values that produce sensible clusters and a reasonable amount of noise can be non‑trivial, especially in high‑dimensional space.

- **Class imbalance handling**  
  Designing a resampling strategy that truly balances the Diabetes dataset without overfitting (e.g., random over‑sampling vs. under‑sampling) can be challenging.

- **Consistent train/test splits for fair comparison**  
  Multiplying the datasets and carefully wiring the process so that all models use the exact same train and test sets is required to make a fair performance comparison and to feed stacking correctly.

- **Configuring and interpreting stacking**  
  Properly wiring base learners and the Logistic Regression meta‑learner, and interpreting performance gains (or lack thereof) relative to single models, requires careful process design and analysis.
