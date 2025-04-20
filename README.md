# Titanic Survival Prediction

## Task Objective

Develop a machine learning model to predict whether a passenger survived the Titanic disaster using classification techniques. The dataset includes features such as age, gender, ticket class, fare, etc. The goal is to build a robust classifier with well-structured preprocessing and model selection pipelines.

---

##  Dataset & Features

The dataset consists of features like:

- **Pclass**: Ticket class
- **Sex**: Gender of the passenger
- **Age**: Age in years
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Fare**: Passenger fare
- **Embarked**: Port of Embarkation
- **Survived**: Target variable (0 = No, 1 = Yes)

---

## Steps to Run the Project

1. **Clone the Repository:**

2. **Install Dependencies:**

```bash
pip install -r requirements.txt
```

3. **Run the Notebook:**
   Open the Jupyter Notebook file:

```bash
jupyter notebook titanic__survival.ipynb
```

---

##  Workflow Summary

### 1. Exploratory Data Analysis (EDA)

- Handled missing values
- Dropped 'Cabin' due to >70% missing data
- Visualized survival rates, distributions, etc.

### 2. Feature Engineering

- Extracted **Title** from names
- Created **FamilySize** = SibSp + Parch + 1
- Binned **Fare** and **Age** (if needed)

### 3. Data Preprocessing

- Label Encoding for categorical variables
- StandardScaler used for Fare and Age

### 4. Model Building

Implemented multiple classifiers:

- Logistic Regression
- Random Forest (final selected model)
- XGBoost
- SVM

### 5. Model Evaluation

- Split into train and validation sets (80-20)
- Metrics used:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix
  - ROC Curve

### 6. Hyperparameter Tuning

- GridSearchCV applied on Random Forest for optimal parameters

### 7. Model Saving

- Saved the final model using `joblib` as `best_model.pkl`

---

## Final Outcome

- Achieved good predictive performance using Random Forest.
- Model evaluated using multiple metrics including ROC-AUC.
- Entire process documented and reproducible via Jupyter Notebook.

---

##  Innovations & Highlights

- Title extraction from Name improved prediction.
- FamilySize helped capture group survival behavior.
- ROC-AUC curve implemented for visual performance analysis.

---

