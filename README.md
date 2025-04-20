# titanic survival
Objective
Develop a machine learning model to predict whether a passenger survived the Titanic disaster using classification techniques. The dataset includes features such as age, gender, ticket class, fare, etc. The goal is to build a robust classifier with well-structured preprocessing and model selection pipelines.
Dataset & Features
The dataset consists of features like:
Pclass: Ticket class


Sex: Gender of the passenger


Age: Age in years


SibSp: Number of siblings/spouses aboard


Parch: Number of parents/children aboard


Fare: Passenger fare


Embarked: Port of Embarkation


Survived: Target variable (0 = No, 1 = Yes)



Steps to Run the Project
Clone the Repository:


git clone https://github.com/your-username/titanic-survival-prediction.git
cd titanic-survival-prediction

Install Dependencies:


pip install -r requirements.txt

Run the Notebook: Open the Jupyter Notebook file:


jupyter notebook titanic__survival.ipynb


Workflow Summary
 1. Preprocessing Steps
This phase is about preparing raw data to be suitable for machine learning models.
What You Did:
Missing Value Handling:


Dropped Cabin column (because it had ~78% missing data).


Filled missing values in Age, Fare using appropriate strategies (like mean, mode, etc.).
Exploratory Data Analysis (EDA)
In this phase, we explored and visualized the Titanic dataset to understand patterns, detect anomalies, and inform our feature engineering.
 Key EDA Steps:
Missing Value Analysis:


Identified that the Cabin column had ~78% missing values — dropped it due to insufficient data.


Filled missing values in Age using median, and Embarked using the mode.


Univariate Analysis:


Visualized distributions of individual features like Age, Fare, and Survived.


Observed that most passengers were in the 20–40 age range.


Fare was right-skewed with a few very high values.


Bivariate Analysis:


Checked survival rates across Pclass, Sex, Embarked, and FamilySize.


Found strong survival correlation with gender (females had higher survival rate) and ticket class (1st class had highest survival).


Visualized relationships using bar plots, histograms, and boxplots.


Outlier Detection:


Identified a few extreme outliers in Fare (very high-paying passengers).


Considered capping or binning such features later during preprocessing.


Correlation Matrix:


Used a heatmap to study correlations between numerical features.


Ensured no multicollinearity between critical variables like Fare, Pclass, and Age.





Feature Engineering:


Extracted Title from the Name (e.g., Mr., Miss., etc.) to capture social status or gender cues.


Created FamilySize = SibSp + Parch + 1 to see if traveling with family affects survival.


Optional binning of Fare and Age for better classification if needed.


Encoding Categorical Variables:


Used LabelEncoder or pd.get_dummies() to convert categorical variables (Sex, Embarked, Title) into numeric form.


Normalization:


Applied StandardScaler on Fare and Age so that they have zero mean and unit variance — important for models like SVM or Logistic Regression.


 2. Model Selection
This phase focuses on choosing and testing different machine learning algorithms to see which performs best.
 What You Tried:
Logistic Regression:
 A linear model used as a strong baseline for binary classification.


Random Forest:
 An ensemble of decision trees. Worked best for this dataset due to its ability to handle non-linear relationships and feature interactions.


XGBoost:
 Advanced boosting model with regularization, performs well but requires careful tuning.


Support Vector Machine (SVM):
 Great for smaller datasets and can classify with margins, but may struggle with non-scaled or noisy data.


 How You Evaluated Models:
Trained all on the same preprocessed data.


Compared their performance using:


Accuracy


Precision, Recall, F1 Score


Confusion Matrix


ROC Curve & AUC (for probabilistic understanding)



 3. Performance Analysis
This part interprets how well each model actually performed.
 What You Used:
Train-Test Split (80:20):
 To evaluate generalization of the model.


Metrics:


Accuracy – % of correct predictions.


Precision – % of positive predictions that were actually positive.


Recall – % of actual positives that were predicted correctly.


F1 Score – Harmonic mean of precision and recall.


Confusion Matrix – Shows true/false positives/negatives.


ROC-AUC Curve – Measures the model's ability to distinguish between classes (higher is better).


 Final Model:
Random Forest was chosen as the final model based on the best overall balance between metrics (especially F1 Score and AUC).


You also applied GridSearchCV to fine-tune Random Forest’s hyperparameters.


The final model was saved using joblib so it can be reused without retraining.




Final Outcome
Achieved good predictive performance using Random Forest.


Model evaluated using multiple metrics including ROC-AUC.


Entire process documented and reproducible via Jupyter Notebook.






