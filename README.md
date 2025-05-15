# Parkinsons-disease-prediction

Parkinson's Disease Prediction using Machine Learning
🧠 Overview
Parkinson's Disease (PD) is a progressive neurological disorder that impacts movement. Symptoms often include stiffness, tremors, and bradykinesia (slowed movements). While there is no single definitive diagnostic test for Parkinson's, machine learning can help predict the likelihood of the disease based on biomarkers. This project explores how machine learning models can be used to classify individuals as healthy or affected by Parkinson’s using diagnostic features.

📁 Dataset
The dataset contains 755 features with 3 observations per patient.

After aggregation and cleaning, the data was reduced to 252 samples.

Target variable: class (0 = Unhealthy, 1 = Healthy)

The dataset is assumed to have no missing values.

🛠️ Libraries Used
Pandas, NumPy – Data manipulation and analysis

Matplotlib, Seaborn – Data visualization

Scikit-learn – Machine learning tools and models

XGBoost – Gradient boosting algorithm

Imblearn – Handling class imbalance

TQDM – Progress bars for iterations

📊 Data Preprocessing
1. Aggregation
Multiple rows per patient were averaged using the id column to get one record per patient.

id column was dropped post-aggregation.

2. Handling Multicollinearity
Features with correlation > 0.7 were removed.

Feature count was reduced from 755 to 287.

3. Feature Selection
Applied Chi-Square test via SelectKBest to retain the top 30 features.

Feature values were scaled to [0, 1] using Min-Max Scaling.

⚖️ Handling Class Imbalance
Class imbalance was addressed using Random Oversampling.

After oversampling, both classes had equal representation in the training set.

🔍 Exploratory Data Analysis
Used .info(), .describe(), and .isnull() for data understanding.

Visualized class distribution using a pie chart.

Identified the dataset to be balanced after preprocessing.

🤖 Model Training
Trained three machine learning models:

Logistic Regression (with balanced class weights)

XGBoost Classifier

Support Vector Classifier (SVC with RBF kernel)

Each model was evaluated using:

ROC AUC Score for both training and validation datasets

Logistic Regression performed best with:

High validation accuracy

Minimal overfitting

📈 Model Evaluation
Confusion Matrix was plotted for Logistic Regression:

True Positives: 35

True Negatives: 10

False Positives: 4

False Negatives: 2

Classification Report indicated:

Strong precision and F1-score for healthy class

Room for improvement in recall for unhealthy class

📌 Conclusion
The Logistic Regression model provided the most balanced and accurate predictions.

Feature selection and class balancing played a critical role in improving performance.

The model is suitable for early screening but may require further enhancement for clinical applications.

✅ Future Improvements
Use ensemble techniques or neural networks for better generalization

Incorporate time-series analysis for longitudinal patient data

Test on larger, real-world datasets for scalability and robustness
