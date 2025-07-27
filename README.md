# Application of Machine Learning Algorithm for Osteoporosis Disease Prediction System

## Osteoporosis Prediction System

This project focuses on **Exploratory Data Analysis (EDA)** and **Machine Learning Modeling** using the **Osteoporosis Dataset**. The dataset contains medical and lifestyle factors associated with osteoporosis, which can help in early detection and prevention.

Osteoporosis is a condition that weakens bones, making them fragile and more likely to break. Early detection through accurate diagnosis plays a crucial role in reducing fractures and improving quality of life. This dataset consists of various **clinical and lifestyle features** used to classify individuals as having **osteoporosis** or **normal bone density**.

---

## Goals:

1. Perform EDA to understand the distribution of features in the dataset and identify missing values.
2. Handling Null Values using Imputation Technique
3. Identify key patterns and correlations between bone density and risk factors.
4. Encode data using Label Encoding and One-Hot Encoding (OHE).
5. Develop Machine Learning models to classify individuals as having **Osteoporosis** or **Non-osteoporosis**
6. Evaluate the model using key performance metrics such as **Classification Report, AUC-ROC, Confusion Matrix, and 10-Fold Cross-Validation**.
7. Analyze the best models for classification.

---

## Dataset Overview:

- The dataset contains **medical and lifestyle factors**, including:
  - **Age, Gender, Bone Mineral Density (BMD), Calcium Intake, Physical Activity, Smoking, Alcohol Consumption**, etc.
- The target variable is **osteoporosis status**:
  - **Osteoporosis** - Low bone density and high fracture risk
  - **Non-Osteoporosis** - Healthy bone density
- Key features include:
  - **Bone Mineral Density (BMD), Calcium Levels, Vitamin D Levels, Exercise Frequency, BMI**, etc.

---

## Insights:

- This study successfully developed an osteoporosis prediction model using three classification algorithms: **Random Forest, Support Vector Machine (SVM), and Gradient Boosting**.
- **Gradient Boosting** achieved the best performance with:
  - **Accuracy:** 90.82%
  - **Precision:** 91.99%
  - **Recall:** 90.95%
  - **F1-Score:** 90.77%
- When data was transformed using **One-Hot Encoding**, Gradient Boosting's performance improved:
  - **Accuracy:** 91.07%
  - **Precision:** 92.32%
  - **Recall:** 91.21%
  - **F1-Score:** 91.02%
- **Gradient Boosting outperformed** SVM and Random Forest in handling complex patterns in osteoporosis data.
- The model has potential for **early osteoporosis detection**, aiding medical practitioners in decision-making.
- **Challenges faced:**
  - Sensitivity of SVM and Random Forest to data transformation complexity.
  - Higher computational cost and longer training time for Gradient Boosting.
- **Future research recommendations:**
  - Exploring other algorithms like **XGBoost** and **LightGBM** for computational efficiency.
  - Fine-tuning hyperparameters to optimize each model’s performance.
  - Validating the model on larger and more diverse datasets to assess generalizability.
- The study emphasizes that selecting the **right algorithm, such as Gradient Boosting,** is crucial for medical applications, particularly in **early detection of complex diseases like osteoporosis**.

---

## Machine Learning Implementation:

- Applied classification models:
  - **Random Forest Classifier**
  - **Support Vector Machine (SVM)**
  - **Gradient Boosting Classifier**
- **Data preprocessing steps:**
  - Handling missing values using **imputation**.
  - Encoding categorical features with **Label Encoding and One-Hot Encoding**.
  - Scaling numerical features for consistency.
- **Evaluation metrics** include:
  - **Accuracy, Precision, Recall, F1-score, and Confusion Matrix**.
- **Findings:**
  - **Gradient Boosting outperformed** other models, achieving the highest accuracy and recall.
  - **One-Hot Encoding improved Gradient Boosting's performance**, highlighting its ability to handle complex data relationships.
  - **Random Forest and SVM were more sensitive to data transformation**, affecting their predictive capabilities.
---

## Key Findings & Recommendations:
- **Findings:**
  - **Gradient Boosting outperformed** other models, achieving the highest accuracy and recall.
  - **One-Hot Encoding improved Gradient Boosting's performance**, highlighting its ability to handle complex data relationships.
  - **Random Forest and SVM were more sensitive to data transformation**, affecting their predictive capabilities.
- **Recommendations for Improvement:**
  - Testing additional models like **XGBoost** and **LightGBM** for better computational efficiency.
  - Further hyperparameter tuning to enhance performance.
  - Applying advanced feature engineering techniques to refine model inputs.



---

### Contact & Feedback:

If you have any suggestions or feedback, feel free to connect with me on LinkedIn or via email:

- **Email**: [wiryawansujana@gmail.com](mailto\:wiryawansujana@gmail.com)
- **LinkedIn**: https\://www\.linkedin.com/in/rajendra-artanto-4698a8306/

#MachineLearning #Osteoporosis #EDA #DataScience

