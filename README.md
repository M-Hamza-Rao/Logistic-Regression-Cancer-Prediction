# Logistic Regression Cancer Prediction
This project uses Logistic Regression, a fundamental classification algorithm in machine learning, to predict whether a tumor is malignant or benign based on features extracted from medical data. The dataset used is the Wisconsin Breast Cancer Dataset, a well-known dataset in the field of medical data analysis and machine learning.

📂 Project Structure
Main.ipynb: The main Jupyter notebook containing all data processing, visualization, model training, and evaluation steps.

(Optional additions):

requirements.txt: Python dependencies for setting up the environment.

data/: Directory where the dataset resides or is downloaded.

🧠 Model Used
Logistic Regression (from sklearn.linear_model)

Binary classification: Predicts Malignant or Benign.

📊 Dataset
Wisconsin Breast Cancer Dataset

Features: Mean, standard error, and worst measurements of radius, texture, perimeter, area, smoothness, etc.

Target: diagnosis (M = malignant, B = benign)

📈 Workflow Overview
Data Import and Exploration

Load the dataset using pandas

Check for null values and understand the distribution of classes

Data Preprocessing

Encode categorical variables (M/B to 1/0)

Standardize features using StandardScaler

Split dataset into training and test sets

Model Training

Fit a Logistic Regression model on the training data

Model Evaluation

Evaluate using accuracy, confusion matrix, and classification report

Visualize results using seaborn and matplotlib

Conclusion

Discussion of model performance and limitations

🚀 Getting Started
Requirements
bash
Copy
Edit
pip install -r requirements.txt
Typical libraries used:

numpy

pandas

matplotlib

seaborn

scikit-learn

Run the Notebook
Launch Jupyter Notebook:

bash
Copy
Edit
jupyter notebook Main.ipynb
✅ Results
Achieved high accuracy on both training and test datasets

Strong separation between malignant and benign classes

Demonstrated that logistic regression is effective for binary classification tasks in healthcare data

📌 Future Improvements
Try other classification algorithms (e.g., Random Forest, SVM)

Perform feature selection to reduce overfitting

Cross-validation for more robust evaluation

📘 References
UCI Machine Learning Repository - Breast Cancer Wisconsin Dataset

Scikit-learn documentation

