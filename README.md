Project Overview

This project builds a Decision Tree Classifier to predict whether a customer will purchase a product or service based on their demographic and behavioral data. The dataset used is the Bank Marketing Dataset from the UCI Machine Learning Repository.

Dataset Information

The Bank Marketing Dataset consists of information about direct marketing campaigns conducted by a Portuguese banking institution. The goal is to predict whether a customer will subscribe to a term deposit.

Features:

Demographic Data: Age, Job, Marital Status, Education, etc.

Behavioral Data: Contact communication type, number of contacts performed, and previous campaign outcomes.

Economic Indicators: Consumer confidence index, employment variation rate, etc.

Target Variable:

y (Binary Output): Indicates whether the customer subscribed to the term deposit (yes or no).

Project Workflow

1. Data Preprocessing

Load and explore the dataset.

Handle missing values and perform data cleaning.

Encode categorical variables using label encoding or one-hot encoding.

Split the data into training and testing sets.

2. Model Training

Train a Decision Tree Classifier using Scikit-Learn.

Tune hyperparameters such as tree depth to avoid overfitting.

Evaluate the model using accuracy, precision, recall, and F1-score.

3. Model Evaluation

Generate confusion matrices and classification reports.

Visualize feature importance.

Compare results with other classification models if necessary.

4. Deployment

Save the trained model for future predictions.

Provide a script to predict new customer data.

Installation & Usage

Prerequisites

Ensure you have the following libraries installed:

pip install pandas numpy scikit-learn matplotlib seaborn

Running the Project

Clone the repository:

git clone https://github.com/yourusername/decision-tree-classifier.git
cd decision-tree-classifier

Run the preprocessing and training script:

python train_model.py

Predict new data:

python predict.py --input new_data.csv

File Structure

├── data
│   ├── bank.csv  # Dataset file
├── src
│   ├── preprocess.py  # Data preprocessing script
│   ├── train_model.py  # Model training script
│   ├── predict.py  # Script for predictions
├── models
│   ├── decision_tree.pkl  # Saved trained model
├── README.md  # Project Documentation

Results

The trained Decision Tree Classifier achieved an accuracy of X%.

Feature importance analysis revealed that feature_1, feature_2, etc. had the highest influence on predictions.

Model performance was evaluated using confusion matrices and precision-recall curves.

Contributing

If you'd like to contribute, feel free to fork the repository and submit a pull request.

License

This project is licensed under the MIT License.

References

UCI Machine Learning Repository - Bank Marketing Dataset

Scikit-Learn Decision Tree Classifier
