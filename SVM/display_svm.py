import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("default of credit card clients.csv")

# Drop the 'ID' column
df = df.drop(columns=['ID'])

# Define features and target variable
X = df[['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
        'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 
        'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']]
y = df['default payment next month']

# Split the data into training and testing sets
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the SVM model
with open('svm_model.pkl', 'rb') as model_file:
    loaded_svm = pickle.load(model_file)

# Use the loaded model to make predictions
y_pred_loaded = loaded_svm.predict(X_test)

# Evaluate the loaded model
print(f"Accuracy of loaded model: {accuracy_score(y_test, y_pred_loaded)}")
print(classification_report(y_test, y_pred_loaded))

# Filter data to show only defaulters (where the target variable is 1)
defaulters = df[df['default payment next month'] == 1]

# Count the number of defaulters in each education level
education_defaulters_counts = defaulters['EDUCATION'].value_counts().sort_index()

# Create a bar plot for defaulters across education levels
plt.figure(figsize=(8, 6))
sns.barplot(x=education_defaulters_counts.index, y=education_defaulters_counts.values, palette='viridis')

# Add labels and title
plt.xlabel('Education Level')
plt.ylabel('Number of Defaulters')
plt.title('Distribution of Credit Card Defaulters Across Different Education Levels')
plt.show()

# Create a histogram for the age distribution of defaulters
plt.figure(figsize=(10, 6))
sns.histplot(defaulters['AGE'], bins=30, kde=True, color='blue')

# Add labels and title
plt.xlabel('Age')
plt.ylabel('Number of Defaulters')
plt.title('Age Distribution of Credit Card Defaulters')

# Show the plot
plt.show()
