import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_model(model_path):
    with open(model_path, 'rb') as model_file:
        return pickle.load(model_file)

def plot_education_distribution(defaulters, save_path):
    education_defaulters_counts = defaulters['EDUCATION'].value_counts().sort_index()
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x=education_defaulters_counts.index, y=education_defaulters_counts.values, palette='viridis')
    plt.xlabel('Education Level')
    plt.ylabel('Number of Defaulters')
    plt.title('Distribution of Credit Card Defaulters Across Different Education Levels')
    plt.savefig(save_path)
    plt.close()

def plot_age_distribution(defaulters, save_path):
    plt.figure(figsize=(10, 6))
    sns.histplot(defaulters['AGE'], bins=30, kde=True, color='blue')
    plt.xlabel('Age')
    plt.ylabel('Number of Defaulters')
    plt.title('Age Distribution of Credit Card Defaulters')
    plt.savefig(save_path)
    plt.close()

def charts():
    # Ensure the 'static' directory exists
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    os.makedirs(static_dir, exist_ok=True)
    
    print(f"Saving charts to directory: {os.path.abspath(static_dir)}")

    # Load the dataset
    try:
        df = pd.read_csv("default of credit card clients.csv")
    except FileNotFoundError:
        print("Error: The dataset file was not found.")
        return

    # Filter defaulters
    defaulters = df[df['default payment next month'] == 1]

    # Generate plots
    try:
        plot_education_distribution(defaulters, os.path.join(static_dir, 'defaulters_education_distribution.png'))
        plot_age_distribution(defaulters, os.path.join(static_dir, 'defaulters_age_distribution.png'))
        print("Charts generated and saved successfully.")
    except Exception as e:
        print(f"Error during plotting: {str(e)}")

# Call the function to generate charts
if __name__ == '__main__':
    charts()