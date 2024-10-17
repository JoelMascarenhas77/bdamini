import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split



import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def charts():
   
    df = pd.read_csv("default of credit card clients.csv")
   
    random_sample = df.sample(n=100)
 
    random_sample = random_sample.drop(columns=['ID', 'default payment next month'])
   
    with open('svm_model.pkl', 'rb') as model_file:
        loaded_svm = pickle.load(model_file)

   
    y_pred_loaded = loaded_svm.predict(random_sample)


    defaulters = df[df['default payment next month'] == 1]

   
    education_defaulters_counts = defaulters['EDUCATION'].value_counts().sort_index()

   
    plt.figure(figsize=(8, 6))
    sns.barplot(x=education_defaulters_counts.index, y=education_defaulters_counts.values, palette='viridis')

   
    plt.xlabel('Education Level')
    plt.ylabel('Number of Defaulters')
    plt.title('Distribution of Credit Card Defaulters Across Different Education Levels')
    
    
    plt.savefig('static/defaulters_age_distribution.png')
    plt.close()  

   
    plt.figure(figsize=(10, 6))
    sns.histplot(defaulters['AGE'], bins=30, kde=True, color='blue')

    
    plt.xlabel('Age')
    plt.ylabel('Number of Defaulters')
    plt.title('Age Distribution of Credit Card Defaulters')
    
    
    plt.savefig('static/defaulters_education_distribution.png')
    plt.close()  
