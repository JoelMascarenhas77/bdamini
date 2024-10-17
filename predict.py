import pickle

def predict(input_data):
    with open('svm_model.pkl', 'rb') as model_file:
        loaded_svm = pickle.load(model_file)
    
  
    prediction = loaded_svm.predict([input_data])
    probabilities = loaded_svm.predict_proba([input_data])
    
    return [prediction,probabilities]
