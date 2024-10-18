from flask import Flask, render_template, request, redirect, url_for, flash
from predict import predict
from charts import charts

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for flashing messages

@app.route('/')
def index():
    charts()  # Call the charts function to generate the charts
    return render_template('index.html')

@app.route('/predict')
def predict_form():
    return render_template('predict.html')

@app.route('/predict/submit', methods=['POST'])
def submit():
    try:
        data = {
            'LIMIT_BAL': int(request.form['LIMIT_BAL']),
            'SEX': int(request.form['SEX']),
            'EDUCATION': int(request.form['EDUCATION']),
            'MARRIAGE': int(request.form['MARRIAGE']),
            'AGE': int(request.form['AGE']),
            'PAY_0': int(request.form['PAY_0']),
            'PAY_2': int(request.form['PAY_2']),
            'PAY_3': int(request.form['PAY_3']),
            'PAY_4': int(request.form['PAY_4']),
            'PAY_5': int(request.form['PAY_5']),
            'PAY_6': int(request.form['PAY_6']),
            'BILL_AMT1': int(request.form['BILL_AMT1']),
            'BILL_AMT2': int(request.form['BILL_AMT2']),
            'BILL_AMT3': int(request.form['BILL_AMT3']),
            'BILL_AMT4': int(request.form['BILL_AMT4']),
            'BILL_AMT5': int(request.form['BILL_AMT5']),
            'BILL_AMT6': int(request.form['BILL_AMT6']),
            'PAY_AMT1': int(request.form['PAY_AMT1']),
            'PAY_AMT2': int(request.form['PAY_AMT2']),
            'PAY_AMT3': int(request.form['PAY_AMT3']),
            'PAY_AMT4': int(request.form['PAY_AMT4']),
            'PAY_AMT5': int(request.form['PAY_AMT5']),
            'PAY_AMT6': int(request.form['PAY_AMT6']),
        }

        data_list = list(data.values())
        prediction, probabilities = predict(data_list)
    
        return render_template('predict.html', prediction=prediction, probabilities=probabilities)

    except Exception as e:
        flash(f'Error: {str(e)}')
        return redirect(url_for('predict_form'))

if __name__ == '__main__':
    app.run(debug=True)
