from flask import Flask, render_template, request, url_for
import numpy as np
import pickle
import pandas as pd

# Decision Tree Pickle
model1 = pickle.load(open("model/model_dt.pkl", "rb"))
# Linear Regression Pickle
model2 = pickle.load(open("model/model_lr.pkl", "rb"))
# Random Forest Pickle
#model3 = pickle.load(open("model/model_rf.pkl", "rb"))


app = Flask(__name__, template_folder="templates")


@app.route("/")
def main():
    return render_template('index.html')


# Decision Tree
@app.route('/predict1', methods=['POST'])
def predict1():
    '''
    For rendering results on HTML GUI
    '''
    features_dt = [x for x in request.form.values()]
    final_features_dt = [np.array(features_dt)]
    prediction_dt = model1.predict(final_features_dt)

    output_dt = round(prediction_dt[0], 2)

    return render_template('index.html', prediction_text_dt='Prediksi Tarif Decision Tree yaitu : $ {}'.format(output_dt))

# Linear regression


@app.route('/predict2', methods=['POST'])
def predict2():
    '''
    For rendering results on HTML GUI
    '''
    features_lr = [y for y in request.form.values()]
    final_features_lr = [np.array(features_lr)]
    prediction_lr = model2.predict(final_features_lr)

    output_lr = round(prediction_lr[0], 2)

    return render_template('index.html', prediction_text_lr='Prediksi Tarif Linear Regression yaitu : $ {}'.format(output_lr))


if __name__ == '__main__':
    app.run(debug=True)
