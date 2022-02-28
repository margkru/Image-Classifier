import matplotlib.pyplot as plt
from flask import Flask, render_template
from test import test_model, predict

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/classifier')
def classifier():
    ds_test, model = test_model()
    accuracy = round(model.evaluate(ds_test, verbose=0)[1] * 100, 3)
    sample, pred = predict(ds_test, model)
    return render_template("classifier.html", acc=accuracy, sample=sample, pred=pred)

if __name__ == "__main__":
    app.run(debug=True)
