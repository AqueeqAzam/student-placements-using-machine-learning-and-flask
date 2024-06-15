from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('student-placement-modle.pkl', 'rb'))

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    cgpa = float(request.form.get('cgpa'))
    iq = int(request.form.get('iq'))
    profile_score = int(request.form.get('profile_score'))

    # prediction
    result = model.predict(np.array([cgpa, iq, profile_score]).reshape(1,3))

    if result[0] == 1:
        result = 'you got placement'
    else:
        result = 'dont worry you may try other company'

    return render_template('index.html', res=result)




if __name__ == '__main__':
    app.run(debug=True, port=8000)
