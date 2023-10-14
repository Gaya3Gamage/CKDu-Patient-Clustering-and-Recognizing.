from flask import Flask,render_template,request
import joblib
import numpy as np
from keras.models import load_model
from keras import backend as k
from sklearn.preprocessing import StandardScaler

model = load_model('models/best_model_top10.h5')
scaler_data = joblib.load('models/scaler_data_top9.sav')

app=Flask(__name__)
@app.route('/')
def index():
	return render_template('patient_details.html')
@app.route('/getresults',methods=['POST'])
def getresults():
	result=request.form
	print(result)

	name=result['name']
	V=float(result['V'])
	Hg=float(result['Hg'])
	Cs=float(result['Cs'])
	As=float(result['As'])

	Bi=float(result['Bi'])
	Ba=float(result['Ba'])
	Ca=float(result['Ca'])
	Na=float(result['Na'])
	Zn=float(result['Zn'])

	test_data = np.array([V, Hg, Cs, As, Bi, Ba, Ca, Na, Zn]).reshape(1, -1)

	scaled_test_data = scaler_data.transform(test_data)

	prediction = model.predict(scaled_test_data)
	prediction = np.argmax(prediction, axis=1)
	resultDict = {"name": name, "Status": prediction}

	return render_template('patient_result.html',results=resultDict)

app.run(debug=True)



 
 