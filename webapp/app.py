from flask import Flask,render_template,request
import joblib
import numpy as np
from keras.models import load_model
from keras import backend as K


model = load_model('models/model-004.keras')


scaler_data =joblib.load('models/scaler_data.sav')
scaler_target =joblib.load('models/scaler_target.sav')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('energy_details.html')

@app.route('/getresults',methods=['POST'])
def getresults():

	result=request.form 

	print(result)

	name=result['name']
	semester=float(result['semester'])
	gender=float(result['gender'])
	age=float(result['age'])
	tc=float(result['tc'])
	hdl=float(result['hdl'])
	smoke=float(result['smoke'])
	bpm=float(result['bpm'])
	diab=float(result['diab'])

	test_data=np.array([semester,gender,age,tc,hdl,smoke,bpm,diab]).reshape(1,-1)

	test_data=scaler_data.transform(test_data) 
	prediction=model.predict(test_data)

	prediction=scaler_target.inverse_transform(prediction) 
	print(prediction,prediction[0],prediction[0][0])
	
	resultDict={"name":name,"risk":round(prediction[0][0],2)}

	return render_template('energy_results.html',results=resultDict)

app.run(debug=True)