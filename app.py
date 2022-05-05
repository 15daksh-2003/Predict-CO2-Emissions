#lets import the libraries
import numpy as np
from flask import Flask, render_template, request
import pickle

#lets initialize the flask app
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

#lets define the route for the default page
@app.route('/')
def home():
    return render_template('index.html')

#Lets redirect the api to predict the CO2 emission
#To use predict button in our web_app
@app.route('/predict',methods=['POST'])
def predict():
    #For rendering results on HTML GUI
    int_features=[float(x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    prediction=model.predict(final_features)
    output= round(prediction[0],2)
    return render_template('index.html',prediction_text='CO2 Emission of the vehicle is : {}'.format(output))

#starting the flask server
if __name__=="__main__":
    app.run(debug=True)