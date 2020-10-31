from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
app = Flask(__name__)
filename="finalized_model.sav"
infile=open(filename,"rb")
model=pickle.load(infile)
feature=['Pclass','Sex','Family_Size','Age_Bin','Capt','Col','Countess','Don','Dona','Dr','Jonkheer','Lady','Major','Master','Miss','Mlle','Mme','Mr','Mrs','Ms','Rev','Sir']
ip=[0]*22
@app.route('/')
def student():
   return render_template('titanic2.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      result = request.form
      ip[0]=int(result['pclass'])
      ip[1]=int(result['gender'])
      age=int(result["age"])
      if(age in range(0,20)):
      	ip[3]=0
      elif(age in range(20,28)):
      	ip[3]=1
      elif(age in range(20,28)):
      	ip[3]=2
      else:
      	ip[3]=3

      ip[2]=int(result["fsize"])
      ip[feature.index(result["title"])]=1
      ip2=np.array(ip)
      ip3=pd.DataFrame(ip2.reshape(1,-1))
      ip3.columns=feature
      pred=model.predict(ip3)
      pred2=model.transform(ip3)
      percent=(np.sum(pred2)/6)
      if(percent==0/6):
      	prob="Result : Impossible Survival odds"
      elif(percent==1/6):
      	prob="Result : Survival is Very Unlikely"
      elif(percent==2/6):
      	prob="Result : Survival is Unlikely"
      elif(percent==3/6):
      	prob="Result : Moderate chance of survival"
      elif(percent==4/6):
      	prob="Result : Survival is likely"
      elif(percent==5/6):
      	prob="Result : Survival is Very likely"
      else:
      	prob="Result : Survival is Certain"
      return render_template("titanic2.html",result = prob)

# app.run(debug = True)
