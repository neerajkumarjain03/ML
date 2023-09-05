from flask import Flask, request, render_template
import numpy as np
import pickle
from sklearn.svm import SVC

# load the model

with open('svm.pkl','rb') as file:
    model_svm = pickle.load(file)

with open('classes_dict.pkl','rb') as filec:
    labels = pickle.load(filec)

# create the server
app = Flask(__name__)

@app.route('/',methods=['GET'])
def root():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    # get input from user

    City = str(request.form.get("City"))
    Target = float(request.form.get("Target"))
    TossWinner = str(request.form.get("TossWinner"))
    TossDecision = str(request.form.get('TossDecision'))
    BattingTeam = str(request.form.get("BattingTeam"))
    Runs_required = float(request.form.get("Runs_required"))
    Balls_left = float(request.form.get("Balls_left"))
    Current_runrate = float(request.form.get("Current_runrate"))
    Required_runrate = float(request.form.get("Required_runrate"))
    Wicket_left = float(request.form.get("Wicket_left"))
    BowlingTeam = str(request.form.get("BowlingTeam"))

    city = float(labels['City'][City])
    tossWinner = float(labels['TossWinner'][TossWinner])
    tossDecision = float(labels['TossDecision'][TossDecision])
    battingTeam = float(labels['BattingTeam'][BattingTeam])
    bowlingTeam = float(labels['BowlingTeam'][BowlingTeam])

    algo = request.form.get("algo")
    x = [city, tossWinner, tossDecision,Target,
         battingTeam, Runs_required,Balls_left,
         Current_runrate, Required_runrate,
         Wicket_left,bowlingTeam]
    prediction = model_svm.predict_proba([x])
    # prediction = model_svm.predict()
    # print(x)
    # print(prediction)

    return render_template(
        "result.html",
        prediction=np.round(prediction*100,2),
        City=City,TossWinner=TossWinner,TossDecision=TossDecision,Target=Target,BattingTeam=BattingTeam,Runs_required=Runs_required,Balls_left=Balls_left,Current_runrate=Current_runrate,Required_runrate=Required_runrate,Wicket_left=Wicket_left,BowlingTeam=BowlingTeam,algo=algo)


# start the server
app.run(port=4006,debug=True,host='0.0.0.0')

