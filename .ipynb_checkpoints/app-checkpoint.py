from flask import Flask, render_template,request
import pickle
import numpy as np
import sklearn

app=Flask(__name__, static_folder='static')


model=pickle.load(open('finalModel.pkl','rb'))

def encode_predictions(pred):
   
    encoded_predictions = ['high' if pred == 1 else ('low' if pred == -1 else 'meduim')]
    return encoded_predictions

print(model)
print(model.predict(np.array([18,0,0,1,1,27,41,49,4,0,0]).reshape(1, -1)))

@app.route('/')
def hello_world():
    return  render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    print('in predict')
    # Récupérer les données du formulaire
    gender = int(request.form['gender'])
    stage_id = int(request.form['StageID'])
    semester = int(request.form['Semester'])
    relation = int(request.form['Relation'])
    raisedhands = int(request.form['raisedhands'])
    visited_resources = int(request.form['VisITedResources'])
    announcements_view = int(request.form['AnnouncementsView'])
    discussion = int(request.form['Discussion'])
    parent_answering_survey = int(request.form['ParentAnsweringSurvey'])
    parent_school_satisfaction = int(request.form['ParentschoolSatisfaction'])
    student_absence_days = int(request.form['StudentAbsenceDays'])

    # Convertir les données en format numérique
    features = [gender, stage_id, semester, relation, raisedhands, visited_resources, announcements_view,
                discussion, parent_answering_survey, parent_school_satisfaction, student_absence_days]
    features_numeric = np.array(features, dtype=np.float64).reshape(1, -1)
    print(features_numeric)


    # Faire des prédictions en utilisant le modèle de régression logistique
    prediction = model.predict(features_numeric)

    # Accéder aux prédictions de chaque modèle à partir du VotingClassifier chargé
    y_pred_nb_loaded = model.estimators_[0].predict(features_numeric)
    y_pred_lr_loaded = model.estimators_[1].predict(features_numeric)
    y_pred_svm_loaded = model.estimators_[2].predict(features_numeric)

    # Préparer le message de prédiction pour l'affichage
    pred_message = f'Your final prediction is: {prediction}.\n'
    nb_message = f'Naive Bayes prediction: {y_pred_nb_loaded[0]}.'
    lr_message = f'Logistic Regression prediction: {y_pred_lr_loaded[0]}.'
    svm_message = f'SVM prediction: {y_pred_svm_loaded[0]}.'

    return render_template('prediction.html', pred=pred_message, nb=nb_message, lr=lr_message, svm=svm_message)

@app.route('/about')
def about():
    return render_template('about.html')


if __name__=='__main__':
    app.run()

    #pip install virtualenv
