from flask import Flask, render_template, request
import numpy as np
import pickle
import MySQLdb

app = Flask(__name__, static_folder='static')

# Chargez le modèle de prédiction
model = pickle.load(open('finalModel.pkl', 'rb'))

# Fonction pour encoder les prédictions
def encode_predictions(pred):
    if pred == 0:
        return 'Medium'
    elif pred == 1:
        return 'High'
    elif pred == -1:
        return 'Low'
    else:
        return 'Unknown'

# Fonction pour se connecter à la base de données et insérer les données
def save_to_database(data):
    conn = MySQLdb.connect(host="localhost", user="root", passwd="", db="studentperformancelogin")
    cursor = conn.cursor()
    query = "INSERT INTO prediction_data (gender, stage_id, semester, relation, raisedhands, visited_resources, announcements_view, discussion, parent_answering_survey, parent_school_satisfaction, student_absence_days, prediction_category) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    cursor.execute(query, data)
    conn.commit()
    conn.close()

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
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

        # Faire des prédictions en utilisant le modèle
        prediction = model.predict(features_numeric)
        pred_message = encode_predictions(prediction)

        # Sauvegarder les données dans la base de données
        data = (gender, stage_id, semester, relation, raisedhands, visited_resources, announcements_view,
                discussion, parent_answering_survey, parent_school_satisfaction, student_absence_days, pred_message)
        save_to_database(data)

        return render_template('prediction.html', pred=pred_message)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/submit_contact_form', methods=['POST'])
def submit_contact_form():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        # Vous pouvez traiter le formulaire ici, par exemple, envoyer un e-mail, enregistrer les données dans une base de données, etc.

        # Vous pouvez également afficher un message de confirmation ou rediriger l'utilisateur vers une autre page après la soumission du formulaire.
        return render_template('contact_confirmation.html', name=name)
    

if __name__ == '__main__':
    app.run()
