from flask import Flask, render_template, request
import numpy as np
import pickle
import MySQLdb

app = Flask(__name__, static_folder='static')

model = pickle.load(open('finalModel.pkl', 'rb'))

def encode_predictions(pred):
    if pred == 0:
        return 'Medium'
    elif pred == 1:
        return 'High'
    elif pred == -1:
        return 'Low'
    else:
        return 'Unknown'

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

        features = [gender, stage_id, semester, relation, raisedhands, visited_resources, announcements_view,
                    discussion, parent_answering_survey, parent_school_satisfaction, student_absence_days]
        features_numeric = np.array(features, dtype=np.float64).reshape(1, -1)

        prediction = model.predict(features_numeric)
        pred_message = encode_predictions(prediction)

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

        feedback_message = "Merci pour votre message. Nous vous répondrons dès que possible !"
        return render_template('contact.html', feedback_message=feedback_message)


if __name__ == '__main__':
    app.run()
