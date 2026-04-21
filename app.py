"""
Flask application: batch scoring of short English text with a learned ensemble.

Pipeline (see README for context and safety limits): load labelled CSV rows,
build TF-IDF and auxiliary numeric features, run genetic-algorithm masks over
feature spaces, fit a hard-voting ensemble, expose HTML routes and CSV upload.

Author: Sai Subodh. Kaggle data and Python libraries follow their own terms.
"""
import os
import random
import secrets
import smtplib
import sqlite3
from email.message import EmailMessage
from string import punctuation

import nltk
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session
from genetic_selection import GeneticSelectionCV
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tag import pos_tag
from nltk.tokenize import sent_tokenize
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from werkzeug.utils import secure_filename
from xgboost import XGBClassifier

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

UPLOAD_FOLDER = os.path.join("static", "uploads")
ALLOWED_EXTENSIONS = {"csv"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()


def cleanText(doc):
    tokens = doc.split()
    table = str.maketrans("", "", punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [ps.stem(token) for token in tokens]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(tokens)


def POS(sentence):
    output = ""
    words = nltk.word_tokenize(sentence)
    words = [word for word in words if word not in stop_words]
    tags = pos_tag(words)
    for word, tag in tags:
        output += word + " " + tag + " "
    return output.strip()


def getStatistics(sentence):
    sentence = sentence.strip()
    total_sentences = len(sent_tokenize(sentence))
    words = len(nltk.word_tokenize(sentence))
    paragraphs = len(sentence.split("\n\n"))
    return total_sentences, words, paragraphs, len(sentence)


def getTopics(sentence):
    temp = sent_tokenize(sentence)
    topics = ""
    try:
        tfidf = TfidfVectorizer(
            stop_words=stop_words,
            use_idf=True,
            smooth_idf=False,
            norm=None,
            decode_error="replace",
        )
        temp = tfidf.fit_transform(temp).toarray()
        feature_names = (
            tfidf.get_feature_names_out()
            if hasattr(tfidf, "get_feature_names_out")
            else tfidf.get_feature_names()
        )
        lda = LatentDirichletAllocation(
            n_components=10,
            max_iter=5,
            learning_method="online",
            learning_offset=50.0,
            random_state=0,
        )
        lda.fit(temp)
        no_top_words = 10
        for topic_idx, topic in enumerate(lda.components_):
            words = " ".join(
                [feature_names[i] for i in topic.argsort()[: -no_top_words - 1 : -1]]
            )
            topics += words
            break
    except Exception:
        pass
    return topics


dataset_path = os.environ.get("SUICIDE_DATASET_PATH", "Dataset/Suicide_Detection.csv")
dataset = pd.read_csv(dataset_path, nrows=10000)

text_sentences = dataset["text"].ravel()
classes = dataset["class"].ravel()
labels = np.unique(dataset["class"])
original_X = []
linguistic_X = []
Y = []
statistics = []

if os.path.exists("model/X.npy"):
    original_X = np.load("model/X.npy")
    Y = np.load("model/Y.npy")
    statistics = np.load("model/statistics.npy")
    linguistic_X = np.load("model/linguistic.npy")
else:
    for i in range(len(text_sentences)):
        sentence = str(text_sentences[i]).strip()
        if len(sentence) > 0:
            total_sentences, total_words, total_paragraphs, total_characters = getStatistics(
                sentence
            )
            topics = getTopics(sentence)
            pos = POS(sentence)
            pos += topics
            pos = cleanText(pos.lower().strip())
            sentence = sentence.lower().strip() + " " + topics.lower().strip()
            sentence = cleanText(sentence)
            original_X.append(pos)
            linguistic_X.append(sentence)
            statistics.append(
                [total_sentences, total_words, total_paragraphs, total_characters]
            )
            if classes[i].strip().lower() == "suicide":
                Y.append(1)
            else:
                Y.append(0)
    original_X = np.asarray(original_X)
    linguistic_X = np.asarray(linguistic_X)
    Y = np.asarray(Y)
    statistics = np.asarray(statistics)
    os.makedirs("model", exist_ok=True)
    np.save("model/X", original_X)
    np.save("model/Y", Y)
    np.save("model/statistics", statistics)
    np.save("model/linguistic", linguistic_X)

print("Dataset Cleaning & Processing Completed")
print("Total Posts Found in Dataset = " + str(original_X.shape[0]))

names, count = np.unique(Y, return_counts=True)
height = count
bars = labels

original_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words=stop_words,
    use_idf=True,
    smooth_idf=False,
    norm=None,
    decode_error="replace",
    max_features=196,
)
original_X = original_vectorizer.fit_transform(original_X).toarray()
original_X = np.hstack([original_X, statistics])
linguistic_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words=stop_words,
    use_idf=True,
    smooth_idf=False,
    norm=None,
    decode_error="replace",
    max_features=160,
)
linguistic_X = linguistic_vectorizer.fit_transform(linguistic_X).toarray()
print(
    "Features Extracted from TEXT using TFIDF & NGRAM for both Original & Linguistics = "
    + str(original_X.shape)
)


def runOriginalGA():
    global original_X, Y
    if os.path.exists("model/original_ga.npy"):
        selector = np.load("model/original_ga.npy")
    else:
        estimator = RandomForestClassifier()
        selector = GeneticSelectionCV(
            estimator,
            cv=5,
            verbose=1,
            scoring="accuracy",
            max_features=86,
            n_population=10,
            crossover_proba=0.5,
            mutation_proba=0.2,
            n_generations=5,
            crossover_independent_proba=0.5,
            mutation_independent_proba=0.05,
            tournament_size=3,
            n_gen_no_change=10,
            caching=True,
            n_jobs=-1,
        )
        ga_selector = selector.fit(original_X, Y)
        selector = ga_selector.support_
        np.save("model/original_ga", selector)
    return selector


def runLinguisticGA():
    global linguistic_X, Y
    if os.path.exists("model/linguistic_ga.npy"):
        selector = np.load("model/linguistic_ga.npy")
    else:
        estimator = RandomForestClassifier()
        selector = GeneticSelectionCV(
            estimator,
            cv=5,
            verbose=1,
            scoring="accuracy",
            max_features=59,
            n_population=10,
            crossover_proba=0.5,
            mutation_proba=0.2,
            n_generations=5,
            crossover_independent_proba=0.5,
            mutation_independent_proba=0.05,
            tournament_size=3,
            n_gen_no_change=10,
            caching=True,
            n_jobs=-1,
        )
        ga_selector = selector.fit(linguistic_X, Y)
        selector = ga_selector.support_
        np.save("model/linguistic_ga", selector)
    return selector


original_ga = runOriginalGA()
linguistic_ga = runLinguisticGA()
original_X = original_X[:, original_ga]
linguistic_X = linguistic_X[:, linguistic_ga]
print("Original Text Features Size after applying GA = " + str(original_X.shape[1]))
print("Linguistic Text Features Size after applying GA = " + str(linguistic_X.shape[1]))

original_X_train, original_X_test, original_y_train, original_y_test = train_test_split(
    original_X, Y, test_size=0.2
)
linguistic_X_train, linguistic_X_test, linguistic_y_train, linguistic_y_test = (
    train_test_split(linguistic_X, Y, test_size=0.2)
)
print("Dataset Train & Test Split")

rf = RandomForestClassifier()
dt = DecisionTreeClassifier()
xg = XGBClassifier(max_depth=60, learning_rate=0.05)
linguistic_extension_model = VotingClassifier(
    estimators=[("rf", rf), ("dt", dt), ("xg", xg)], voting="hard"
)
linguistic_extension_model.fit(linguistic_X_train, linguistic_y_train)
predict = linguistic_extension_model.predict(linguistic_X_test)


@app.route("/home", methods=["GET", "POST"])
def home():
    return render_template("home.html")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/notebook")
def notebook():
    return render_template("SuicidalDetection.html")


@app.route("/PredictAction", methods=["GET", "POST"])
def PredictAction():
    if request.method != "POST":
        return render_template("result.html", msg="")
    f = request.files.get("file")
    if not f or not f.filename:
        return render_template("result.html", msg="<p>No file selected.</p>")
    data_filename = secure_filename(f.filename)
    if not data_filename.lower().endswith(".csv"):
        return render_template("result.html", msg="<p>Please upload a .csv file.</p>")
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], data_filename)
    f.save(save_path)
    session["uploaded_data_file_path"] = save_path
    data_file_path = session.get("uploaded_data_file_path")
    testData = pd.read_csv(data_file_path, sep=";")
    testData = testData.values
    temp = []
    for i in range(len(testData)):
        sentence = testData[i, 0]
        topics = getTopics(sentence)
        sentence = sentence.lower().strip() + " " + topics.lower().strip()
        sentence = cleanText(sentence)
        temp.append(sentence)
    temp = linguistic_vectorizer.transform(temp).toarray()
    temp = temp[:, linguistic_ga]
    predict = linguistic_extension_model.predict(temp)
    output = ""
    for i in range(len(predict)):
        output += (
            "Test Data = "
            + str(testData[i])
            + " <font size='3' color='blue'>Predicted As =====> "
            + str(labels[predict[i]])
            + "</font><br/><br/>"
        )
    return render_template("result.html", msg=output)


@app.route("/logon")
def logon():
    return render_template("signup.html")


@app.route("/login")
def login():
    return render_template("signin.html")


def _send_otp_email(to_addr, otp_code):
    """Send OTP via SMTP if MAIL_USERNAME and MAIL_PASSWORD are set; else log for local dev."""
    mail_user = os.environ.get("MAIL_USERNAME")
    mail_pass = os.environ.get("MAIL_PASSWORD")
    if not mail_user or not mail_pass:
        print(
            f"[local dev] OTP for signup: {otp_code} (not emailed — set MAIL_USERNAME and MAIL_PASSWORD in .env to use Gmail SMTP)"
        )
        return
    msg = EmailMessage()
    msg.set_content("Your OTP is : " + str(otp_code))
    msg["Subject"] = "OTP"
    msg["From"] = mail_user
    msg["To"] = to_addr
    with smtplib.SMTP(os.environ.get("MAIL_SMTP_HOST", "smtp.gmail.com"), int(os.environ.get("MAIL_SMTP_PORT", "587"))) as s:
        s.starttls()
        s.login(mail_user, mail_pass)
        s.send_message(msg)


@app.route("/signup")
def signup():
    global otp, username, name, email, number, password
    username = request.args.get("user", "")
    name = request.args.get("name", "")
    email = request.args.get("email", "")
    number = request.args.get("mobile", "")
    password = request.args.get("password", "")
    otp = random.randint(1000, 5000)
    _send_otp_email(email, otp)
    return render_template("val.html")


@app.route("/predict_lo", methods=["POST"])
def predict_lo():
    global otp, username, name, email, number, password
    if request.method == "POST":
        message = request.form["message"]
        if int(message) == otp:
            con = sqlite3.connect("signup.db")
            cur = con.cursor()
            cur.execute(
                "insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",
                (username, email, password, number, name),
            )
            con.commit()
            con.close()
            return render_template("signin.html")
    return render_template("signup.html")


@app.route("/signin")
def signin():
    mail1 = request.args.get("user", "")
    password1 = request.args.get("password", "")
    con = sqlite3.connect("signup.db")
    cur = con.cursor()
    cur.execute(
        "select `user`, `password` from info where `user` = ? AND `password` = ?",
        (mail1, password1),
    )
    data = cur.fetchone()
    con.close()

    if data is None:
        return render_template("signin.html")
    if mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("home.html")
    return render_template("signin.html")


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
