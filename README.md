# Suicidal ideation detection from text (ML + genetic algorithms)

Hi, I am Sai Subodh. I am sharing this on GitHub so people can see what I built and how it runs.

It is a small Flask site plus a Jupyter notebook: short English text goes in, the code builds numbers from it, uses a genetic algorithm to trim features, and trains a voting ensemble (Random Forest, Decision Tree, XGBoost) on labels from the public Suicide Watch data on Kaggle.

I wrote the Python, the wiring, and this readme. The UI skin under static/ is integrated template assets. The post text in the CSV comes from that dataset, not from me.

## Quick facts

- Name: Sai Subodh
- Email: saisubodh2812004@gmail.com
- What it is for: learning, demos, portfolio. Not for clinics, hiring, or emergency triage.
- License file: I am not shipping a LICENSE file here. The code is mine to show. If you want to reuse chunks, ask me first. Kaggle and each Python library still have their own rules.

## Safety

This app can only guess from patterns it saw in data. It is easy to get wrong answers, and it will never know a real person's context. Do not treat the output as therapy, diagnosis, or a reason to panic or judge someone.

If you or someone nearby might be in immediate danger, use your country's emergency number or a crisis line you trust. Do not rely on this software.

## What lives in this folder

- app.py: loads data, builds features, runs GA selection, trains the ensemble, serves pages
- GA.py: extra helpers around genetic feature selection
- SuicidalDetection.ipynb: where I experimented and plotted results
- templates/: HTML the Flask app renders
- static/: CSS, JS, images, and uploads/testData.csv as a tiny sample file
- requirements.txt: Python packages you need
- .env.example: copy to .env and fill in secrets locally (never commit .env)
- DatasetLink.txt: same Kaggle link as below
- run.bat: double-click on Windows if you like; it starts python app.py

## Files Git is told to ignore

- .env: your keys and passwords stay on your machine
- venv/, __pycache__/: virtualenv and cache, everyone rebuilds their own
- Dataset/Suicide_Detection.csv: huge download from Kaggle; GitHub also caps file size
- Database files ending in .db: local SQLite from the optional signup flow
- Files in model/ ending in .npy or .pkl: caches and dumps; they come back when you run the app or notebook
- Files in static/uploads/ except the sample: real uploads stay local; only testData.csv and .gitkeep are meant for Git
- Final_Project_Report.pdf: I keep my full write-up next to the project on disk, but I do not push the PDF so the repo stays lighter

## Getting the dataset

1. Open Suicide Watch on Kaggle: https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch
2. Download the CSV into Dataset/Suicide_Detection.csv, or set SUICIDE_DATASET_PATH in .env to wherever you put it
3. Follow Kaggle's own terms. I do not redistribute the raw dump from here.

## Install and run

Windows (Command Prompt or PowerShell):

    cd path\to\this-folder
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    copy .env.example .env

macOS / Linux:

    cd path/to/this-folder
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env

Then edit .env and set at least FLASK_SECRET_KEY (a long random string). Optional mail variables are in .env.example for signup OTP.

NLTK data once:

    python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"

Start the app:

    python app.py

First launch can feel slow while it builds the model/ cache. The app reads up to 10,000 rows from the CSV at startup (see app.py). For a quick batch test, upload static/uploads/testData.csv (semicolon-separated, same shape the upload route expects).

Notebook: install Jupyter if you need it (pip install notebook), then:

    jupyter notebook SuicidalDetection.ipynb

- Labels and text: Suicide Watch dataset on Kaggle (see link above)
- Code libraries: Flask, NumPy, pandas, scikit-learn, XGBoost, NLTK, sklearn-genetic, matplotlib, seaborn, joblib, and whatever pip pulls in
- UI skin: stock HTML/CSS/JS under static/ from the template pack I started from


