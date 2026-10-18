# SMS Spam Classifier

A small first machine-learning project that predicts whether an SMS message is **spam** or **not spam**. It uses Python, scikit-learn and a simple Streamlit page. The code is deliberately split into a training script and an app so the steps are easy to follow.

> **Learning project:** this model is a demonstration. It can make mistakes and should not be used to block important messages automatically.

![Interface preview for the message checker](screenshots/demo-preview.png)

## What it does

1. Downloads the public SMS Spam Collection from the UCI Machine Learning Repository.
2. Removes rows with missing or empty messages, normalizes the labels, and removes exact duplicate rows.
3. Splits the data into training and test sets while preserving the spam/not-spam ratio.
4. Turns message text into TF-IDF numerical features and trains a Logistic Regression classifier.
5. Reports accuracy, precision, recall, F1-score, ROC-AUC and a confusion matrix on the test set.
6. Saves the trained pipeline so the Streamlit demo can classify new messages.

## Dataset and credit

The [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) contains 5,574 labelled SMS messages. The dataset is licensed **CC BY 4.0**. It is downloaded by the training script with the `ucimlrepo` Python package; it is not copied into this repository.

Citation: Almeida, T. & Hidalgo, J. (2011). *SMS Spam Collection* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84.

## Set up

You need Python 3.10 or newer. Open a terminal in this project folder and run:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install the packages:

```bash
python -m pip install -r requirements.txt
```

## Train and evaluate

```bash
python train_model.py
```

The first run needs an internet connection to download the dataset. The script prints the evaluation metrics and creates:

- `models/spam_classifier.joblib` — the complete TF-IDF + classifier pipeline.
- `results/metrics.json` — machine-readable test metrics and class counts.
- `results/confusion_matrix.png` — a test-set confusion matrix image.

The test set is kept separate from training. The text vectorizer is part of the scikit-learn pipeline, so it learns its vocabulary from training messages only.

## Run the demo

Train the model once, then start the local web app:

```bash
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`). Paste a message such as `Congratulations! You won a free iPhone. Claim now!` and select **Check message**.

## Project files

```text
spam-message-classifier/
├── app.py                   # Streamlit text-entry demo
├── train_model.py           # Download, clean, train, evaluate, and save
├── upload_to_github.py      # Python helper to create a public repo and upload
├── requirements.txt         # Python package list
├── screenshots/
│   └── demo-preview.png     # Preview image for this README
├── .gitignore
└── README.md
```

## How the model works

- **TF-IDF** gives more weight to words that help distinguish one message from another, and converts text to numbers.
- **Logistic Regression** learns patterns in those numbers to choose between `spam` and `ham` (not spam).
- The test set is not used to fit the model. Precision, recall and F1-score help explain performance beyond accuracy, especially because the dataset has many more legitimate messages than spam messages.

## Publish as a public GitHub repository

You can publish with the included Python helper. First create a GitHub personal access token with permission to create repositories. Keep the token private; the helper asks for it without showing it and does not save it. From this project folder, run:

```bash
python upload_to_github.py
```

Enter a repository name when prompted (or press Enter to use `sms-spam-classifier`). The helper creates a new **public** repository under the GitHub account belonging to the token, then uploads the project files in an initial commit. If the name is already used in your account, choose another name. You can also use the GitHub website to create the repository and Git commands to push manually.

```bash
git init
git add README.md app.py train_model.py upload_to_github.py requirements.txt .gitignore screenshots/
git commit -m "Build SMS spam classifier project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/sms-spam-classifier.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username. Do not commit `.venv/`, downloaded data, generated models, or private credentials.

## Possible next improvements

- Try Multinomial Naive Bayes and compare its test metrics.
- Add cross-validation or tune the decision threshold.
- Try the model on recent messages and other languages; this old English SMS dataset may not represent today's spam.
