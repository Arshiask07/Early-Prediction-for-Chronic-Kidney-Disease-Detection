\# Chronic Kidney Disease Prediction



Predicts whether a patient has chronic kidney disease from 8 lab values, using a Stochastic Gradient Boosting classifier picked out of 10 candidates, served through a Flask web app.



\## How it works



Training happens in `Early prediction of kidney disease.ipynb`. Data comes from Kaggle's `mansoordaku/ckdisease` dataset (400 rows, 25 features plus a patient ID that gets dropped immediately after loading) via `kagglehub`.



Preprocessing:

\- Numeric columns that got parsed as strings (`packed\_cell\_volume`, `white\_blood\_cell\_count`, `red\_blood\_cell\_count`) get coerced to numeric.

\- Split 80/20 (`train\_test\_split`, `random\_state=42`, stratified on `classification`) before any imputation, so the test set never leaks into the fill values.

\- Missing values filled with the train-set mean (numeric) or mode (categorical), computed on `train\_df` and applied to both splits.

\- All categorical columns label-encoded with `LabelEncoder`, fit on train only.



Feature selection: `RFE` with a `RandomForestClassifier(n\_estimators=100, random\_state=42)` as the estimator, cut down to 8 features. What it picked: `specific\_gravity`, `albumin`, `blood\_glucose\_random`, `serum\_creatinine`, `haemoglobin`, `packed\_cell\_volume`, `red\_blood\_cell\_count`, `hypertension`.



10 classifiers get trained on those 8 features, each with its own `GridSearchCV` (5-fold): KNN, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, Stochastic Gradient Boosting, XGBoost, CatBoost, Extra Trees, LightGBM. KNN is the only one wrapped in a `StandardScaler` pipeline, the tree-based models train on the raw label-encoded values.



\## Known issues



\- Extra Trees and AdaBoost both hit 100% test accuracy (see Results), but the model actually shipped is Stochastic Gradient Boosting at 96.25%. Two classifiers scoring perfectly on an 80-row test set reads more like overfitting to a small split than a genuinely better model, but the notebook doesn't state that reasoning anywhere, it just trains SGB again in the deployment cell and pickles it.

\- `catboost\_info/` (CatBoost's training logs) is in the uploaded project despite being listed in `.gitignore`. Gitignore only stops new commits, it won't remove something already tracked, if this got committed before the ignore rule was added, `git rm -r --cached catboost\_info/` first.

\- `results/output -4.png` is actually a screenshot of the input form's HTML5 validation message ("Please fill out this field"), not a fourth prediction result. Naming is inconsistent with the three input/output pairs before it.



\## Results



Test accuracy across all 10 models (from the notebook, 80-row held-out test set):



| Model | Accuracy |

|---|---|

| Extra Trees Classifier | 1.0000 |

| Ada Boost Classifier | 1.0000 |

| Decision Tree Classifier | 0.9625 |

| KNN | 0.9625 |

| Gradient Boosting Classifier | 0.9625 |

| Random Forest Classifier | 0.9625 |

| Stochastic Gradient Boosting | 0.9625 |

| XGBoost | 0.9625 |

| CatBoost | 0.9625 |

| LightGBM | 0.9625 |



Stochastic Gradient Boosting is what's deployed (`best\_sgb`, saved as `models/CKD.pkl`). Test set is 50 CKD / 30 not-CKD (matches the dataset's actual class split at 20% held out). Confusion matrix: `\[\[50, 0], \[3, 27]]`, rows are true CKD then true not-CKD. Every one of the 50 real CKD cases got caught (100% recall on CKD), the 3 errors are the other direction: 3 of the 30 not-CKD patients got flagged as CKD (94% precision on CKD, 90% recall on not-CKD).



App screenshots in `results/` show three manual test cases against the running Flask app:



| Case | specific\_gravity | albumin | blood\_glucose | serum\_creatinine | haemoglobin | packed\_cell\_volume | rbc\_count | hypertension | Result |

|---|---|---|---|---|---|---|---|---|---|

| 1 | 1.020 | 0 | 95 | 0.9 | 15.0 | 45 | 5.2 | no | Not CKD |

| 2 | 1.010 | 4 | 210 | 5.8 | 8.5 | 26 | 3.1 | yes | CKD |

| 3 | 1.015 | 1 | 140 | 1.6 | 11.5 | 36 | 4.2 | no | CKD |



Case 3 is worth noting: only mildly elevated creatinine (1.6) and albumin (1), normal-ish everything else, no hypertension, and the model still flagged it as CKD. That lines up with the confusion matrix above, the model leans toward flagging borderline profiles as CKD rather than clearing them, which is why it never misses a real case (100% recall) at the cost of a few false alarms (94% precision).



\## Running it



Train the model:

```bash

pip install -r requirements.txt

jupyter notebook "Early prediction of kidney disease.ipynb"

```



Run the app (uses the already-trained pickles in `models/`):

```bash

cd app

python app.py

```

Serves at `127.0.0.1:5000`. `/` is the home page, `/Prediction` is the input form, `/predict` handles the POST and renders the result page.



\## Files



```

Early prediction of kidney disease.ipynb   full training pipeline

requirements.txt

.gitignore

app/

&#x20; app.py                    Flask app, loads models/ and serves predictions

&#x20; templates/

&#x20;   home.html

&#x20;   indexnew.html           input form

&#x20;   result.html             CKD / not-CKD branch on prediction\_text

&#x20; static/css/

&#x20;   style.css, style-predict.css, style-result.css

&#x20;   images/kidneyhealth.png

models/

&#x20; CKD.pkl                   trained Stochastic Gradient Boosting model

&#x20; label\_encoders.pkl        per-feature LabelEncoders, fit on train

&#x20; target\_encoder.pkl        LabelEncoder for ckd/notckd

results/

&#x20; home page.png

&#x20; input -1.png, output-1.png     not-CKD case

&#x20; input -2.png, output -2.png    CKD case

&#x20; input -3.png, output -3.png    borderline CKD case

&#x20; output -4.png                  form validation screenshot (mislabeled, see Known issues)

catboost\_info/              CatBoost training logs, gitignored

Documents/                  phase-by-phase project docs (proposal, data quality report,

&#x20;                            feature selection report, model selection report, tuning report)

```



\## If picking this back up



\- Run `git rm -r --cached catboost\_info/` if it's already tracked, the gitignore entry alone won't remove it.

\- Rename `results/output -4.png` to something like `validation-error.png` and drop it from the input/output numbering, or move it out of `results/` entirely since it's not a prediction result.

\- Worth writing a short note (in this README or the model selection report) on why SGB was deployed over the two 100%-accuracy models, right now that reasoning only lives in whoever wrote the notebook's head.

