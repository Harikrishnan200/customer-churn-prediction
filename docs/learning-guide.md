# ChurnOps Learning Guide

This document teaches the project, not just describes it. Read it
top to bottom if you're learning MLOps for the first time.

## 1. What is ML, really?

Machine learning is: show a program lots of examples (rows of data
with a known answer), and it learns a mathematical function that
predicts the answer for new, unseen rows. In this project, the
examples are past customers (their tenure, charges, contract type,
...) and the known answer is whether they churned. The "learning" part
is `pipeline.fit(X_train, y_train)` in `src/train.py` — one line. That
one line is a small fraction of this entire project, and that's the
point of this guide.

## 2. What is MLOps, and why isn't training a model enough?

Training a model produces one file (or in-memory object) that maps
inputs to a prediction. On its own, that's not useful to anyone:

- Nobody else can call it — it's sitting in your Python process.
- Nobody knows which version it is, or what data/parameters produced it.
- Nobody finds out if it starts performing worse next month.
- Nobody can safely replace it with a better version without risk.

**MLOps** is the set of practices and tooling that takes "a model that
works on my machine" and turns it into "a model reliably serving
predictions, that the team can trust, improve, and safely update."
It borrows heavily from DevOps (version control, CI/CD, monitoring)
and adds ML-specific concerns on top (experiment tracking, model
versioning, data drift).

## 3. What happens after a model is trained? (The journey)

```
dataset → training → evaluation → experiment tracking →
model registry → API → Docker → CI → monitoring → drift detection
```

Walk through this project in that exact order:

1. **Dataset** (`data/raw/churn.csv`, versioned with DVC) — the raw
   material. Nothing works without trustworthy, versioned data.
2. **Training** (`src/train.py`) — fits a `Pipeline` (preprocessing +
   Logistic Regression) on the training split.
3. **Evaluation** — metrics computed on data the model didn't train
   on (validation set during training; test set in `src/evaluate.py`
   before promotion), so the numbers reflect real generalization, not
   memorization.
4. **Experiment tracking** (MLflow) — every training run's inputs and
   outputs get recorded automatically, so nothing is lost or forgotten.
5. **Model registry** (MLflow) — the model that came out of a good run
   gets a permanent, versioned home, separate from the run itself.
6. **API** (FastAPI) — makes the model callable over HTTP by anything
   (a website, another service, a script).
7. **Docker** — packages the API and its dependencies into one
   reproducible unit that runs the same on your laptop, a teammate's
   laptop, or a server.
8. **CI** (GitHub Actions) — automatically checks that new code
   doesn't break tests, lint rules, or the Docker build.
9. **Monitoring** (Prometheus + Grafana) — once live, tells you how the
   API is actually behaving: how many requests, how fast, how many
   errors, what it's predicting.
10. **Drift detection** (Evidently) — tells you when the *data* the
    model sees in the real world has quietly diverged from what it was
    trained on.

Each numbered step above is a section below.

## 4. Why do we need MLflow?

Without it: you run `train.py`, get 82% accuracy, tweak a
hyperparameter, run it again, get 79%, tweak something else, get 84%
— and a week later you have no idea which combination of settings
produced which number, or which `.pkl` file on your disk corresponds
to which result.

MLflow solves this by automatically recording, for every run: the
parameters you used, the metrics that came out, any files worth
keeping (plots, reports), and the model itself — all queryable later
through the MLflow UI at http://localhost:5001. See `src/train.py` for
exactly what gets logged and why, with comments explaining each MLflow
concept the first time it's used.

## 5. Why do we need a Model Registry?

MLflow experiment tracking answers "what happened during this run?"
The **Model Registry** answers a different question: "which model
version should currently be serving real traffic?"

Without a registry, "deploying a new model" usually means manually
copying a file and restarting a service — risky, hard to roll back,
and easy to lose track of which file is where. With a registry:

- Every registered model gets an immutable version number (v1, v2, v3, ...).
- An **alias** like `production` is a movable label you point at
  whichever version should currently be live.
- Promoting a new model = moving the label (`src/promote_model.py`) —
  no file copying, no code change, no redeploy of model artifacts.
- Rolling back = pointing the label at the previous version.

## 6. Why do we need FastAPI?

A trained model is a Python object. Most consumers of a prediction
(a website's backend, a mobile app, another internal service) aren't
Python processes that can just import your model — they need to call
it over the network. FastAPI turns the model into an HTTP service with
almost no boilerplate, validates incoming requests automatically via
Pydantic (`app/schemas.py`), and generates interactive API docs for
free at `/docs`.

## 7. Why Docker?

"Works on my machine" is a real, common failure: different Python
versions, missing system libraries, different OS behavior. Docker
packages the exact Python version, exact dependency versions, and
application code into one image that runs identically anywhere Docker
runs — your laptop, a teammate's laptop, a CI runner, or a real
server. See `Dockerfile` — it's short on purpose: install deps, copy
code, run `uvicorn`.

## 8. Why CI/CD?

**CI (Continuous Integration)**: every time code changes, automatically
run checks — lint, tests, does the Docker image even build — instead
of relying on a human to remember to do it locally before pushing.
Catches breakage immediately, for everyone.

**CD (Continuous Deployment)**: automatically ship a change that
passes CI to a real running environment. This project implements CI
only (see `.github/workflows/ci.yml`) — there's no cloud environment
to deploy to here, and adding one would mean adding cloud
infrastructure, which this project deliberately avoids to stay
learnable. The README's "Future improvements" section explains what CD
would add on top.

## 9. Why Prometheus?

Once an API is live, "is it working?" stops being a question you can
answer by looking at code — you need data about its actual behavior
over time: how many requests, how fast, how many failures. Prometheus
is built exactly for this: it periodically "scrapes" (pulls) numeric
metrics from a `/metrics` endpoint your app exposes
(`app/main.py` + `prometheus_client`), and stores them as a
time series you can query and alert on.

## 10. Why Grafana?

Prometheus stores numbers; Grafana turns them into readable graphs and
dashboards. Rather than someone hand-writing Prometheus queries every
time they want to check "is the API healthy," Grafana gives a
always-available visual dashboard (`monitoring/grafana/provisioning/`)
showing request rate, latency, errors, and churn predictions at a
glance.

## 11. Why data drift, and why detect it manually?

A model learns statistical patterns from a fixed snapshot of data. If
the real world changes — a marketing campaign shifts the typical
customer profile, pricing changes, a new contract type launches — the
model's learned patterns can become stale even though nothing about
the model *code* changed. This is **data drift**, and it's one of the
most common, hardest-to-notice causes of a model quietly getting worse
over time.

Evidently (`src/monitor.py`) compares a reference dataset (what
training data normally looks like) against a current dataset
(standing in for recent production traffic) and flags columns whose
distributions have shifted.

**Why not auto-retrain when drift is detected?** A drift signal tells
you data *changed*, not *why*, and not whether retraining is even the
right fix — it could just as easily mean a bug in an upstream data
pipeline, in which case retraining on bad data would make the model
*worse*, not better. A human should look at *what* drifted and *why*
before deciding retraining is the right response. Automating that
decision away is exactly the kind of "impressive-looking but risky"
complexity this project avoids.

## 12. Why DVC?

Git is built for tracking small text files (code) and their history
efficiently. Large binary files (datasets) don't fit that model well —
committing them bloats the repository and Git's diffing doesn't help
you understand what changed in a CSV.

DVC solves this by storing a tiny pointer file in Git (a content hash,
`data/raw/churn.csv.dvc`) while the actual data file lives in a
separate cache. Git tracks *code and pointers*; DVC tracks *data*. You
never need cloud storage to benefit from this — a local DVC cache
already gives you dataset versioning tied to Git commits.

## 13. Why is the model itself kept so simple?

Logistic Regression, on 8 features, is not a cutting-edge model. On
purpose. The skill being demonstrated by this project is *operating* a
model reliably — tracking it, versioning it, serving it, monitoring
it — and that skill transfers directly to a far more complex model.
Swapping Logistic Regression for a deep neural network wouldn't
demonstrate anything different about MLOps; it would just add
complexity to the 25% of the project that this project intentionally
keeps small.

## 14. How to explain this project in an interview

A tight, 60-second version:

> "ChurnOps is a small production-style MLOps pipeline for a churn
> classifier. The ML side is intentionally simple — a Logistic
> Regression pipeline trained on the IBM Telco dataset — because the
> project is really about the lifecycle around the model. Every
> training run is tracked in MLflow with its parameters, metrics, and
> artifacts, and good models get registered and versioned in the MLflow
> Model Registry. A FastAPI service loads whichever model version
> currently holds the `production` alias and serves predictions over
> REST, validated with Pydantic. It's containerized with Docker, and
> GitHub Actions runs linting, tests, and a Docker build-and-smoke-test
> on every push. In production, Prometheus scrapes request/latency/error
> metrics from the API and Grafana visualizes them. Separately, I use
> Evidently to compare live data against the training distribution and
> generate a drift report — deliberately as a human-in-the-loop signal,
> not an auto-retrain trigger, since drift alone doesn't tell you *why*
> the data changed. And the dataset itself is version-controlled with
> DVC alongside the code in Git."

If asked "what would you add for real production use," point to the
README's **Future improvements** section — cloud deployment, automatic
retraining with human approval, and A/B testing between model
versions are the natural next steps, deliberately left out here to
keep the project learnable.

See [interview-questions.md](interview-questions.md) for a full Q&A set.
