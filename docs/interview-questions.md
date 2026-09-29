# Interview Prep — ChurnOps

Beginner-friendly, technically accurate answers. Read
[learning-guide.md](learning-guide.md) first for the full context
behind these.

**1. What is MLOps?**
The set of practices and tools for taking a trained ML model and
reliably tracking, versioning, deploying, serving, and monitoring it
in production — the engineering discipline around the model, not the
model itself.

**2. Why did you choose Logistic Regression?**
It's simple, fast to train, and its coefficients are directly
interpretable (which feature pushes churn risk up or down). Since this
project's goal is to demonstrate the MLOps lifecycle, not to win a
Kaggle competition, a simple, well-understood model lets the focus
stay on tracking/serving/monitoring rather than model tuning.

**3. Why not use a deep learning model?**
On small tabular data like this (7K rows, 8 features), deep learning
offers no real accuracy advantage over Logistic Regression or a tree
model, but adds real cost: harder to interpret, more hyperparameters,
slower to train and serve, and overkill for the problem size. Using
the right-sized tool is itself a good engineering signal.

**4. What is MLflow?**
An open-source platform for ML experiment tracking and model
management: it records the parameters, metrics, and artifacts of
training runs, and hosts a Model Registry for versioning models.

**5. What is an MLflow experiment?**
A named container that groups related training runs together — in
this project, all churn-model training runs live in one experiment
called `churnops`.

**6. What is an MLflow run?**
One execution of a training script. Each run has its own parameters,
metrics, and artifacts, and is timestamped and stored independently,
even if it's part of the same experiment.

**7. What is a model artifact?**
Any file produced during a run that's worth keeping — in this
project, the confusion matrix PNG, the classification report text
file, and the serialized model pipeline itself.

**8. What is Model Registry?**
A versioned, named catalog of models inside MLflow, separate from
individual runs. It lets you track a model's lifecycle (which version
is a candidate, which is in production) independent of the
experiment/run that produced it.

**9. What is a model version?**
An immutable, numbered snapshot of a registered model (v1, v2, v3,
...). Registering a new model never overwrites an old version — it
adds a new one, so you can always go back.

**10. What is a model alias?**
A movable, named pointer (e.g. `production`, `candidate`) to one
specific model version. Instead of hardcoding "load version 7,"
serving code loads "whatever version currently holds the `production`
alias" — so promoting a model is just moving the alias.

**11. Why use a model registry?**
It decouples "which model version exists" from "which model version is
live." Without it, deploying a new model usually means manually
copying files and restarting services — risky and hard to audit or
roll back. The registry makes promotion, rollback, and history
explicit and safe.

**12. Why FastAPI?**
It's fast to write, has built-in request/response validation via
Pydantic, and generates interactive API docs (`/docs`) automatically
— low ceremony for a service this size.

**13. Why Docker?**
To eliminate "works on my machine" — Docker packages the exact Python
version, dependencies, and code into one image that runs identically
anywhere Docker runs, including CI runners and real servers.

**14. What is CI?**
Continuous Integration — automatically running checks (lint, tests,
build) on every code change, so problems are caught immediately
instead of relying on a human to remember to check.

**15. What is CD?**
Continuous Deployment — automatically shipping a change that passes CI
to a real running environment. This project implements CI only; there
is no cloud target to deploy to here (see README's "Future
improvements").

**16. What does GitHub Actions do in this project?**
On every push/PR, it installs dependencies, runs Ruff (lint), runs
Pytest (tests), builds the Docker image, and does a smoke test —
starting the container and hitting `/health` — before reporting pass
or fail.

**17. What is Prometheus?**
A metrics collection and storage system. It "scrapes" (pulls) numeric
metrics from an app's `/metrics` endpoint on a schedule and stores
them as a queryable time series.

**18. What is Grafana?**
A visualization tool that reads from a data source like Prometheus
and renders dashboards — graphs, stats, alerts — so metrics are
readable at a glance instead of requiring manual queries.

**19. What is data drift?**
A change in the statistical distribution of input data over time,
relative to what a model was trained on — e.g. average customer tenure
shifting, or a new contract type becoming common. The model itself
doesn't change, but its assumptions about the world become stale.

**20. How does Evidently detect drift?**
It statistically compares the distribution of each shared column
between a reference dataset and a current dataset (e.g. a
Kolmogorov-Smirnov test for numeric columns) and flags columns whose
distributions differ beyond a threshold, then compiles the results
into an HTML report.

**21. What is DVC?**
Data Version Control — a tool that versions large data files alongside
Git, without storing the files themselves in Git. It stores a small
pointer file (content hash) in Git while the actual data lives in a
separate cache/storage.

**22. Git vs DVC?**
Git efficiently tracks and diffs small text files (code) and their
full history. DVC handles large binary files (datasets, models) that
Git handles poorly, while still tying each data version to a specific
Git commit via the pointer file.

**23. Why use Pipeline and ColumnTransformer?**
Bundling preprocessing (scaling, one-hot encoding) and the model into
a single scikit-learn `Pipeline` means one object handles both steps
consistently, and MLflow logs/loads that one object — guaranteeing
inference always preprocesses data exactly the way training did.

**24. Why shouldn't preprocessing be duplicated in the API?**
If the API reimplemented preprocessing separately from training, the
two implementations could quietly drift apart over time (e.g. someone
updates one but forgets the other) — a subtle, hard-to-detect bug
where the model receives differently-shaped data at serving time than
it saw during training, silently degrading predictions.

**25. What happens when a new model is trained?**
`src/train.py` fits a new pipeline, evaluates it, logs the run to
MLflow, and registers the resulting model as a new version of
`churn-model` — it does not automatically become the live model.

**26. How is the production model selected?**
Manually: after evaluating a candidate version (`src/evaluate.py`), a
human runs `src/promote_model.py` to point the `production` alias at
that version. The API always loads whichever version the `production`
alias currently points to.

**27. What happens if the model fails to load?**
The API doesn't crash. `/health` and `/docs` keep working so the
service stays diagnosable; `/predict` and `/model-info` return a `503`
until a model becomes available, rather than returning wrong answers
or an unhandled 500 error.

**28. How would you deploy this to the cloud?**
Push the Docker image to a container registry (e.g. ECR), run it on a
managed container service (e.g. ECS or Cloud Run), and replace local
MLflow storage (SQLite + local files) with a managed backend (e.g.
RDS for the tracking database, S3 for artifacts).

**29. How would you add Kubernetes later?**
Once there are multiple services to orchestrate, or a need for
autoscaling/self-healing across many replicas, you'd containerize each
component (already done here) and deploy them as Kubernetes
Deployments/Services, with the model-serving pod horizontally scaled
based on request load. For a single FastAPI service at this scale,
Kubernetes adds operational overhead without a corresponding benefit.

**30. How would you implement automatic retraining?**
Add a scheduled or drift-triggered job that reruns `src/train.py`,
evaluates the new candidate against the current production model on
the same test set, and — critically — requires a human approval step
before promotion, rather than auto-promoting. Fully automatic
promotion risks silently shipping a worse model if the trigger (e.g. a
drift signal) doesn't actually indicate the right fix.
