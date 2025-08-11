# Deploy SATutor (Flask) on Render

This app is a Python/Flask server (`app.py`). These files are included to make deployment on Render one-click simple:

- `requirements.txt` – Python dependencies
- `Procfile` – Start command for Render (Gunicorn)
- `runtime.txt` – Pin Python version (3.11)

## 1) Push to GitHub
Push the folder `sat backup/` (or its contents) to a new GitHub repository.

## 2) Create a Web Service on Render
- Dashboard → New → Web Service → Connect your GitHub repo
- Build command:
  ```bash
  pip install -r requirements.txt
  ```
- Start command:
  ```bash
  gunicorn app:app --bind 0.0.0.0:$PORT --worker-tmp-dir /dev/shm --timeout 120
  ```
- Environment: Python 3.11 (render detects from `runtime.txt`)

## 3) Files used by the app
- `questions.csv` should be present in the repo so it’s available at runtime.
- If you plan to upload files or persist user data, attach a Render Disk and write to that mount.

## 4) Custom domain
- In the Render service → Settings → Custom domains → Add domain.
- Update your DNS as instructed. Render will provision TLS automatically.

## 5) Notes
- Gunicorn is required for production serving; Flask dev server is not used.
- `torch`, `transformers`, and `accelerate` can be heavy; the first build may take longer.
- If you hit memory limits on free tiers, consider a larger instance.
