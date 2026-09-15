# SemaTrace backend

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The MVP API runs at `http://localhost:8000`. Swagger docs are at `/docs`.
