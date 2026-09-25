# ScamShield Backend

## PostgreSQL setup

This backend is configured for PostgreSQL 18 on `localhost:5433`. It expects a dedicated database named `scamshield_db`.

1. In pgAdmin Query Tool connected to PostgreSQL 18, run:

```sql
CREATE DATABASE scamshield_db;
```

2. Put your PostgreSQL `postgres` password in `backend/.env`:

```env
DB_HOST=localhost
DB_PORT=5433
DB_NAME=scamshield_db
DB_USER=postgres
DB_PASSWORD=your_password
```

3. Activate the existing virtual environment and run:

```bash
uvicorn app.main:app --reload
```

4. Open `/db-health`. A successful response should show `status: connected`, database `scamshield_db`, user `postgres`, and port `5433`.

## Detection endpoints

The detection routes do not require a database connection:

- `POST /api/v1/detect/message` with JSON `{ "message": "..." }`
- `POST /api/v1/detect/url` with JSON `{ "url": "https://..." }`
- `POST /api/v1/detect/pdf` with a multipart `file` upload

PDF uploads are parsed server-side and analyzed without opening them in the user's browser. Text-based PDFs are supported up to 10 MB and 50 pages. Image-only PDFs require OCR before their text can be analyzed.

No application tables are created yet. Models should be added only after the ScamShield data design is finalized.
