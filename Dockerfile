FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install application dependencies.
COPY requirements.txt .

RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy the fitted model artifact.
COPY model.joblib ./model.joblib

# Copy the FastAPI application.
COPY app/ ./app/

EXPOSE 8000

# Start the API server.
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]