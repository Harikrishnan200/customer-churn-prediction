# Small, official Python base image keeps the final image size down.
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first, separately from app code. Docker caches
# each layer - as long as requirements.txt doesn't change, this layer
# is reused on rebuilds instead of reinstalling everything.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the application code.
COPY app/ ./app/
COPY src/ ./src/
COPY configs/ ./configs/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
