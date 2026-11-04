# Reproducible environment for the notebook and the Streamlit app
FROM python:3.13.5-slim

WORKDIR /project

# Exact package versions from the lock file
COPY requirements-lock.txt .
RUN pip install --no-cache-dir -r requirements-lock.txt

COPY . .

EXPOSE 8501
# Default: start the app. To re-run the notebook instead, see the README.
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
