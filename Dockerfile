FROM python:3.12
WORKDIR "/DEVOPS PROJECT"
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN echo "Requiremnts Installed"
COPY src/ ./src/
COPY data/ ./data/
CMD ["python", "src/Fraud_Detection.py"]
