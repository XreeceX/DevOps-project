# Use a lightweight version of Python as the base image
FROM python:3.12

# Set the working directory in the container
WORKDIR "/DEVOPS PROJECT"

# Copy the dependencies file and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code and data into the container
COPY src/ ./src/
COPY data/ ./data/

# Set the command to run your application
CMD ["python", "src/Fraud_Detection.py"]
