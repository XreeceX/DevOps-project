# Use Python official image
FROM python:3.12

# Set the working directory
WORKDIR /DEVOPS PROJECT

# Copy project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (for API)
EXPOSE 5000

# Run the application
CMD ["python", "Fraud Detection.py"]
