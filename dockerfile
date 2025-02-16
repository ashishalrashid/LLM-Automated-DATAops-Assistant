# Use an official Python runtime as a base image
FROM python:3.11-slim

# Set environment variables to prevent Python from writing .pyc files and buffering output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Ensure requirements.txt is copied before installing dependencies
COPY requirements.txt .

# Print the contents of requirements.txt for debugging
RUN cat requirements.txt

# Install dependencies explicitly with verbose output
RUN pip install --no-cache-dir -r requirements.txt && pip list

# Copy application files
COPY app.py .

# Expose the application port
EXPOSE 8000

# Run the application
CMD ["uv", "run", "app.py"]
