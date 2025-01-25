# Use the official Python image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the raw data folder into the container
COPY data/raw /app/data/raw

# Copy the training script into the container
COPY src/train.py /app/train.py

# Copy the application script into the container
COPY app.py /app/app.py

# Expose the app's port
EXPOSE 5000

# Set the command to run the training script
CMD ["python", "/app/train.py"]
