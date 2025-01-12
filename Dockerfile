# Use a base image with Python
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the training script into the container
COPY src/train.py ./

# Run the training script to generate the model
RUN python train.py

# Copy the model file into the container (assuming it's generated in the previous step)
COPY models/model.pkl ./models/model.pkl

# Copy the application code into the container
COPY app.py ./

# Expose the application port (e.g., 5000 for Flask)
EXPOSE 5000

# Set the command to run the application
CMD ["python", "app.py"]
