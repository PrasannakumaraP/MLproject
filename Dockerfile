# Use a base image with Python
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the training script into the container
COPY src/train.py ./
# Copy the data directory into the container
COPY data ./data  # This will copy the entire data directory

# Run the training script to generate the model
RUN python train.py

# Copy the application code (if any) into the container
COPY src/app.py ./

# Set the command to run the application (modify as needed)
CMD ["python", "app.py"]
