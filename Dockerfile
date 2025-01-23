# Use the official Python image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire data directory into the container (Ensure data is in the build context)
COPY ./data/raw ./data  # Adjusted to reflect the correct path within the build context

# Copy the training script from the src directory
COPY ./src/train.py ./ 

# Copy the application code into the container (app.py is in the same folder as Dockerfile)
COPY ./app.py ./ 

# Set the command to run the training script
CMD ["python", "train.py"]
