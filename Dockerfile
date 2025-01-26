# Use an official Python runtime as a parent image
FROM python:3.8

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY . .

# Expose port 5000 for Flask
EXPOSE 5000

# Command to run the Flask app
CMD ["python", "hyperParameterTunedApp.py"]
