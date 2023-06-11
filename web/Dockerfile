FROM python:3.10

WORKDIR /usr/src/SiameseSearchWebapp

COPY . .

RUN pip install --no-cache-dir -r requirements.txt


# Expose port 8000 for the web app
EXPOSE 8000
# Run the command to start the application
CMD ["python", "./manage.py", "runserver", "0.0.0.0:8000"]
