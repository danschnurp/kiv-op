FROM python:3.10

WORKDIR /usr/src/SiameseSearchWebapp

COPY requirements.txt .

COPY . .

RUN pip install --no-cache-dir -r requirements.txt


# Expose port 8000 for the web app
EXPOSE 8000

CMD ["python", "manage.py", "runserver"]
