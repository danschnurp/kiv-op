FROM python:3.10

WORKDIR /usr/src/SiameseSearchWebapp

COPY . .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

RUN chmod 777 ./search/indexed_data/Body.index

RUN chmod 777  -R  ./search/static


# Expose port 8000 for the web app
EXPOSE 8000
# Run the command to start the application
CMD ["python", "./manage.py", "runserver", "0.0.0.0:8000"]
