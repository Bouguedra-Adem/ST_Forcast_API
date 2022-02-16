FROM python:3.8
#COPY . /app
WORKDIR /app
ADD ./requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
ADD . /app
Expose 5000
CMD ["python" ,"./app.py"]