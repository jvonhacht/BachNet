FROM "ubuntu"


RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip

WORKDIR /
RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3"]
CMD ["app.py"]