FROM "ubuntu"
RUN apt-get update && yes | apt-get upgrade


RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip