FROM tiangolo/uwsgi-nginx-flask:python3.8
COPY ./requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt
COPY ./final.h5 ./overfit.h5 ./encoder_config.json /app/
COPY . /app