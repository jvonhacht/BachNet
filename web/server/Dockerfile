FROM tiangolo/uwsgi-nginx-flask:python3.8
COPY ./requirements.txt /var/www/requirements.txt
COPY ./final.h5 ./overfit.h5 ./encoder_config.json /var/www/
RUN pip install --upgrade pip && pip install -r /var/www/requirements.txt