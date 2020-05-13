from setuptools import find_packages
from setuptools import setup
import os

setup(
    name='trainer',
    version='0.0.2',
    intall_requires=[
        'absl-py==0.9.0',
        'appdirs==1.4.3',
        'astunparse==1.6.3',
        'attrs==19.3.0',
        'black==19.10b0',
        'cachetools==4.1.0',
        'certifi==2020.4.5.1',
        'chardet==3.0.4',
        'click==7.1.2',
        'cycler==0.10.0',
        'gast==0.3.3',
        'google-auth==1.14.1',
        'google-auth-oauthlib==0.4.1',
        'google-pasta==0.2.0',
        'grpcio==1.28.1',
        'h5py==2.10.0',
        'idna==2.9',
        'joblib==0.14.1',
        'Keras-Preprocessing==1.1.0',
        'kiwisolver==1.2.0',
        'Markdown==3.2.1',
        'matplotlib==3.2.1',
        'MIDIUtil==1.2.1',
        'numpy==1.18.4',
        'oauthlib==3.1.0',
        'opt-einsum==3.2.1',
        'pathspec==0.8.0',
        'protobuf==3.11.3',
        'pyasn1==0.4.8',
        'pyasn1-modules==0.2.8',
        'pydot==1.4.1',
        'pyparsing==2.4.7',
        'python-dateutil==2.8.1',
        'PyYAML==5.3.1',
        'regex==2020.4.4',
        'requests==2.23.0',
        'requests-oauthlib==1.3.0',
        'rsa==4.0',
        'scikit-learn==0.22.2.post1',
        'scipy==1.4.1',
        'six==1.14.0',
        'sklearn==0.0',
        'tensorboard==2.2.1',
        'tensorboard-plugin-wit==1.6.0.post3',
        'tensorflow==2.2.0rc4',
        'tensorflow-estimator==2.2.0',
        'termcolor==1.1.0',
        'toml==0.10.0',
        'tqdm==4.46.0',
        'typed-ast==1.4.1',
        'urllib3==1.25.9',
        'Werkzeug==1.0.1',
        'wrapt==1.12.1',
    ],
    packages=find_packages(),
    include_package_data=True,
    description='BachNet trainer package.'
)