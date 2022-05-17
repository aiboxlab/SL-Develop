# FacialActionLibras

## Code and file structure

~~~
├ FacialActionLibras

├── LICENSE
├── Makefile           <- Makefile with commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
|   ├── examples       <- Data used to test code examples.
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default Sphinx project; see sphinx-doc.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
|   ├── examples       <- Data generated from test codes.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.py           <- Make this project pip installable with `pip install -e`
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data           <- Scripts to download or generate data
│   │   └── make_dataset.py
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │   │                 predictions
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
│       └── visualize.py
│
└── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
~~~


Forma de execução via docker antiga:

~~~bash
docker run --rm -it -v /home/jms2/Documentos/projetos/libras/FacialActionLibras/app:/home/work/app facialactionlibras:0.0.5 /bin/bash
~~~
--ipc=host (nesni de --hostipc do x11docker) de permitir que o contêiner do docker se comunique com os processos do host e também acesse as memórias compartilhadas.
--

## Procedimentos de execução de testes via Dlib através do Env Docker

Gere uma imagem.


Abra um terminal e deixe o serviço abaixo executando:

~~~bash

x11docker --hostipc --hostdisplay --webcam --share /home/jms2/Documentos/projetos/libras/FacialActionLibras facialactionlibras:0.0.8 bash

~~~

Em outro bash, liste o último nome do container gerado por último e copie: 

~~~bash
docker ps -a
~~~

Altere o comando abaixo, substituindo o nome do bash:

~~~bash
docker exec -it x11docker_X0_facialactionlibras-0-0-8-bash_27633084693 bash
~~~

Navegue até o diretório e execute os scripts:

~~~bash
cd ../home/jms2/Documentos/projetos/libras/FacialActionLibras/src/features/
~~~

~~~bash
python video_landmarks_detection.py
~~~

