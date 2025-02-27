Tarea 5 Carlos Rivera
=========================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   train
   
   inference


DockerFiles
===============================

Dockerfile para entrenamiento:
--------------------------------

.. code-block:: dockerfile

   FROM python:3.11.2
   WORKDIR /usr/src/app
   RUN python -m pip install \
      pandas scikit-learn joblib

   COPY train.py .
   CMD ["python", "train.py"]


Dockerfile para inferencia:
----------------------------

.. code-block:: dockerfile

   FROM python:3.11.2
   WORKDIR /usr/src/app
   RUN python -m pip install \
      pandas scikit-learn joblib

   COPY inference.py .
   ENTRYPOINT ["python", "inference.py"]
   CMD []



Las imagenes de Docker
===============================

Para construir la imagen de entrenamiento, usa:

.. code-block:: bash

    docker build -t train -f dockerfiletrain .

Para construir la imagen de inferencia, usa:

.. code-block:: bash

    docker build -t inference -f dockerfileinference .


Ejecución de los Contenedores
=============================

Ejecutar el contenedor de entrenamiento:

.. code-block:: bash

    sh dockerrun.sh

Ejecutar el contenedor de inferencia:

.. code-block:: bash

    sh dockerinference.sh