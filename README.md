# CityBikePredictor

A machine learning model to predict the shifting availability in the HSL region citybikes. (Started 27.7.2026)


## How the model works and documentation

This model is a classification model that aims to predict wether a certain bike station will have a high outflow of bikes, a high inflow of bikes or a stable in/outflow of bikes. This means that the model gives a prediction in one of 3 classes.

Here are some of the most important documentation files:
- [Neural network architecture documentation](documentation/nn_arch.md)
- [Docker services documentation](documentation/docker_services.md)
- [Data gathering documentation](documentation/data_gathering.md)

## Running the model

After cloning the repository, you need to install Docker and PostgreSQL (if not already installed) and set up the PSQL connection with a .env file with the following configuration: \
USER=your_username \
PASSWORD=your_password \
API_KEY=your_api_key

The API key can be obtained from [digitransit's website](https://portal-api.digitransit.fi/) (sing in required)

The env file needs to be in the same directory as the docker-compose.yaml file

After that, to collect data, simply run `docker compose up --build scraper`. 

To train the model, simply run `docker compose up --build trainer`.

To run inference on the model, simply run `docker compose up --build inference`.


## Tools and techologies used

| ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | ![Pytorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white) | ![Scikit](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) | ![Docker](https://img.shields.io/badge/docker-257bd6?style=for-the-badge&logo=docker&logoColor=white) | ![Postgresql](https://img.shields.io/badge/postgresql-4169e1?style=for-the-badge&logo=postgresql&logoColor=white) | ![Numpy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white) | ![Matplotlib](https://img.shields.io/badge/-Matplotlib-000000?style=flat&logo=python)