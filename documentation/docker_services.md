# Docker services 🐳

Here is the explanation and use case for every service in [docker-compose.yaml](../docker-compose.yaml)

### Scraper

This service is used to scrape data for the model. It uses [scraper.py](../data/scraper.py) as its main script and fetches data from the digitransit's and FMI's APIs and stores the data inside a PostgreSQL database (not in the repo). The dockerfile for scraper is [Dockerfile.scraper](../Dockerfile.scraper). More about data gathering can be read from [Data gathering documentation](data_gathering.md)

Run scraper with
`docker compose up --build scraper`

### Trainer

This service is used to train the model. It uses [neuralnet.py](../neuralnet.py) as its main script and pulls data from the PostgreSQL database to train the model. When network.save_weights() is called, the weights, scaler and the encoder are saved into [~/parameters/](../parameters/). The dockerfile for trainer is [Dockerfile.trainer](../Dockerfile.trainer). More about training and how the model works can be read from [Neural network architecture documentation](nn_arch.md)

Run trainer with
`docker compose up --build trainer`

### Inference

This service is used to inference the model with live data. It can be used to measure the real results against the predictions or to just see the predicted in/outflow of each station. This service uses [infernce.py](../inference.py) as its main script and loads the model from parameters/ fetches data from the digitransit's and FMI's APIs to make predictions for each bike station. The dockerfile for inference is [Dockerfile.inference](../Dockerfile.inference)

Run inference with
`Docker compose up --build inference`
