# Multi-Media Representational Learning for Social Media Popularity Prediction

This porject use the following features of a image posted on Reddit to predict it's popularity
* Title Embeddings
* Image Embeddings
* Text embeddings of caption generated from image
* Visual Embeddings (combined embedding of both image and text)
* Post Metadata
* Author's Metadata

The whole pipeline is managed using Apache Airflow. This includes
*  Scrapping data
*  Filtering data
*  Fetching images
*  Generating image captions
*  Creating text, visual & combined embeddings
*  Feature merging
*  Training & saving the model

The pipeline is scheduled to run daily and train the model using new scrapped data. Each model trained is also saved with all it's evaluation metric.

## Installation

### Dependencies

**Airlfow**   

```shell

export AIRFLOW_HOME=~/airflow

AIRFLOW_VERSION=2.9.1

# Extract the version of Python you have installed. If you're currently using a Python version that is not supported by Airflow, you may want to set this manually.
# See above for supported versions.
PYTHON_VERSION="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
# For example this would install 2.9.1 with python 3.8: https://raw.githubusercontent.com/apache/airflow/constraints-2.9.1/constraints-3.8.txt

pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"

```

**FlagEmbedding**

```shell
git clone https://github.com/FlagOpen/FlagEmbedding.git
cd FlagEmbedding
pip install -e .
pip install torchvision timm einops ftfy
```

You are also required to download the model for the Visual Embedding in order to generate combined embeddings. Download the model of your choice from https://huggingface.co/BAAI/bge-visualized and specify the path to weights in `src\t06.4_create_embeddings_combined.py:50`


**Other dependencies**

```shell
pip install -r requirements.txt
```

### Setting up the project

1. Clone the project

```bash
git clone https://github.com/DistilledCode/mmrl.git
cd mmrl
```

2. Create `praw.ini` & save Reddit credentials

```ini
[bot1]
client_id=secret
client_secret=secret
username=secret
password=secret

[bot2]
client_id=secret
client_secret=secret
password=secret
username=secret
```


Using 2 accounts is recommended, one for scraping and one for fetching the images. Using one account for both might give 429 response.


3. Start the scrapper. You may start the scrapper at any directory of your choice but `monitor_scrapper.sh`, `scrap_comments.py` & `praw.ini` should be in a single directory.

```shell
./monitor_scrapper.sh
```

4. Export the `PROJ_DIR` variable

```bash
cd mmrl
export PROJ_DIR=$PWD
```

5. Copy `praw.ini` and `smpp_pipeline.py` to `~/airflow/dags`

```shell
cp praw.ini smpp_pipeline.py ~/airflow/dags
cp smpp_pipeline.py ~/airflow/dags
```

6. Start the Airflow scheduler from the `PROJ_DIR` directory 

```shell
cd mmrl
airflow scheduler &
```
7. Start the Airflow Webserver & enable the pipeline
```shell
airflow webserver -p 8080 &
```
This will start the server at `http://localhost:8080`


