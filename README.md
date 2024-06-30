# Multi-Modal Representation Learning for Social Media Popularity Prediction

This project leverages advanced machine learning and data engineering techniques to predict the popularity of Reddit posts based on various features. It showcases the integration of multiple cutting-edge technologies to create a robust, automated pipeline for data processing and model training.

## Key Technologies and Features

### ETL Pipeline
- **Apache Airflow**: Orchestrates the entire data pipeline, from scraping to model training, ensuring daily updates and seamless workflow management.

#### Deep Learning and Embeddings
- **Image Caption Generation**: Automatically generates detailed descriptive captions for images using multi-modal large language models (LLMs).
- **TensorFlow**: Powers the multimodal deep learning model for popularity prediction.
- **Text Embeddings**: Utilizes advanced NLP techniques to create meaningful representations of post titles. The model used: [FlagEmbedding's bge-m3](https://huggingface.co/BAAI/bge-m3).
- **Image Embeddings**: Generates rich visual features from post images. The model used: [Vision Transformer Image Classification](https://huggingface.co/timm/vit_large_patch16_384.augreg_in21k_ft_in1k).
- **Visual Embeddings**: Combines image and text data for a comprehensive multimodal representation. The model used: [FlagEmbedding's VisualBGE](https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/visual).

### Data Processing
- **Reddit API (PRAW)**: Facilitates efficient data scraping from Reddit.
- **FlagEmbedding**: Employed to create sophisticated visual and combined embeddings.

### Features Used for Prediction
1. Title Embeddings
2. Image Embeddings
3. Caption Embeddings (generated from images)
4. Visual Embeddings (combined image and text)
5. Post Metadata
6. Author's Metadata

## Airlfow Pipeline Overview

The Airflow-managed pipeline includes:
1. Data Scraping
2. Data Filtering
3. Image Fetching
4. Image Caption Generation
5. Embedding Creation (Text, Visual, Combined)
6. Feature Merging
7. Model Training and Evaluation
8. Model Persistence

The pipeline runs daily, continuously improving the model with new data. Each trained model is saved along with its evaluation metrics for tracking performance over time.

## Installation and Setup

### Dependencies

#### Apache Airflow
```shell
export AIRFLOW_HOME=~/airflow
AIRFLOW_VERSION=2.9.1
PYTHON_VERSION="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"
```

#### FlagEmbedding
```shell
git clone https://github.com/FlagOpen/FlagEmbedding.git
cd FlagEmbedding
pip install -e .
pip install torchvision timm einops ftfy
```

Note: Download the Visual Embedding model from [BAAI/bge-visualized](https://huggingface.co/BAAI/bge-visualized) and specify the path in `src/t06.4_create_embeddings_combined.py:50`.

#### Other Dependencies
```shell
pip install -r requirements.txt
```

### Project Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/DistilledCode/mmrl.git
   cd mmrl
   ```

2. Configure Reddit credentials in `praw.ini`:
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

3. Start the scraper:
   ```shell
   ./monitor_scrapper.sh
   ```

4. Set up the Airflow environment:
   ```bash
   export PROJ_DIR=$PWD
   cp praw.ini smpp_pipeline.py ~/airflow/dags
   ```

5. Launch Airflow:
   ```shell
   airflow scheduler &
   airflow webserver -p 8080 &
   ```

   Access the Airflow web interface at `http://localhost:8080` to enable and monitor the pipeline.
