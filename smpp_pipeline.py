from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import os

PROJ_DIR = os.environ.get("PROJ_DIR")
SCRAPPED_DATA_DIR = os.environ.get("SCRAPPED_DATA_DIR")

os.chdir(PROJ_DIR)

DEFAULT_ARG = {
    "owner": "airflow",
    "start_date": datetime(2024, 6, 10),
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    "smpp_pipeline",
    default_args=DEFAULT_ARG,
    tags=["smpp"],
    schedule="@daily",
    catchup=False,
) as dag:
    task00_sync = BashOperator(
        task_id="task00_sync",
        bash_command=f"rsync -av --ignore-existing {SCRAPPED_DATA_DIR}/data/ {PROJ_DIR}/data/",
    )
    task01_extract = BashOperator(
        task_id="task01_extract",
        bash_command=f"python3 {PROJ_DIR}/src/t01_extract.py",
        depends_on_past=True,
    )

    task02_fetch_info = BashOperator(
        task_id="task02_fetch_info",
        bash_command=f"python3 {PROJ_DIR}/src/t02_fetch_info.py",
        depends_on_past=True,
    )
    task03_filter_posts = BashOperator(
        task_id="task03_filter_posts",
        bash_command=f"python3 {PROJ_DIR}/src/t03_filter_posts.py",
        depends_on_past=True,
    )
    task04_fetch_images = BashOperator(
        task_id="task04_fetch_images",
        bash_command=f"python3 {PROJ_DIR}/src/t04_fetch_images.py",
        depends_on_past=True,
    )
    task05_caption_images = BashOperator(
        task_id="task05_caption_images",
        bash_command=f"python3 {PROJ_DIR}/src/t05_image_captioning.py",
        depends_on_past=True,
    )
    task06_1_embedding_caption = BashOperator(
        task_id="task06_1_embedding_caption",
        bash_command=f"python3 {PROJ_DIR}/src/t06.1_create_embeddings_caption.py",
        depends_on_past=True,
    )
    task06_2_embedding_title = BashOperator(
        task_id="task06_2_embedding_title",
        bash_command=f"python3 {PROJ_DIR}/src/t06.2_create_embeddings_titles.py",
        depends_on_past=True,
    )
    task06_3_embedding_images = BashOperator(
        task_id="task06_3_embedding_images",
        bash_command=f"python3 {PROJ_DIR}/src/t06.3_create_embeddings_images.py",
        depends_on_past=True,
    )
    task06_4_embedding_combined = BashOperator(
        task_id="task06_4_embedding_combined",
        bash_command=f"python3 {PROJ_DIR}/src/t06.4_create_embeddings_combined.py",
        depends_on_past=True,
    )
    task07_merge_features = BashOperator(
        task_id="task07_merge_features",
        bash_command=f"python3 {PROJ_DIR}/src/t07_merge_features.py",
        depends_on_past=True,
    )
    task08_train = BashOperator(
        task_id="task08_train",
        bash_command=f"python3 {PROJ_DIR}/src/t08_train.py",
        depends_on_past=True,
    )
    task00_sync.set_downstream(task01_extract)
    task01_extract.set_downstream(task02_fetch_info)
    task02_fetch_info.set_downstream(task03_filter_posts)
    task03_filter_posts.set_downstream(task04_fetch_images)
    task04_fetch_images.set_downstream(task05_caption_images)
    task05_caption_images.set_downstream(task06_1_embedding_caption)
    task06_1_embedding_caption.set_downstream(task06_2_embedding_title)
    task06_2_embedding_title.set_downstream(task06_3_embedding_images)
    task06_3_embedding_images.set_downstream(task06_4_embedding_combined)
    task06_4_embedding_combined.set_downstream(task07_merge_features)
    task07_merge_features.set_downstream(task08_train)
