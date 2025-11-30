from datetime import datetime,timedelta
from airflow.decorators import dag,task
import pandas as pd 
import sys

# include
sys.path.append("/usr/local/airflow/include")
from include.utils.data_generator import RealisticSalesDataGenerator
default_args = {
    'owner': 'anubhavtewari05',
    'depends_on_past': False,
    'start_date': datetime(2025,11,2),
    'retries' : 1,
    'retry_delay':  timedelta(minutes=1),
    'catchup': False,
    'schedule': '@weekly'
}

@dag(
    default_args=default_args,
    description='Sales Forecast Training DAG',
    tags=['ml', 'training', 'sales_forecast', 'sales'],
)

def sales_forecast_training():
    @task()
    def extract_data_task():

        data_output_dir = '/tmp/sales_data'

        generator = RealisticSalesDataGenerator(
            start_date = "2021-01-01",
            end_date = "2021-03-31"
        )

        print('Generating Realistic Sales Data ...')

        file_paths = generator.generate_sales_data(output_dir=data_output_dir)
        total_files = sum(len(paths) for paths in file_paths.values())
        print(f"Generated {total_files} files")

        for data_type, paths in file_paths.items():
            printf(f"{data_type}: {len(paths)} files")
        
        return {
            'data_output_dir': data_output_dir,
            'file_paths': file_paths,
            'total_files' : total_files
        }



    extract_result = extract_data_task()
    
sales_forecast_training()

