import yaml
from complaince.mobito2_0.logging import log
from configs.path import INPUT_DIR, CONFIG_DIR
import boto3

with open(CONFIG_DIR / 'config.yaml') as f:
        config = yaml.safe_load(f)
        config = config['AWS']

s3 = boto3.client('s3', aws_access_key_id = config['access_id'],
                        aws_secret_access_key = config['access_key'])

@log()
def S3_downloader(s3_directory, s3_filename):
    s3.download_file(config['bucket_name'], f'{s3_directory}{s3_filename}', str(INPUT_DIR / s3_filename))
    return str(INPUT_DIR / s3_filename)

def S3_Upload(input_path : str, destination : str):
    """
    input_path = local path
    destination = replace the inputs with outputs from the config json eg. invoice-parser-test/inputs/ -> invoice-parser-test/outputs/<filename>
    """
    try:
        print(destination, input_path, config['bucket_name'])
        s3.upload_file(input_path, config['bucket_name'], destination)
        return True#f'https://{config["bucket_name"]}.s3.{config["location"]}/{destination}'
    except Exception as e:
         return f'{e}'
    