import uuid
import tempfile
from pathlib import Path
import os
import cv2
import layoutparser as lp
from PIL import Image
from configs.path import WEIGHT_DIR
from fastapi import FastAPI, Request, status, HTTPException
from utils.aws_utils import S3_downloader, S3_Upload
import pydantic
import logging
from utils.custom_logging import CustomizeLogger
from fastapi.middleware.cors import CORSMiddleware 

import os
import tempfile
import cv2
import boto3
from concurrent.futures import ProcessPoolExecutor
from ultralytics import YOLO

model = YOLO('')

def inverse_bboxes(row, h, w):
  row = row.replace('\n', '')
  _, x_center, y_center, width, height = map(float, row.split(' '))
  x_center, y_center, width, height = (x_center * w,
                                      y_center * h,
                                      width * w,
                                      height * h)
  x_min = int((x_center - (width / 2)))
  y_min = int((y_center - (height / 2)))
  x_max = int((x_center + (width / 2)))
  y_max = int((y_center + (height / 2)))
  return (x_min, y_min, x_max, y_max)

def identify_articles(image_path, model):
  coords_full = []
  img = cv2.imread(image_path)
  h, w = img.shape[:2]
  with tempfile.TemporaryDirectory() as temp_dir:
    results = model.predict(image_path, save = True, save_txt = True, project = temp_dir, name = 'results')
    filename = os.listdir(f'{temp_dir}/results/labels/')[0]
    with open(f'{temp_dir}/results/labels/{filename}') as f:
      rows = f.readlines()
    for row in rows:
      coords = inverse_bboxes(row, h, w)
      coords_full.append(coords)
    coords_full = sorted(coords_full, key = lambda x : (x[0], x[1]))
  return coords_full

app = FastAPI()

logger = logging.getLogger(__name__)
logger = CustomizeLogger.make_logger("logging_config.json")
app.logger = logger

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)
								 
class MessageBody(pydantic.BaseModel):
    s3_input_stem : str
    input_file_name : str

async def ocr(body: MessageBody, request: Request):
    request_id = str(uuid.uuid4())
    try:
        image_path = body.input_file_name + body.s3_input_stem
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdirname = Path(tmpdirname)
            # You need to define or import the S3_downloader function to download the image from S3.
            S3_downloader(body.s3_input_stem, body.input_file_name, tmpdirname)
            if os.listdir(tmpdirname):
                request.app.logger.info(f"{request_id} : Downloaded file from S3 bucket successful")
            image_path = body.input_file_name + body.s3_input_stem
            image = cv2.imread(str(tmpdirname / body.input_file_name))[..., ::-1]

            # Add OCR processing logic here to extract coordinates from the image.
            coordinates = identify_articles(image_path, model)

            return {'data': coordinates}
    except Exception as e:
        request.app.logger.exception(f"{request_id} : Error in API")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))