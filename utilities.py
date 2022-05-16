import base64
import json

import boto3
import requests
from urllib.parse import urlparse
from io import BytesIO
from elasticsearch import Elasticsearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import uuid
import os
import cv2
import numpy as np
import re
import config
from PIL import Image
from numpy import dot
from numpy.linalg import norm
#import torch

# Global variables that are reused
sm_runtime_client = boto3.client('sagemaker-runtime')
s3_client = boto3.client('s3')
time_format = "%Y-%m-%d %H:%M:%S"
bucket = config.bucket
elasticsearch_endpoint = config.es_host
sagemaker_endpoint = config.sm_endpoint
#model = torch.hub.load('ultralytics/yolov5', 'yolov5x6')


def generate_filename():
    return os.path.join(str(uuid.uuid4())+".jpg")

def get_features(sm_runtime_client, sagemaker_endpoint, img_bytes):
    response = sm_runtime_client.invoke_endpoint(
        EndpointName=sagemaker_endpoint,
        ContentType='application/x-image',
        Body=img_bytes)
    response_body = json.loads((response['Body'].read()))
    features = response_body['predictions'][0]

    return features


def get_neighbors(features, es, k_neighbors=3):
    idx_name = 'idx_zalando'
    res = es.search(
        request_timeout=30, index=idx_name,
        body={
            'size': k_neighbors,
            'query': {'knn': {'zalando_img_vector': {'vector': features, 'k': k_neighbors}}}}
        )
    rs_output = [res['hits']['hits'][x]['_source'] for x in range(k_neighbors)]

    return rs_output


def generate_presigned_urls(s3_uris,k=0):
    if k:
        presigned_urls = [s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': urlparse(x[0]).netloc,
                'Key': urlparse(x[0]).path.lstrip('/')},
            ExpiresIn=300
        ) for x in s3_uris[:k]]
    else:
        presigned_urls = [s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': urlparse(x).netloc,
                'Key': urlparse(x).path.lstrip('/')},
            ExpiresIn=300
        ) for x in s3_uris]

    return presigned_urls



def getImage():
    uris_top20 = []
    kwargs = {'Bucket': bucket,'Prefix': 'data/uploads'}
    s3_response = s3_client.list_objects_v2(**kwargs)
    if s3_response['KeyCount'] == 0:
        kwargs = {'Bucket': bucket,'Prefix': 'data'}
        s3_response = s3_client.list_objects_v2(**kwargs)
    sorted_contents = sorted(s3_response['Contents'], key=lambda d: d['LastModified'], reverse=True)
    count = 0
    for i in sorted_contents:
        uris_top20.append('s3://' + bucket + '/' + i.get('Key'))
        count += 1
        if count == 20:
            break
    count_uris = len(uris_top20)
    if count_uris < 20:
        kwargs = {'Bucket': bucket,'Prefix': 'data/feidegger'}
        s3_response = s3_client.list_objects_v2(**kwargs)
        sorted_contents = sorted(s3_response['Contents'], key=lambda d: d['LastModified'], reverse=True)
        count = 0
        for i in sorted_contents:
            uris_top20.append('s3://' + bucket + '/' + i.get('Key'))
            count += 1
            if (count + count_uris)  == 20:
                break
    return uris_top20

def readb64(uri):
   nparr = np.fromstring(base64.b64decode(uri), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   return img

def delete_all_uploaded(bucket):
    kwargs = {'Bucket': bucket,'Prefix': 'data/uploads'}
    while True:
        resp = s3_client.list_objects_v2(**kwargs)
        if resp['KeyCount'] == 0:
            break
        for obj in resp['Contents']:
            s3_client.delete_object(Bucket=bucket,Key=obj['Key'])
        try:
            kwargs['ContinuationToken'] = resp['NextContinuationToken']
        except KeyError:
            break

def resize_image(path=None,image_src=None):
    if path:
        fx=1.0
        fy=1.0
        input_img = cv2.imread(path)
        #height, width = input_img.shape[:2]
        img_resized = cv2.resize(src=path, dsize=None, fx=fx, fy=fy)
        pil_img = Image.fromarray(img_resized)
        b = BytesIO()
        pil_img.save(b, 'jpeg')
        image = b.getvalue()
    else:
        fx=1.0
        fy=1.0
        height, width = image_src.shape[:2]
        img_resized = cv2.resize(src=image_src, dsize=(width,height), fx=fx, fy=fy)        
        pil_img = Image.fromarray(img_resized)
        b = BytesIO()
        pil_img.save(b, 'jpeg')
        image = b.getvalue()
    return image, img_resized

def read_url(url):
    pageSource = requests.get(url,verify=False)
    return pageSource

def not_null(href):
    return href and not re.compile("#").search(href)


def similarity_score(img_vector1, imgs_vector2):
    result = []
    for i in imgs_vector2:
        img_vector2 = i['zalando_img_vector']
        cos_sim = dot(img_vector1, img_vector2)/(norm(img_vector1)*norm(img_vector2))
        result.append((i['image'],cos_sim))
    
    result.sort(key=lambda i:i[1],reverse=True)

    return result


def get_awsauth():
    service = 'es'
    region = config.region
    session = boto3.session.Session()
    credentials = session.get_credentials()
    awsauth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        region,
        service,
        session_token=credentials.token
        )
    
    return awsauth


def conn_es(awsauth):
    es = Elasticsearch(
        hosts=[{'host': elasticsearch_endpoint, 'port': 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection
    )

    return es


def es_import(es,result):
        for i in result:
            if "url" not in i:
                es.index(index='idx_zalando',
                        body={"zalando_img_vector": i['feature'], 
                            "image": i['s3_uri'],
                            "name_img":i['s3_uri'].split("/")[-1],
                            "url": ""}
                        )
            else:
                es.index(index='idx_zalando',
                        body={"zalando_img_vector": i['feature'], 
                            "image": i['s3_uri'],
                            "name_img":i['name_img'],
                            "url": i['url']}
                        )


def get_status(es):
        resp = es.cluster.stats()
        if resp['status'] == 'yellow' or resp['status'] == 'green':
            status = 'Ready'
        else:
            status = 'Error'
        key = 'backup/lastupdate.txt'
        payload = s3_client.get_object(Bucket=bucket,Key=key)['Body'].read()
        last_update = str(payload).replace("b'","").replace("'","")

        return {
            "status": status,
            "count_index" : resp['indices']['docs']['count'],
            "last_update" : last_update
        }