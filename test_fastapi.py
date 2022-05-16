from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import json
from utilities import *
from bs4 import BeautifulSoup
import re
from models import *
from PIL import Image
from datetime import datetime
import pytz

app = FastAPI()

origins = ["*"]
app.add_middleware(CORSMiddleware, allow_origins=origins,allow_credentials=True,allow_methods=["*"],allow_headers=["*"])


@app.post("/postURL")
async def post_url(item:PayloadUrl):
    r = read_url(item.url)
    k = item.k
    with open("temp_image.jpg", "wb") as f:
        f.write(r.content)
    image_bytes,image_nparr = resize_image(path="temp_image.jpg")
    os.remove("temp_image.jpg")
    auth = get_awsauth()
    es = conn_es(auth)
    INCREASE = 120
    features = get_features(sm_runtime_client, sagemaker_endpoint, image_bytes)
    neighbors = get_neighbors(features, es, k_neighbors=k+INCREASE)
    s3_uris_best_neighbors = similarity_score(features, neighbors)
    s3_presigned_urls = generate_presigned_urls(s3_uris_best_neighbors,k)
    return { "images": s3_presigned_urls}

@app.post("/postImage")
async def post_img(item:PayloadBase64):
    img_string = item.base64img
    k = item.k
    image = readb64(img_string)
    image_bytes,image_nparr = resize_image(image_src=image)
    auth = get_awsauth()
    es = conn_es(auth)
    INCREASE = 120
    features = get_features(sm_runtime_client, sagemaker_endpoint, image_bytes)
    neighbors = get_neighbors(features, es, k_neighbors=k+INCREASE)
    s3_uris_best_neighbors = similarity_score(features, neighbors)
    s3_presigned_urls = generate_presigned_urls(s3_uris_best_neighbors,k)
    return { "images": s3_presigned_urls}

# @app.post("/cropObject")
# async def crop_object(item:PayloadBase64):
#     img_string = item.base64img
#     image = readb64(img_string)
#     image_bytes,image_nparr = resize_image(image_src=image)
#     results = model(image_nparr)
#     imgs = results.crop(save=False)
#     base64_imgs = []
#     for i in imgs:
#         pil_img = Image.fromarray(i['im'])
#         buff = BytesIO()
#         pil_img.save(buff, format="JPEG")
#         base64_img = "data:image/jpg;base64,"+base64.b64encode(buff.getvalue()).decode("utf-8")
        
#         base64_imgs.append(base64_img)
    
#     return { "images": base64_imgs}
    

@app.get("/getStatusES")
async def get_status_es():
    auth = get_awsauth()
    es = conn_es(auth)
    result = get_status(es)
    return result


@app.get("/getImageLastModified")
async def home():
    uris_top20 = getImage()
    s3_presigned_urls = generate_presigned_urls(uris_top20)
    return {"images": s3_presigned_urls} 


@app.post("/productCrawl")
async def product_crawl(item:PayloadUrl):
    url = item.url
    response = read_url(url).text
    soup = BeautifulSoup(response, 'html.parser')
    result = []
    count = 0
    info_products = soup.find_all(["td","li","tr"],attrs={"class": [re.compile('product')]})
    if not(info_products):
        info_products = soup.find_all(["div"],attrs={"class": 'grid-product'})
    for product in info_products:
        name = product.find(['h2','h4',"h6"])
        if not(name):
            name = product.find('a', re.compile("title"))
        url = product.find('a',href=not_null)
        url_img = product.find('img')
        try:
            if url_img:
                dataframe ={}
                dataframe["product_id"] = count
                dataframe["product_name"] = name.text.replace("\xa0", " ").replace("\t","").replace("\n","")
                dataframe["product_url"] = url['href']
                dataframe["product_img"] = url_img['src']
                result.append(dataframe)
                count += 1
        except:
            continue

    return {"result": result}


@app.post("/uploadCrawled")
async def upload_Crawled(item:PayloadCrawled):
    crawled = item.crawled
    for i in range(len(crawled)):
        url = crawled[i]['img']
        r = read_url(url)
        file_name = generate_filename()
        path_filename = f"{file_name}" 
        with open(path_filename, "wb") as f:
            f.write(r.content)
        image, image_nparr = resize_image(path=path_filename)
        features = get_features(sm_runtime_client, sagemaker_endpoint, image)
                #upload s3
        prefix = "data/uploads/"
        key = prefix + file_name
        s3_client.upload_file(path_filename,bucket,key)
        uri_s3 = f"s3://{bucket}/{key}"
        temp = { "s3_uri": uri_s3,
                "feature": features,
                "name_img": crawled[i]['name'],
                "url": crawled[i]['url']}
        result_index = []
        result_index.append(temp)
        es_import(result_index)
        os.remove(path_filename)
        
    time_today = datetime.now(pytz.timezone('Asia/Hong_Kong'))
    time_today = time_today.strftime(time_format)
    with open('lastupdate.txt','w') as f:
        f.write(time_today)
    key = 'backup/lastupdate.txt'
    s3_client.delete_object(Bucket=bucket,Key=key)
    s3_client.upload_file('lastupdate.txt',bucket,key)
    return {
        "images": crawled
        }
    
@app.get("/resetES")
async def resetES():
    key = 'backup/features.json'
    payload = s3_client.get_object(Bucket=bucket,Key=key)['Body'].read()
    features_json = payload.decode('utf8').replace("'", '"')
    rs_features = json.loads(features_json)
    knn_index = {
        "settings": {
            "index.knn": True
        },
        "mappings": {
            "properties": {
                    "zalando_img_vector": {
                    "type": "knn_vector",
                    "dimension": 2048
                }
            }
        }
    }
    awsauth = get_awsauth()
    es = conn_es(awsauth)
    requests.delete(url=f'https://{elasticsearch_endpoint}/_all',auth=awsauth)
    es.indices.create(index="idx_zalando",body=knn_index,ignore=400)
    es_import(rs_features)
    time_today = datetime.now(pytz.timezone('Asia/Hong_Kong'))
    time_today = time_today.strftime(time_format)
    delete_all_uploaded(bucket)
    with open('lastupdate.txt','w') as f:
        f.write(time_today)
    key = 'backup/lastupdate.txt'
    s3_client.delete_object(Bucket=bucket,Key=key)
    s3_client.upload_file('lastupdate.txt',bucket,key)
    return {
        "message" : "Finish resetting the index to the original!!"
    }

@app.post("/postUploadImage")
async def uploadImage(item:PayloadBase64):
    for img_string in item.base64img:
        image = readb64(img_string)
        image, image_nparr = resize_image(image_src=image)
        features = get_features(sm_runtime_client, sagemaker_endpoint, image)
        image_open = Image.open(image)
        file_name = generate_filename()
        path_filename = f"{file_name}"                    
        image_open.save(path_filename)
        prefix = "data/uploads/"
        key = prefix + file_name

        s3_client.upload_file(path_filename,bucket,key)
        uri_s3 = f"s3://{bucket}/{key}"
        temp = {"s3_uri": uri_s3,
                "feature": features}
        result_index = []
        result_index.append(temp)
        es_import(result_index)
        os.remove(path_filename)
        #Get 20 images
    uris_top20 = getImage()
    s3_presigned_urls = generate_presigned_urls(uris_top20)

        #update Status
    time_today = datetime.now(pytz.timezone('Asia/Hong_Kong'))
    time_today = time_today.strftime(time_format)
    with open('lastupdate.txt','w') as f:
        f.write(time_today)
    key = 'backup/lastupdate.txt'
    s3_client.delete_object(Bucket=bucket,Key=key)
    s3_client.upload_file('lastupdate.txt',bucket,key)            
    return {
            "images": s3_presigned_urls
        }

@app.get("/")
def home():
    return "Home"





