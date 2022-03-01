from uuid import uuid4
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image
import requests
from s3_connect import s3_connection
from aws_config import AWS_S3_BUCKET_NAME
import tensorflow_hub as hub


hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

def load_style(style_path, max_dim):

    img = tf.io.read_file(style_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    # 전체 이미지의 비율을 유지하면서, 원하는 크기로 변환
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def upload_tensor_img(tensor, s3):
    tensor = np.array(tensor*255 ,dtype=np.uint8)
    image = Image.fromarray(tensor[0])
    buffer = BytesIO()
    image.save(buffer, 'PNG')
    buffer.seek(0)
    file_name = str(uuid4().hex) + '.png'
    s3.put_object(
        Bucket = AWS_S3_BUCKET_NAME,
        Body = buffer,
        Key = f"nst/{file_name}",
        ACL = 'public-read'
    )
    print('업로드 완료')
    url = f'https://sparta-team4-project.s3.ap-northeast-2.amazonaws.com/nst/{file_name}'
    return url

def nst_apply(url) :
    s3 = s3_connection()
    directory = s3.list_objects_v2(
        Bucket=AWS_S3_BUCKET_NAME,
        Prefix=('style/')
    )
    styles = directory['Contents'][1:]
    styles = [i['Key'].split('/')[-1].split('.')[0] for i in styles]
    style_list = [ f'https://sparta-team4-project.s3.ap-northeast-2.amazonaws.com/style/{i}.jpg' for i in styles]
    image_urls = [url]
    res = requests.get(url)

    img = Image.open(BytesIO(res.content)).convert('RGB')
    content_image = tf.keras.preprocessing.image.img_to_array(img)    

    content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.
    content_image = tf.image.resize(content_image, (512, 512))
    for i in range(len(style_list)):
        style_path = tf.keras.utils.get_file(f'style_{i}.jpg', style_list[i])
        style_image = load_style(style_path, 512)

        stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
        image_url = upload_tensor_img(stylized_image, s3)
        image_urls.append(image_url)
        
    return image_urls
