from io import BytesIO
from flask import jsonify, request
import cv2
from libs.utils import *
from libs.imageProcessing import *
from libs.drawLine import *
from libs.painting import *
from PIL import Image
from uuid import uuid4
from s3_connect import s3_connection
from aws_config import AWS_S3_BUCKET_NAME




from multiprocessing import Process, Manager



def reduce_color_process(paintingTool, img, cluster, result, colorNames, colors):


    print(f'프로세스 컬러 군집화 시작')
    clusteredImage = paintingTool.colorClustering( img, cluster = cluster )
    
    print(f'프로세스 이미지 확장')
    expandedImage = imageExpand(clusteredImage, guessSize = False, size=3)
    
    print(f'프로세스 컬러 매칭 시작')
    paintingMap = paintingTool.expandImageColorMatch(expandedImage)
    

    print(f'프로세스 컬러 추출 시작')
    colorNames_, colors_ = getColorFromImage(paintingMap)

    print(f'프로세스 컬러 {len(colorNames_)}개')

    colorNames[1] = colorNames_
    colors[1] = colors_

    result.put(paintingMap)
    return


def s3_upload(result_img, s3, job):
    file_name = uuid4().hex
    image = result_img.astype('uint8')
    image = Image.fromarray(image)
    buffer = BytesIO()
    image.save(buffer, 'PNG')
    buffer.seek(0)
    s3.upload_fileobj(buffer, AWS_S3_BUCKET_NAME, f"{job}/{file_name}.png", ExtraArgs={'ACL':'public-read'})
    print('업로드 완료')
    url = f'https://sparta-team4-project.s3.ap-northeast-2.amazonaws.com/{job}/{file_name}.png'
    return url

def convert(job, url):
    colorNames = {}
    colors = {}
    img_lab = None
    lab = None

    s3 = s3_connection()
    
    paintingTool = Painting(url)
    blurImage = 0
    if job =='sketch':

        print(f'블러 시작')

        # 색 단순화 + 블러 처리
        blurImage = paintingTool.blurring(
            div = 8, 
            radius = 10, 
            sigmaColor =20, 
            medianValue=3
        )  
        img = s3_upload(blurImage, s3, job)
        return job, img

    elif job =='pipo':
        clusters = 32
        manager = Manager()
        result_list = manager.Queue()
        colorNames_ = manager.dict()
        colors_ = manager.dict()
        process = Process(target=reduce_color_process, args=(paintingTool, paintingTool.image, clusters, result_list, colorNames_, colors_))
        process.start()
        process.join()

        painting_map_1 = result_list.get()
        colorNames = dict(colorNames_)
        colors = dict(colors_)
        paintingMap = painting_map_1


        # 선 그리기
        print(f'선 그리기 시작')
        drawLineTool = DrawLine(paintingMap)
        lined_image = drawLineTool.getDrawLine()
        lined_image = drawLineTool.drawOutline(lined_image)

        # 레이블 추출
        img_lab, lab = getImgLabelFromImage(colors[1], paintingMap)
        lined_image = cv2.convertScaleAbs(lined_image)

        # contour, hierarchy 추출
        print(f'컨투어 추출 시작')
        contours, hierarchy, thresh = getContoursFromImage(lined_image.copy())


        # 결과 이미지 백지화
        result_img = makeWhiteFromImage(paintingMap)
        result_img = setBackgroundAlpha(paintingMap, result_img)

        # 결과이미지 렌더링
        # image를 넣으면 원본이미지에 그려주고, result_img에 넣으면 백지에 그려줌
        print(f'넘버링 시작')
        result_img = setColorNumberFromContours(result_img, thresh, contours, hierarchy, img_lab, lab, colorNames[1])

        print(f'컬러 레이블링 시작')
        result_img2 = setColorLabel(result_img.copy(), colorNames[1], colors[1])

        print(f'작업 완료')
        img = s3_upload(result_img, s3 ,job)
        label_img = s3_upload(result_img2, s3, job)

        return job, [img, label_img]
