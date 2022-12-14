import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt

images_path = '/Users/hahmwj/Desktop/augmentation_dataset/Data/super_dataset/images/'
label_path = '/Users/hahmwj/Desktop/augmentation_dataset/Data/super_dataset/labels/'

img_file_list = os.listdir(images_path)
label_file_list = os.listdir(label_path)


# 좌우반전 이미지 생성
count = 0
for img_name in img_file_list:

    # DS_Store Error 해결
    if '.DS_Store' in img_name:
        print(img_name)
        continue
    # 새로운 파일 이름 지정
    split_name = img_name.split('.')
    new_name = split_name[0] + 'LR.' + split_name[1]

    LR_img_file_list = os.listdir(images_path)

    # 이미 이동된 이미지면 추가하지 않는다.
    if new_name in LR_img_file_list:
        continue


    # 이미지 오픈
    image = Image.open(images_path + img_name)

    # 이미지 좌우반전
    image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # OSError: cannot write mode RGBA as JPEG 에러 처리
    image = image.convert("RGB")

    # 이미지 저장
    image.save(images_path + new_name)
    count+=1
    print(count)


count_label = 0
for label in label_file_list:
    if '.DS_Store' in label:
        print(label)
        continue

    #label 받아오기
    f = open(label_path + label, 'r')
    lines = f.readlines()
    f.close()

    # 저장할 새로운 이름 지정
    split_name = label.split('.')
    new_name = split_name[0] + 'LR.' + split_name[1]

    LR_label_file_list = os.listdir(label_path)

    # 이미 이동된 이미지면 추가하지 않는다.
    if new_name not in LR_label_file_list:
        # 작성할 파일 생성
        new_f = open(label_path + new_name, 'w')
        new_f.close()
    else:
        continue

    #파일이 다 작성 되면 줄바꿈을 안하게 해주는 변수
    count = 0

    #label 줄 단위로 읽기
    for line in lines:
        line_list = line.split()

        #어떤 타입의 label인지 표시
        label_type = line_list[0]

        # 기존 라벨 값 불러오기
        x = float(line_list[1])
        y = line_list[2]
        w = line_list[3]
        h = line_list[4]

        # 라벨 좌우대칭으로 변경
        new_x = str(round((1 - x), 6))


        # 파일에 내용 작성
        new_f = open(label_path + new_name, 'a')
        new_f.write(label_type+' '+new_x+' '+y+' '+w+' '+h)
        count += 1
        if count < len(lines):
            new_f.write('\n')
            new_f.close()

    print(count_label)


