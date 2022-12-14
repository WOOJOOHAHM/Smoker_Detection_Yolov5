import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt

images_path = '/Users/hahmwj/Desktop/augmentation_dataset/Data/super_dataset/images/'
label_path = '/Users/hahmwj/Desktop/augmentation_dataset/Data/super_dataset/labels/'

img_file_list = os.listdir(images_path)
label_file_list = os.listdir(label_path)

img_file_list.sort()
label_file_list.sort()

cut_rate = 0.2

count_img = 0
for img in img_file_list:

    #DS-Store는 넘어간다
    if '.DS_Store' in img:
        continue

    #새롭게 지정할 이름 생성
    split_name = img.split('.')
    new_name = split_name[0] + 'CUT.' + split_name[1]

    #만약 이동된 객체가 있으면 넘어간다
    new_img_file_list = os.listdir(images_path)
    if new_name in new_img_file_list:
        continue

    image = cv2.imread(images_path + img)

    image_height = image.shape[0]
    image_width = image.shape[1]

    cropped = image[0 : image_height, int(cut_rate * image_width):image_width]

    cv2.imwrite(images_path + new_name, cropped)
    count_img += 1

print(count_img)

count_label = 0
for label in label_file_list:
    #DS_store문제 해결
    if '.DS_Store' in label:
        continue

    f = open(label_path + label, 'r')
    lines = f.readlines()

    #새롭게 저장할 split이름 저장
    split_name = label.split('.')
    new_name = split_name[0] + 'CUT.' + split_name[1]

    cut_label_file_list = os.listdir(label_path)

    # 이미 이동된 이미지면 추가하지 않는다.
    if new_name not in cut_label_file_list:
        # 작성할 파일 생성
        new_f = open(label_path + new_name, 'w')
        new_f.close()
    else:
        continue

    for line in lines:
        line_list = line.split()

        x = float(line_list[1]) * image_width
        y = float(line_list[2]) * image_height
        w = float(line_list[3]) * image_width
        h = float(line_list[4]) * image_height

        new_x = x - image_width * cut_rate

        if new_x <= 0:
            new_x = w / 2
            w = w / 2


        b_x_1 = int(new_x - w / 2)
        b_y_1 = int(y - h / 2)
        b_x_2 = int(new_x + w / 2)
        b_y_2 = int(y + h / 2)

        new_image_width = image_width * (1 - cut_rate)

        if b_x_1 < 0:
            b_x_1 = 0

        if b_x_2 > new_image_width:
            b_x_2 = int(new_image_width)


        new_x = str(new_x / new_image_width)
        new_y = str(y / image_height)
        new_w = str((b_x_2 - b_x_1) / new_image_width)
        new_h = str((b_y_2 - b_y_1) / image_height)


        count  = 0

        label_type = line_list[0]

        # 파일에 내용 작성
        new_f = open(label_path + new_name, 'a')
        new_f.write(label_type+' '+new_x+' '+new_y+' '+new_w+' '+new_h)
        count += 1
        if count < len(lines):
            new_f.write('\n')
            new_f.close()
    count_label += 1
print(count_label)



