import cv2
import os
import shutil

images_path = '/Users/hahmwj/Desktop/augmentation_dataset/Data/super_dataset/images/'
label_path = '/Users/hahmwj/Desktop/augmentation_dataset/Data/super_dataset/labels/'

img_file_list = os.listdir(images_path)
label_file_list = os.listdir(label_path)

def mosaic(src, ratio=0.5):
    small = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

c = 0

for img_name in img_file_list:

    if '.DS_Store' in img_name:
        print(img_name)
        continue

    # 저장할 새로운 이름 지정
    split_name = img_name.split('.')
    new_name = split_name[0] + 'RS.' + split_name[1]

    # 새로운 이미지 지정
    img = cv2.imread(images_path + img_name)
    new_img = mosaic(img)
    new_img_full_path = images_path + new_name

    # 새로운 이미지 저장
    cv2.imwrite(new_img_full_path, new_img)
    c += 1
print(c)

c = 0
for label_name in label_file_list:

    if '.DS_Store' in label_name:
        print(label_name)
        continue
    # 저장할 새로운 이름 지정
    split_name = label_name.split('.')
    new_name = split_name[0] + 'RS.' + split_name[1]

    RS_label_file_list = os.listdir(label_path)

    # 이미 이동된 이미지면 추가하지 않는다.
    if new_name in RS_label_file_list:
        continue

    # 파일 복사
    basic_directory = label_path + label_name
    new_directory = label_path + new_name
    shutil.copy(basic_directory, new_directory)
    c += 1
print(c)