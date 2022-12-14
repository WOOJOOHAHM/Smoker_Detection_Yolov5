from PIL import Image
import os
import shutil

# 이미지 흑백처리 함수
def bw_transepose(image_path):
    B_W_set = set()

    image = Image.open(image_path)
    img_size = image.size
    for i in range(0, img_size[0]):  # x방향 탐색
        for j in range(0, img_size[1]):  # y방향 탐색

            # i,j 위치에서의 RGB 취득
            rgb = image.getpixel((i, j))

            # rgb 평균 값 계산
            if type(rgb) == int:
                B_W_set.add(image_path)
                continue
            else:
                rgb_num = round((rgb[0] + rgb[1] + rgb[2]) / 3)

                # 흑백으로 변경
                image.putpixel((i, j), (rgb_num, rgb_num, rgb_num))
    return image, B_W_set


images_path = '/Users/hahmwj/Desktop/augmentation_dataset/Data/super_dataset/images/'
label_path = '/Users/hahmwj/Desktop/augmentation_dataset/Data/super_dataset/labels/'


img_file_list = os.listdir(images_path)
label_file_list = os.listdir(label_path)

count = 0
for img_name in img_file_list:
    # DS_Store Error 해결
    if '.DS_Store' in img_name:
        print(img_name)
        continue#

     #새로운 파일 이름 지정
    split_name = img_name.split('.')
    new_name = split_name[0] + 'BW.' + split_name[1]

     #이미 이동된 이미지면 추가하지 않는다.
    BW_img_file_list = os.listdir(images_path)
    if new_name in BW_img_file_list:
        continue

    # 이미지 흑백처리
    #1. 기존 변환할 이미지 불러오기
    image_path = images_path + img_name
    image, BW_set = bw_transepose(image_path)



    #OSError: cannot write mode RGBA as JPEG 에러 처리
    image = image.convert("RGB")

    # 이미지 저장
    image.save(images_path + new_name)
    count += 1

print(count)
# 라벨은 동일 이름만 변경해서 작성
count = 0
for label_name in label_file_list:

    # 새로운 파일 이름 지정
    split_name = label_name.split('.')
    new_name = split_name[0] + 'BW.' + split_name[1]

    BW_label_file_list = os.listdir(label_path)

    # 이미 이동된 이미지면 추가하지 않는다.
    if new_name in BW_label_file_list:
        continue

    # 파일 복사
    basic_directory = label_path + label_name
    new_directory = label_path + new_name
    shutil.copy(basic_directory, new_directory)
    count += 1
print(count)

