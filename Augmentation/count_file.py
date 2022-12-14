import os

images_path = '/Users/hahmwj/Desktop/augmentation_dataset/Data/super_dataset/images/'
label_path = '/Users/hahmwj/Desktop/augmentation_dataset/Data/super_dataset/labels/'

img_file_list = os.listdir(images_path)
label_file_list = os.listdir(label_path)

for img in img_file_list:
    if '.DS_Store' in img:
        os.remove(images_path + img)

for label in label_file_list:
    if '.DS_Store' in label:
        os.remove(label_path + label)

img_file_list = os.listdir(images_path)
label_file_list = os.listdir(label_path)
print(len(img_file_list))
print(len(label_file_list))
