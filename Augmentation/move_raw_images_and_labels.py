import shutil
import os

basic_image_path = '/Users/hahmwj/Desktop/augmentation_dataset/Data/Raw_data/images/'
basic_label_path = '/Users/hahmwj/Desktop/augmentation_dataset/Data/Raw_data/labels/'
basic_img_file_list = os.listdir(basic_image_path)
basic_label_file_list = os.listdir(basic_label_path)


new_images_path = '/Users/hahmwj/Desktop/augmentation_dataset/Data/super_dataset/images/'
new_label_path = '/Users/hahmwj/Desktop/augmentation_dataset/Data/super_dataset/labels/'

count = 0
for image in basic_img_file_list:
    image_root = basic_image_path + image
    new_image_root = new_images_path + image
    shutil.copy(image_root, new_image_root)
    count+=1
print(count)

count = 0
for label in basic_label_file_list:
    label_root = basic_label_path + label
    new_label_root = new_label_path + label
    shutil.copy(label_root, new_label_root)
    count+=1
print(count)


