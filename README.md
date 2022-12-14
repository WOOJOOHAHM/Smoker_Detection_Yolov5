# Remove Duplicate by remove_duplicate.ipynb in two ways.
### 1. Use difPY package that remove duplicate image in raw image Dataset
### 2. Find same file name, size and name than removed by and condition.
---
# Data Augmentation at 4 ways 
### 1. Gray Scale
### 2. Mirroring
### 3. Change Resolution
### 4. Cropping
---
# Train
### Final Selection 'Super_S_Sgd Epochs 50'
### We tried 72 cased train model to change 4 type of Hyper Parameter
### Dataset
1. Raw_Data
2. Part Augmentation
3. Total Augmentation
4. Super Dataset(* Augmentation)
### Optimizer
1. SGD
2. Adam
3. Momentum
### Pre Trained Model
1. N
2. S
3. M
### Epochs
1. 30
2. 50
---
# Smoker_detection_yolov5
### I customed detect.py file with 2 conditions.
### 1. If model detected Smoker 100%, it reflect in final result.[234 - 240]
---

    if conf > result_of_confusionmatrix:  # 돌린 AI모델에서 나온 Smoker의 최대 Confusion matrix값 초과로 탐지시에만 Smoker로 출력
        if checking_plus == 0:
            smoker_num += 1
            condition = 'smoker - cigarett'
        bb_key = 1
        smoker = True
        
---
### 2. If model detected Smoker and Cigarett in Smoker Bounding Box * 1.2, reflect in final result.[180 - 233]
---

    for *xyxy, conf, cls in reversed(det):
        if c == 2:#만약 Smoker를 100%로 만드는데 사용된 담배의 경우 출력화면에서 배제
            # 탐지된 객체중 담배의 경우 따로 분리해서 정리
            cigarett_center = [int(xyxy[0] + (xyxy[2] - xyxy[0]) / 2), int(xyxy[1] + (xyxy[3] - xyxy[1]) / 2)]
            cigarett_xyxy.append(cigarett_center)

    # Write results
    for *xyxy, conf, cls in reversed(det):
        bb_key = 0
        checking_plus = 0
        print_cigarett_key = 0

        c = int(cls)  # integer class
        increase = 1.2
        if c == 0:#smoker 탐지시 smoker의 범위를 increase 배만큼 증가(1.2)
            b_width = xyxy[2] - xyxy[0] #bounding_box_width
            b_height = xyxy[3] - xyxy[1] #bounding_box_height
            x_center = int((xyxy[2] + xyxy[0]) / 2)
            y_center = int((xyxy[3] + xyxy[1]) / 2)

            i_b_width = int(b_width * increase / 2) #increased bounding box width
            i_b_height = int(b_height * increase / 2) #increased bounding box height

            xyxy[0] = x_center - i_b_width
            xyxy[1] = y_center - i_b_height
            xyxy[2] = x_center + i_b_width
            xyxy[3] = y_center + i_b_height

            if xyxy[0] < 0:
                xyxy[0] = 0

            if xyxy[1] < 0:
                xyxy[1] = 0

            if xyxy[2] > width:
                xyxy[2] = width

            if xyxy[3] > height:
                xyxy[3] = height


            for cigarett in cigarett_xyxy:
                a = cigarett[0] >= xyxy[0] and cigarett[0] <= xyxy[2] and cigarett[1] >= xyxy[1] and cigarett[1] <= xyxy[3]
                if a:
                    smoker_num += 1
                    checking_plus += 1
                    smoker_for_remove_cigarett.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3]])
                    conf = 1.00
                    condition = 'smoker + cigarett'
                    bb_key = 1
                    smoker = True
                    break
                    
---
### print code that alert smokers number and detected time[286 -287]
---

     if smoker_num >0:
        print('100% 흡연자', smoker_num, '가 있습니다. ',now)

---

### Saved .png file named Smoker detected time at ./runs/detect/total_smoke/images/Smoker to Customed plots.py save_one_box function[474 - 489]
---

    def save_one_box(im, file=Path('im.jpg'), gain=1.02, pad=10, square=False, BGR=False, save=True):
        # Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop
        #xyxy = torch.tensor(xyxy).view(-1, 4)
        #b = xyxy2xywh(xyxy)  # boxes
        #if square:
            #b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
        #b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
        #xyxy = xywh2xyxy(b).long()
        #clip_coords(xyxy, im.shape)
        #crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1 if BGR else -1)]
        if save:
            file.parent.mkdir(parents=True, exist_ok=True)  # make directory
            f = str(increment_path(file).with_suffix('.jpg'))
            # cv2.imwrite(f, crop)  # https://github.com/ultralytics/yolov5/issues/7007 chroma subsampling issue
            cv2.imwrite(f, im)
            #Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).save(f, quality=95, subsampling=0)
        
---
