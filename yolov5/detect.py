# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
import datetime
import csv

@torch.no_grad()
def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=True,  # save results to *.txt
        save_conf=True,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    (save_dir / 'images').mkdir(parents = True, exist_ok=True)
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    save_term = 50
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], [0.0, 0.0, 0.0]
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        now = datetime.datetime.now()
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            #save_path = str(save_dir /'images'/ p.name)  # im.jpg
            save_path = str(save_dir/'images'/p.stem) + f'_{frame}' + '.jpg'
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))


            # 추가적인 파라미터
            cigarett_xyxy = [] #담배 좌표저장

            smoker_num = 0 # 감지된 흡연자 수
            result_of_confusionmatrix = 1 # 해당 확률 이상으로 탐지하면 흡연자로 판단
            smoker_for_remove_cigarett = [] # 만약 흡연자를 확정으로 만든 담배가 있다면 출력에서 제외
            global bb_key # 바운딩 박스 그림 여부를 결정하는 역할
            global checking_plus # 담배가 있는 흡연자와 없는 흡연자를 구분하는 역할
            global print_cigarett_key # 담배 출력 여부를 결정하는 역할
            smoker = False

            det = det[det[:, -1].sort(descending=True)[1]] #결과값을 정렬

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                width = imc.shape[1]
                height = imc.shape[0]

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



                        #만약 위에서 저장된 담배의 중심 x,y좌표가 smoker의 이미지 사이즈에 1.2배한 영역에 들어오게 된다면, 해당 smoker는 100%로 지정, 만약 하나의 담배에 여려명의 사람이 걸렸을 경우 걸린 모든 사람 smoker 100%로 지정
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

                        if conf > result_of_confusionmatrix:  # 돌린 AI모델에서 나온 Smoker의 최대 Confusion matrix값 초과로 탐지시에만 Smoker로 출력
                            if checking_plus == 0:
                                smoker_num += 1
                                condition = 'smoker - cigarett'
                            bb_key = 1
                            smoker = True





                    if c == 2:#만약 Smoker를 100%로 만드는데 사용된 담배의 경우 출력화면에서 배제
                        # 탐지된 객체중 담배의 경우 따로 분리해서 저장
                        smoker = False
                        cigarett_center = [int(xyxy[0] + (xyxy[2] - xyxy[0]) / 2), int(xyxy[1] + (xyxy[3] - xyxy[1]) / 2)]
                        for smoker_xyxy in smoker_for_remove_cigarett:
                            if cigarett_center[0] >= smoker_xyxy[0] and cigarett_center[1] >= smoker_xyxy[1] and cigarett_center[0] <= smoker_xyxy[2] and cigarett_center[1] <= smoker_xyxy[3]:
                                print_cigarett_key = 1
                                break

                        if print_cigarett_key == 0:
                            bb_key = 1
                            condition = 'cigarett'

                        #print(names[c], ' ', bb_key)
                    if save_img or save_crop or view_img:  # Add bbox to image
                        if bb_key == 1:
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]}{conf:.2f}' + condition)
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        else:
                            continue


                    if save_txt:  # Write to file
                        if smoker:
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')



                #if save_crop and smoker_num > 0 and save_term == 100:
                if save_crop and smoker_num > 0:
                    save_term = 0
                    save_one_box(im0, file=save_dir / 'images' / 'Smoker' / f'{now.strftime("%H:%M:%S")}.jpg', BGR=True)



        save_term += 1

        if smoker_num >0:
            print('100% 흡연자', smoker_num, '가 있습니다. ',now)




        # Stream results
        im0 = annotator.result()
        if view_img:
            if p not in windows:
                windows.append(p)
                cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
            cv2.imshow(str(p), im0)
            cv2.waitKey(1)  # 1 millisecond
            #ret, buffer = cv2.imencode('.jpg', im0)
            #frame = buffer.tobytes()
            #yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
           #        bytearray(frame) + b'\r\n')





def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    #print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
