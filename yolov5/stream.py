import cv2

def streaming(im0, p, windows):
    if p not in windows:
        windows.append(p)
        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
        cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
    cv2.imshow(str(p), im0)
    cv2.waitKey(1)  # 1 millisecond
