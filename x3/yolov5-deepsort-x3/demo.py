from AIDetector_x3 import Detector
import imutils
import cv2

def main():

    name = 'demo'
    det = Detector()
    cap = cv2.VideoCapture('test1.mp4')
    fps = int(cap.get(5))
    print('fps:', fps)
    t = int(1000/fps)

    videoWriter = None

    while True:

        # try:
        _, im = cap.read()
        if im is None:
            break
        result = det.feedCap(im)
        result = result['frame']
        result = imutils.resize(result, height=500)
        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v')  # opencv3.0
            videoWriter = cv2.VideoWriter(
                'result.mp4', fourcc, fps, (result.shape[1], result.shape[0]))

        videoWriter.write(result)
if __name__ == '__main__':
    
    main()