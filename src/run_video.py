import argparse
import logging
import time

import cv2
import numpy as np
import os

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--zoom', type=float, default=1.0)
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    sub='MG107'
    video_root = '../../../video_data/MG107/0f4db67a-4533-45ff-b2e3-86cef598973d/'
    l_vids = os.listdir(video_root)
    l_vids = sorted(l_vids)

    for vid_f in l_vids:
        #logger.debug('cam read+')
        #cam = cv2.VideoCapture(args.camera)
        cap = cv2.VideoCapture(video_root + vid_f)
        #ret_val, image = cap.read()
        #logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))
        if (cap.isOpened()== False):
            print("Error opening video stream or file")

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
	outfile = '../out/pose_' + vid_f
	#if os.path.isfile(outfield):
	#    continue
        out = cv2.VideoWriter(outfile, fourcc, 25, (320, 240))

        while(cap.isOpened()):
        #for i in range(1,1000):
            ret_val, image = cap.read()


            humans = e.inference(image)
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

            #logger.debug('show+')
            cv2.putText(image,
                        "FPS: %f" % (1.0 / (time.time() - fps_time)),
                        (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

            # write the flipped frame
            out.write(image)

            cv2.imshow('tf-pose-estimation result', image)
            fps_time = time.time()
            height, width = image.shape[:2]

            if cv2.waitKey(1) == 27:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
logger.debug('finished+')
