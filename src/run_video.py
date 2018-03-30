import argparse
import logging
import time

import cv2
import numpy as np
import os

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

import csv

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
    video_root = '/media/clpsshare/pgupta/0f4db67a-4533-45ff-b2e3-86cef598973d/'
    #video_root = '/media/clpsshare/pgupta/1c92e577-9f23-4564-a8cd-f2b3e2cbf27e/'
    l_vids = os.listdir(video_root)
    l_vids = sorted(l_vids)

    for vid_f in l_vids:
        #logger.debug('cam read+')
        #cam = cv2.VideoCapture(args.camera)
        cap = cv2.VideoCapture(video_root + vid_f)
        vid_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        vid_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        #ret_val, image = cap.read()
        #logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))
        if (cap.isOpened()== False):
            print("Error opening video stream or file")


        # open csv file to write joints
        csvfile = open('../out/pose_' + vid_f[:-4] + '.csv', "w")
        jointwriter = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        columnTitle = ["video_name,frame_n,nose,r_shoulder,r_elbow,r_wrist,l_shoulder,l_elbow,l_wrist"]
        jointwriter.writerow(columnTitle)
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        outfile = '../out/pose_' + vid_f
        #if os.path.isfile(outfield):
        #    continue
        out = cv2.VideoWriter(outfile, fourcc, 25, (320, 240))
        i_fr = 0
        while(cap.isOpened()):
        #for i in range(1,1000):
            ret_val, image = cap.read()

            if ret_val==True:

                humans = e.inference(image)
                image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

                aa = dict
                joints = dict(zip(range(18),[{'x':np.nan, 'y':np.nan}]*18))
                for human in humans:
                    for key in range(18):
                        if key in human.body_parts:
                            body_part = human.body_parts[key]
                            center = (int(body_part.x * vid_width + 0.5), int(body_part.y * vid_height + 0.5))
                            if not ((40 < center[0] < 260) or (50 < center[1] < 200)):
                                break
                            joints[key]['x'] = center[0]
                            joints[key]['y'] = center[1]
                        else:
                            joints[key]['x'] = joints[key]['y'] = np.nan

                    jointwriter.writerow(["{},{},{}-{},{}-{},{}-{},{}-{},{}-{},{}-{},{}-{}".format(
                        vid_f, i_fr,
                        joints[0]['x'], joints[0]['y'],
                        joints[2]['x'], joints[2]['y'],
                        joints[3]['x'], joints[3]['y'],
                        joints[4]['x'], joints[4]['y'],
                        joints[5]['x'], joints[5]['y'],
                        joints[6]['x'], joints[6]['y'],
                        joints[7]['x'], joints[7]['y'])])
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

                print "Video: {} frame: {}".format(vid_f, i_fr)
                i_fr += 1
            else:
                break
            if cv2.waitKey(1) == 27:
                break

        csvfile.close()
        cap.release()
        out.release()
        cv2.destroyAllWindows()
logger.debug('finished+')
