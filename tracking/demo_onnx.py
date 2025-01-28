import _init_paths
import os
import cv2

import random
import argparse
import numpy as np

import lib.models.models as models

from os.path import exists, join, dirname, realpath
from easydict import EasyDict as edict
from lib.utils.utils import load_pretrain, cxy_wh_2_rect, get_axis_aligned_bbox, load_dataset, poly_iou

from lib.eval_toolkit.pysot.datasets import VOTDataset
# from lib.eval_toolkit.pysot.evaluation import EAOBenchmark
from lib.tracker.lighttrack_ONNX import Lighttrack

import onnxruntime as ort   


def parse_args():
    parser = argparse.ArgumentParser(description='LightTrack Demo')
    parser.add_argument('--video', default=None, type=str, help='test a video', required=True)
    parser.add_argument('--stride', type=int, help='network stride', default=16)
    parser.add_argument('--even', type=int, default=0)
    args = parser.parse_args()

    return args


DATALOADER_NUM_WORKER = 2


def track(siam_tracker, siam_net, video_path, args):
    start_frame, toc = 1, 0

    regions = []
    lost = 0

    cam = cv2.VideoCapture(video_path)

    cnt_frame = 0

    while cam.isOpened():
        ret, frame = cam.read()
        if ret is False:
            break
        cnt_frame += 1
        tic = cv2.getTickCount()
        if cnt_frame == start_frame:  # init
            cx, cy, w, h = 505, 245, 40, 100
            # Draw initial rectangle
            drawing = False
            ix, iy = -1, -1

            def draw_rectangle(event, x, y, flags, param):
                nonlocal ix, iy, cx, cy, w, h, drawing
                if event == cv2.EVENT_LBUTTONDOWN:
                    drawing = True
                    ix, iy = x, y
                elif event == cv2.EVENT_MOUSEMOVE:
                    if drawing:
                        frame_copy = frame.copy()
                        cv2.rectangle(frame_copy, (ix, iy), (x, y), (255, 0, 0), 3)
                        cv2.imshow('LightTrack', frame_copy)
                elif event == cv2.EVENT_LBUTTONUP:
                    drawing = False
                    cx, cy = (ix + x) // 2, (iy + y) // 2
                    w, h = abs(x - ix), abs(y - iy)
                    cv2.rectangle(frame, (ix, iy), (x, y), (255, 0, 0), 3)
                    cv2.imshow('LightTrack', frame)

            cv2.namedWindow('LightTrack')
            cv2.setMouseCallback('LightTrack', draw_rectangle)

            while True:
                cv2.imshow('LightTrack', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('r'):
                    frame = cam.read()[1]
                elif key == ord('q') or key == 27:
                    break
                elif not drawing and key == ord(' '):
                    break
            cv2.rectangle(frame, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), (255, 0, 0), 3)

            target_pos = np.array([cx, cy])
            target_sz = np.array([w, h])

            state = siam_tracker.init(frame, target_pos, target_sz, siam_net)  # init tracker

            
        else: # tracking
            state = siam_tracker.track(state, frame)

            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])

            x, y, w, h = [int(l) for l in location]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        cv2.imshow('LightTrack', frame)
        key = cv2.waitKey(0) & 0xFF
        if key == 27 or key == ord('q'):
            break

    toc += cv2.getTickCount() - tic

    toc /= cv2.getTickFrequency()
    print('Video: {:12s} Time: {:2.1f}s Speed: {:3.1f}fps  Lost {}'.format(video_path, toc, cnt_frame / toc, lost))


def main():
    args = parse_args()

    info = edict()

    info.stride = args.stride

    siam_info = edict()

    siam_info.stride = args.stride
    # build tracker
    siam_tracker = Lighttrack(siam_info, even=args.even)
    # build siamese network
    siam_net_init = ort.InferenceSession(f"onnx_models/backbone.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    siam_net_track = ort.InferenceSession(f"onnx_models/litetrack.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    siam_net = {'template': siam_net_init, 
                'track': siam_net_track}

    track(siam_tracker, siam_net, args.video, args)

if __name__ == '__main__':
    main()
