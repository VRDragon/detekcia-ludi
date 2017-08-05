#!/usr/bin/env python

import numpy as np
import cv2
import cv2.cv as cv
from video import create_capture
from common import clock, draw_str

help_message = '''
USAGE: facedetect.py [--cascade <cascade_fn>] [--nested-cascade <cascade_fn>] [<video_source>]
'''

def detect(img, cascade):
    global vypis
    global necinnost
    global osoby
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags = cv.CV_HAAR_SCALE_IMAGE)
    if len(rects) == 0:
        necinnost=necinnost+1
        if necinnost>=fps:
            print("do"+str(videotime))
            vypis = 0
            necinnost = 0
        return []
    rects[:,2:] += rects[:,:2]
    if vypis <= 0:
        print(str(osoby)+" osôb na obzore od  "+str(videotime))
    vypis = 1
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        global osoby
        osoby=osoby+1
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

if __name__ == '__main__':
    import sys, getopt
    print help_message
    
    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try: video_src = video_src[0]
    except: video_src = 'md_stud_cam2_201702222000.mp4'
    args = dict(args)
    cascade_fn = args.get('--cascade',"haarcascade_mcs_upperbody.xml")
    nested_fn  = args.get('--nested-cascade', "../../data/haarcascades/haarcascade_mcs_lowerbody.xml")
    
    cascade = cv2.CascadeClassifier(cascade_fn)
    nested = cv2.CascadeClassifier(nested_fn)
    global necinnost
    necinnost=0
    global curent_frame_number
    global vypis
    vypis = 0
    current_frame_number=0
    cam = create_capture(video_src, fallback='synth:bg=../cpp/lena.jpg:noise=0.05')

    while True:
        osoby=0
        global osoby
        global current_frame_number
        current_frame_number = current_frame_number+1
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        t = clock()
        fps = cam.get(cv2.cv.CV_CAP_PROP_FPS)
        videotime = current_frame_number / fps
        rects = detect(gray, cascade)
        vis = img.copy()
        draw_rects(vis, rects, (0, 255, 0))
        for x1, y1, x2, y2 in rects:
            osoby=osoby+1
            roi = gray[y1:y2, x1:x2]
            vis_roi = vis[y1:y2, x1:x2]
            subrects = detect(roi.copy(), nested)
            draw_rects(vis_roi, subrects, (255, 0, 0))
        dt = clock() - t
        subrects = detect(roi.copy(), nested)
        draw_rects(vis_roi, subrects, (255, 0, 0))

        draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
        cv2.imshow('facedetect', vis)

        if 0xFF & cv2.waitKey(5) == 27:
            break
    cv2.destroyAllWindows()
