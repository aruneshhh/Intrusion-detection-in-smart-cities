import argparse
from PIL import Image as Img
import time
from pathlib import Path
from sendingmail import sendgmail
import torch.backends.cudnn as cudnn
from numpy import random
from tkinter.ttk import *
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, \
    check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, \
    increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from sort import *
from PIL import Image as PILImage
import pandas
import cv2
import torch
import torchvision
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, \
    check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, \
    increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import numpy as np
import time
# import tensorflow.compat.v1 as tf
from sort import *
from collections import defaultdict
from pygame import mixer
from tkinter import *
from tkinter.ttk import Combobox
import tkinter.messagebox
from datetime import datetime
from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool, ColumnDataSource
import os


mailthreshold=25
mailcount=True
initialobjects=[]
initialframe=True
global mailid

global first_name
global intrusion_type
global intfreezonebbox
global ip_address
def alert():
    mixer.init()
    alert = mixer.Sound('beep-07.wav')
    alert.play()
    time.sleep(0.1)
    alert.play()


def roipoly(image):
    CANVAS_SIZE = (600, 800)

    FINAL_LINE_COLOR = (0, 0, 255)
    WORKING_LINE_COLOR = (0, 0, 255)

    class PolygonDrawer(object):
        def __init__(self, window_name):
            self.window_name = window_name  # Name for our window

            self.done = False  # Flag signalling we're done
            self.current = (0, 0)  # Current position, so we can draw the line-in-progress
            self.points = []  # List of points defining our polygon

        def on_mouse(self, event, x, y, buttons, user_param):
            # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)

            if self.done:  # Nothing more to do
                return

            if event == cv2.EVENT_MOUSEMOVE:
                # We want to be able to draw the line-in-progress, so update current mouse position
                self.current = (x, y)
            elif event == cv2.EVENT_LBUTTONDOWN:
                # Left click means adding a point at current position to the list of points
                print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
                self.points.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN:
                # Right click means we're done
                print("Completing polygon with %d points." % len(self.points))
                self.done = True

        def run(self):
            # Let's create our working window and set a mouse callback to handle events

            cv2.imshow(self.window_name, np.zeros(CANVAS_SIZE, np.uint8))
            cv2.waitKey(1)
            cv2.setMouseCallback(self.window_name, self.on_mouse)

            while (not self.done):
                # This is our drawing loop, we just continuously draw new images
                # and show them in the named window
                canvas = image.copy()
                if (len(self.points) > 0):
                    # Draw all the current polygon segments
                    cv2.polylines(canvas, np.array([self.points]), False, FINAL_LINE_COLOR, 3)
                    # And  also show what the current segment would look like
                    cv2.line(canvas, self.points[-1], self.current, WORKING_LINE_COLOR, 3)
                # Update the window
                cv2.imshow(self.window_name, canvas)
                # And wait 50ms before next iteration (this will pump window messages meanwhile)
                if cv2.waitKey(50) == 27:  # ESC hit
                    self.done = True

            # User finised entering the polygon points, so let's make the final drawing

            cv2.destroyWindow(self.window_name)
            return self.points

    pd = PolygonDrawer("Polygon")
    pd.run()
    return pd.points


def point_inside_polygon(x, y, poly):
    n = len(poly)
    inside = False

    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside
"""Function to Draw Bounding boxes"""


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def draw_boxes(img, bbox, identities=None, categories=None, confidences=None, names=None, colors=None,ip_address=None):
    #single frame
    global mailcount,mailthreshold,initialframe,initialobjects,mailid,first_name
    for i, box in enumerate(bbox):

        x1, y1, x2, y2 = [int(i) for i in box]
        tl = opt.thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness

        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        # conf = confidences[i] if confidences is not None else 0

        color = colors[cat]



        if not opt.nolabel :#and names[cat] == intrusion_type:

            a = int((int(x1) + int(x2)) / 2)
            b = int((int(y1) + int(y2)) / 1.8)
            if ip_address!="0":
                cv2.waitKey(1)

            cv2.polylines(img, np.array([intfreezonebbox]), True, (0, 0, 255), 3)
            if (point_inside_polygon(a, b, intfreezonebbox)) and (id not in initialobjects or initialframe):
                if not opt.nobbox:
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, tl)
                flag = True
                print("intrusion detected")
                #print(names[cat], intrusion_type)
                alert()
                cv2.putText(img, "intrusion", (a, b), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                cv2.circle(img, (a, b), 5, (0, 255, 255), -1)

                status = 1#cat 0 denotes only humans
                if initialframe:
                    initialobjects.append(id)
                label = str(id) + ":" + names[cat] if identities is not None else f'{names[cat]} {confidences[i]:.2f}'
                tf = max(tl - 1, 1)  # font thickness
                t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)



                if mailcount:


                    image = Img.fromarray(img.astype('uint8')).convert('RGB')
                    image.save('intrusion.jpg')
                    sendgmail(mailid,first_name)
                    mailcount=False


                key = cv2.waitKey(1)

            #else:
            #    mailcount=0
    if initialframe:
        print("initial people",initialobjects)
        initialframe=False
    return img



def detect(save_img=False,ip_address=None):

    global intfreezonebbox
    global mailthreshold,mailcount,initialframe,initialobjects
    initialframe=True
    initialobjects=[]

    mailcount=True


    typesofclasses = {0: 'person'}
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() #or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    if not opt.nosave:
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size



    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()



    # getting intrusion area
    capture = cv2.VideoCapture(ip_address)
    ret, initial_frame = capture.read()
    intfreezonebbox = roipoly(initial_frame)
    #initial_bbox = bbox_cal(initial_frame,intfreezonebbox)
    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1
    track_frame = {}
    track_frame = defaultdict(lambda: 0, track_frame)
    count_flag = False
    status = 0
    status_list = [None, None]
    times = []
    t0 = time.time()


    startTime = 0

    # checking feed


    for path, img, im0s, vid_cap in dataset:

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (
                old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                #print("entered")
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                dets_to_sort = np.empty((0, 6))
                # NOTE: We send in detected object class too
                for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                    dets_to_sort = np.vstack((dets_to_sort,
                                              np.array([x1, y1, x2, y2, conf, detclass])))
                #print("outer track")
                if opt.track:
                    #print("track")
                    tracked_dets = sort_tracker.update(dets_to_sort, opt.unique_track_color)
                    tracks = sort_tracker.getTrackers()

                    # draw boxes for visualization
                    if len(tracked_dets) > 0:
                        bbox_xyxy = tracked_dets[:, :4]
                        identities = tracked_dets[:, 8]
                        categories = tracked_dets[:, 4]
                        confidences = None

                        if opt.show_track:
                            # loop over tracks
                            for t, track in enumerate(tracks):
                                track_color = colors[int(track.detclass)] if not opt.unique_track_color else \
                                sort_tracker.color_list[t]

                                [cv2.line(im0, (int(track.centroidarr[i][0]),
                                                int(track.centroidarr[i][1])),
                                          (int(track.centroidarr[i + 1][0]),
                                           int(track.centroidarr[i + 1][1])),
                                          track_color, thickness=opt.thickness)
                                 for i, _ in enumerate(track.centroidarr)
                                 if i < len(track.centroidarr) - 1]
                else:
                    bbox_xyxy = dets_to_sort[:, :4]
                    identities = None
                    categories = dets_to_sort[:, 5]
                    confidences = dets_to_sort[:, 4]

                im0 = draw_boxes(im0, bbox_xyxy, identities, categories, confidences, names, colors,ip_address)

            # Print time (inference + NMS)
            #print("det 0 finished")
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results

            if dataset.mode != 'image' and opt.show_fps:
                currentTime = time.time()

                fps = 1 / (currentTime - startTime)
                startTime = currentTime
                cv2.putText(im0, "FPS: " + str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)


            if view_img:
                cv2.imshow(str(p), im0)


            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''


    print(f'Done. ({time.time() - t0:.3f}s)')


def main(device, classids,bnosave,bfps):
    global opt
    global sort_tracker
    #print("entry")
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=str(device), help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', default=True, help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', default=not bnosave, help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int,default=classids, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')

    parser.add_argument('--track', action='store_true',default=True, help='run tracking')
    parser.add_argument('--show-track', action='store_true',default=False, help='show tracked path')
    parser.add_argument('--show-fps', action='store_true', default=bfps, help='show fps')
    parser.add_argument('--thickness', type=int, default=2, help='bounding box and font size thickness')
    parser.add_argument('--seed', type=int, default=1, help='random seed to control bbox colors')
    parser.add_argument('--nobbox', action='store_true', help='don`t show bounding box')
    parser.add_argument('--nolabel', action='store_true', help='don`t show label')
    parser.add_argument('--unique-track-color', action='store_true', help='show each track in unique color')

    opt = parser.parse_args()
    #print(opt)
    np.random.seed(opt.seed)

    sort_tracker = Sort(max_age=5,
                        min_hits=2,
                        iou_threshold=0.2)


    with torch.no_grad():
        if opt.update:
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect(ip_address=device)







 


def flaskinput(ip,intrtype,bnosave,bfps,address=None,mailId=None,firstname=None):
    global ip_address  # source
    global intrusion_type  # which class u want to find
    global mailid
    global first_name
    mailid=mailId
    first_name=firstname
    print("mailId" ,mailId)
    if intrtype=="NO PERSON ENTRY ZONE":
        intrusion_type=0
    elif intrtype=="NO VEHICLE ENTRY ZONE":
        intrusion_type=[1 ,2, 3 ,5 ,7]
    else:
        intrusion_type = [0 ,1 ,2 ,3,5 ,7]
    if (ip == "webcam"):
        ip_address = 0
    else:
        ip_address=address
    print(ip)
    print(ip_address,intrusion_type,bnosave,bfps)
    main(ip_address,intrusion_type,bnosave,bfps)




