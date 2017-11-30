import ctypes
import numpy as np
import time
from cv2 import *
import sys
from common import clock, draw_str, draw_rectangles

import math
import random
from ctypes import *

# References
# Darknet Examples- https://github.com/pjreddie/darknet
# Darknet NNPack Examples - https://github.com/digitalbrain79/darknet-nnpack
# Darknet with MKL - https://github.com/jdz1993/yolo-so
# Polymage Examples - https://bitbucket.org/udayb/polymage
# Opencv documentation - https://opencv.org/
# Mkl Documentation - https://software.intel.com/en-us/mkl


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


class YOLO():

    def __init__(self):

        self.lib = None
        self.predict = None
        self.detect = None
        self.set_gpu = None
        self.make_image = None
        self.make_boxes = None
        self.free_ptrs = None
        self.num_boxes = None
        self.make_probs = None
        self.detect = None
        self.reset_rnn = None
        self.load_net = None
        self.free_image = None
        self.letterbox_image = None
        self.load_meta = None
        self.load_image = None
        self.rgbgr_image = None
        self.predict_image = None
        self.network_detect = None
        self.ipl_to_image = None


        self.thresh = 0.5
        self.hier_thresh = 0.5
        self.nms = 0.45

        # self.config_file = './cfg/tiny-yolo.cfg'.encode('utf8')
        # self.weights_file = './weights/tiny-yolo.weights'.encode('utf8')
        self.config_file = './cfg/yolo.cfg'.encode('utf8')
        self.weights_file = './weights/yolo.weights'.encode('utf8')
        self.coco_data = './cfg/coco.data'.encode('utf8')

        self.net = None
        self.meta = None

    


    def yl_construction(self):

        self.lib = CDLL("./libso/libdarknet.so", RTLD_GLOBAL)
        # self.lib = ctypes.cdll.LoadLibrary("./libso/libdarknet.so")
        self.lib.network_width.argtypes = [c_void_p]
        self.lib.network_width.restype = c_int
        self.lib.network_height.argtypes = [c_void_p]
        self.lib.network_height.restype = c_int

        self.predict = self.lib.network_predict
        self.predict.argtypes = [c_void_p, POINTER(c_float)]
        self.predict.restype = POINTER(c_float)

        self.set_gpu = self.lib.cuda_set_device
        self.set_gpu.argtypes = [c_int]

        self.make_image = self.lib.make_image
        self.make_image.argtypes = [c_int, c_int, c_int]
        self.make_image.restype = IMAGE

        self.make_boxes = self.lib.make_boxes
        self.make_boxes.argtypes = [c_void_p]
        self.make_boxes.restype = POINTER(BOX)

        self.free_ptrs = self.lib.free_ptrs
        self.free_ptrs.argtypes = [POINTER(c_void_p), c_int]

        self.num_boxes = self.lib.num_boxes
        self.num_boxes.argtypes = [c_void_p]
        self.num_boxes.restype = c_int

        self.make_probs = self.lib.make_probs
        self.make_probs.argtypes = [c_void_p]
        self.make_probs.restype = POINTER(POINTER(c_float))

        self.detect = self.lib.network_predict
        self.detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]

        self.reset_rnn = self.lib.reset_rnn
        self.reset_rnn.argtypes = [c_void_p]

        self.load_net = self.lib.load_network
        self.load_net.argtypes = [c_char_p, c_char_p, c_int]
        self.load_net.restype = c_void_p

        self.free_image = self.lib.free_image
        self.free_image.argtypes = [IMAGE]

        self.letterbox_image = self.lib.letterbox_image
        self.letterbox_image.argtypes = [IMAGE, c_int, c_int]
        self.letterbox_image.restype = IMAGE

        self.load_meta = self.lib.get_metadata
        self.lib.get_metadata.argtypes = [c_char_p]
        self.lib.get_metadata.restype = METADATA

        self.load_image = self.lib.load_image_color
        self.load_image.argtypes = [c_char_p, c_int, c_int]
        self.load_image.restype = IMAGE

        self.rgbgr_image = self.lib.rgbgr_image
        self.rgbgr_image.argtypes = [IMAGE]

        self.ipl_to_image = self.lib.ipl_to_image
        self.ipl_to_image.argtypes = [c_void_p]
        self.ipl_to_image.restype = IMAGE

        self.predict_image = self.lib.network_predict_image
        self.predict_image.argtypes = [c_void_p, IMAGE]
        self.predict_image.restype = POINTER(c_float)

        self.network_detect = self.lib.network_detect
        self.network_detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]

        self.net = self.load_net(self.config_file, self.weights_file, 0)
        self.meta = self.load_meta(self.coco_data)

    def sample(self, probs):
        s = sum(probs)
        probs = [a/s for a in probs]
        r = random.uniform(0, 1)
        for i in range(len(probs)):
            r = r - probs[i]
            if r <= 0:
                return i
        return len(probs)-1

    def c_array(self, ctype, values):
        arr = (ctype*len(values))()
        arr[:] = values
        return arr

    def array_to_image(self, frame_array):
        frame_array = frame_array.transpose(2,0,1)
        clr = frame_array.shape[0]
        height = frame_array.shape[1]
        width = frame_array.shape[2]
        frame_array = (frame_array/255.0).flatten()
        im_data = self.c_array(c_float, frame_array)
        im = IMAGE(width, height, clr, im_data)
        return im

    def classify(self, net, meta, im):
        out = self.predict_image(net, im)
        res = []
        for i in range(meta.classes):
            res.append((meta.names[i], out[i]))
        res = sorted(res, key=lambda x: -x[1])
        return res

         
    def detect_objects(self, net, meta, image):
        boxes = self.make_boxes(net)
        probs = self.make_probs(net)
        num =   self.num_boxes(net)
        self.network_detect(net, image, self.thresh, self.hier_thresh, self.nms, boxes, probs)
        res = []
        for j in range(num):
            for i in range(meta.classes):
                if probs[j][i] > 0:
                    res.append((meta.names[i], probs[j][i], (boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h)))
        res = sorted(res, key=lambda x: -x[1])
        self.free_ptrs(cast(probs, POINTER(c_void_p)), num)
        return res
    
    '''
    def process_image(self, frame):
       
        bitmap = CreateImageHeader((frame.shape[1], frame.shape[0]), IPL_DEPTH_8U, 3)
        SetData(bitmap, frame.tostring(), 
        frame.dtype.itemsize * 3 * frame.shape[1])

        im = self.ipl_to_image(bitmap)
        self.rgbgr_image(im)
        detection_results = self.detect_objects(self.net, self.meta, im)
        return detection_results
        
    def process_image(self, frame):
        im_file = "temp_img.jpg"
        # imwrite(im_file, frame)
        imwrite(im_file, frame, [int(IMWRITE_JPEG_QUALITY), 10])
        im_file = "temp_img.jpg".encode('utf8')
        detection_results = self.detect_objects(self.net, self.meta, im_file)
        return detection_results
    '''


    def process_image(self, frame):
        im = self.array_to_image(frame)
        self.rgbgr_image(im)
        detection_results = self.detect_objects(self.net, self.meta, im)
        return detection_results
    

class PolyMage():

    def __init__(self, video_file):

        self.video_file = video_file

        self.cv_mode = False
        self.naive_mode = False

        self.harris_mode = False
        self.unsharp_mode = False
        self.bilateral_mode = False
        self.laplacian_mode = False
        self.yolo_mode = False



        self.thresh = 0.001
        self.weight = 3

        self.levels = 4
        self.alpha = 1.0/(self.levels-1)
        self.beta = 1.0

        self.modes = ['Unsharp Mask (Naive)','Unsharp Mask (Opt)','Laplacian (Naive)','Laplacian (Opt)',\
                    'Bilateral (Naive)','Bilateral (Opt)','Harris (OpenCV)','Unsharp Mask (OpenCV)', \
                    'Harris (Naive)','Harris (Opt)', 'YOLO (Naive)', 'YOLO (Opt)']

        self.harris = None
        self.harris_naive = None
        self.unsharp = None
        self.unsharp_naive = None
        self.bilateral = None
        self.bilateral_naive = None

        self.laplacian = None
        self.laplacian_naive = None



        self.sums = {}
        self.frames = {}

        self.libharris = None
        self.libunsharp = None
        self.libbilateral = None
        self.liblaplacian = None

        self.libharris_naive= None
        self.laplacian_naive = None
        self.libunsharp_naive = None
        self.libbilateral_naive = None

        self.yolo_object = None


    def pm_construction(self):

        self.libharris = ctypes.cdll.LoadLibrary("./harris.so")
        self.libharris_naive = ctypes.cdll.LoadLibrary("./harris_naive.so")
        self.libunsharp = ctypes.cdll.LoadLibrary("./unsharp.so")
        self.libunsharp_naive = ctypes.cdll.LoadLibrary("./unsharp_naive.so")
        self.libbilateral = ctypes.cdll.LoadLibrary("./bilateral.so")
        self.libbilateral_naive = ctypes.cdll.LoadLibrary("./bilateral_naive.so")
        self.liblaplacian = ctypes.cdll.LoadLibrary("./laplacian.so")
        self.liblaplacian_naive = ctypes.cdll.LoadLibrary("./laplacian_naive.so")

        self.harris = self.libharris.pipeline_harris
        self.harris_naive = self.libharris_naive.pipeline_harris_naive

        self.unsharp = self.libunsharp.pipeline_mask
        self.unsharp_naive = self.libunsharp_naive.pipeline_mask_naive

        self.bilateral = self.libbilateral.pipeline_bilateral
        self.bilateral_naive = self.libbilateral_naive.pipeline_bilateral_naive

        self.laplacian = self.liblaplacian.pipeline_laplacian
        self.laplacian_naive = self.liblaplacian_naive.pipeline_laplacian_naive

        for mode in self.modes:
            self.sums[mode] = 0.0
            self.frames[mode] = 0


    def pm_initilization(self):
        self.libharris_naive.pool_init()
        self.libharris.pool_init()

        self.libunsharp_naive.pool_init()
        self.libunsharp.pool_init()

        self.liblaplacian_naive.pool_init()
        self.liblaplacian.pool_init()

        self.libbilateral_naive.pool_init()
        self.libbilateral.pool_init()

        self.yolo_object = YOLO()
        self.yolo_object.yl_construction()

        

    def unsharp_mask_cv(self, image,weight,thresh,rows,cols):
        mask = image
        kernelx = np.array([1,4,6,4,1],np.float32) / 16
        kernely = np.array([[1],[4],[6],[4],[1]],np.float32) / 16
        blury = sepFilter2D(image,-1,kernelx,kernely)
        sharpen = addWeighted(image,(1 + weight),blury,(-weight),0)
        th,choose = threshold(absdiff(image,blury),thresh,1,THRESH_BINARY)
        choose = choose.astype(bool)
        np.copyto(mask,sharpen,'same_kind',choose)
        return mask

    def harris_function(self, frame, rows, cols):
        if self.cv_mode:
            gray = cvtColor(frame, COLOR_BGR2GRAY)
            gray = np.float32(gray) / 4.0
            res = cornerHarris(gray, 3, 3, 0.04)
        else:
            res = np.empty((rows, cols), np.float32)
            if self.naive_mode:
                self.harris_naive(ctypes.c_int(cols-2), \
                             ctypes.c_int(rows-2), \
                             ctypes.c_void_p(frame.ctypes.data), \
                             ctypes.c_void_p(res.ctypes.data))
            else:
                self.harris(ctypes.c_int(cols-2), \
                       ctypes.c_int(rows-2), \
                       ctypes.c_void_p(frame.ctypes.data), \
                       ctypes.c_void_p(res.ctypes.data))

        return res

    def unsharp_function(self, frame, rows, cols):
        if self.cv_mode:
            res = self.unsharp_mask_cv(frame,self.weight,self.thresh,rows,cols)
        else:
            res = np.empty((rows-4, cols-4, 3), np.float32)
            if self.naive_mode:
                self.unsharp_naive(ctypes.c_int(cols - 4), \
                          ctypes.c_int(rows - 4), \
                          ctypes.c_float(self.thresh), \
                          ctypes.c_float(self.weight), \
                          ctypes.c_void_p(frame.ctypes.data), \
                          ctypes.c_void_p(res.ctypes.data))
            else:
                self.unsharp(ctypes.c_int(cols-4), \
                    ctypes.c_int(rows-4), \
                    ctypes.c_float(self.thresh), \
                    ctypes.c_float(self.weight), \
                    ctypes.c_void_p(frame.ctypes.data), \
                    ctypes.c_void_p(res.ctypes.data))
        return res

    def laplacian_function(self, frame, rows, cols):

        total_pad = 92
        res = np.empty((rows, cols, 3), np.uint8)
        if self.naive_mode:
            self.laplacian_naive(ctypes.c_int(cols+total_pad), \
                            ctypes.c_int(rows+total_pad), \
                            ctypes.c_float(self.alpha), \
                            ctypes.c_float(self.beta), \
                            ctypes.c_void_p(frame.ctypes.data), \
                            ctypes.c_void_p(res.ctypes.data))
        else:
            self.laplacian(ctypes.c_int(cols+total_pad), \
                      ctypes.c_int(rows+total_pad), \
                      ctypes.c_float(self.alpha), \
                      ctypes.c_float(self.beta), \
                      ctypes.c_void_p(frame.ctypes.data), \
                      ctypes.c_void_p(res.ctypes.data))
        return res

    def bilateral_function(self, frame, rows, cols):

        res = np.empty((rows, cols), np.float32)
        if self.naive_mode:
            self.bilateral_naive(ctypes.c_int(cols+56), \
                            ctypes.c_int(rows+56), \
                            ctypes.c_void_p(frame.ctypes.data), \
                            ctypes.c_void_p(res.ctypes.data))
        else:
            self.bilateral(ctypes.c_int(cols+56), \
                      ctypes.c_int(rows+56), \
                      ctypes.c_void_p(frame.ctypes.data), \
                      ctypes.c_void_p(res.ctypes.data))

        return res


    def draw_string(self, res):

        if self.cv_mode and self.harris_mode:
            draw_str(res, (40, 80),  "Pipeline        :  " + str("OpenCV"))

        elif self.cv_mode and self.unsharp_mode:
            draw_str(res, (40, 80),  "Pipeline        :  " + str("OpenCV"))

        elif self.bilateral_mode or self.harris_mode or self.unsharp_mode or self.laplacian_mode:

            if self.naive_mode:
                draw_str(res, (40, 80),  "Pipeline        :  " + str("PolyMage (Naive)"))
            else:
                draw_str(res, (40, 80),  "Pipeline        :  " + str("PolyMage (Opt)"))

        else:
            draw_str(res, (40, 80),  "Pipeline        :  ")


        if self.harris_mode:
            draw_str(res, (40, 120), "Benchmark    :  " + str("Harris Corner"))
        elif self.bilateral_mode:
            draw_str(res, (40, 120), "Benchmark    :  " + str("Bilateral Grid"))
        elif self.unsharp_mode:
            draw_str(res, (40, 120), "Benchmark    :  " + str("Unsharp Mask"))
        elif self.laplacian_mode:
            draw_str(res, (40, 120), "Benchmark    :  " + str("Local Laplacian"))
        else:
            draw_str(res, (40, 120), "Benchmark    :  ")

        return res



    def update_frame_sums(self, value):

        if self.harris_mode:
            if self.cv_mode:
                self.sums['Harris (OpenCV)'] += value
                self.frames['Harris (OpenCV)'] += 1
            elif self.naive_mode:
                self.sums['Harris (Naive)'] += value
                self.frames['Harris (Naive)'] += 1
            else:
                self.sums['Harris (Opt)'] += value
                self.frames['Harris (Opt)'] += 1
        elif self.unsharp_mode:
            if self.cv_mode:
                self.sums['Unsharp Mask (OpenCV)'] += value
                self.frames['Unsharp Mask (OpenCV)'] += 1
            elif self.naive_mode:
                self.sums['Unsharp Mask (Naive)'] += value
                self.frames['Unsharp Mask (Naive)'] += 1
            else:
                self.sums['Unsharp Mask (Opt)'] += value
                self.frames['Unsharp Mask (Opt)'] += 1

        elif self.laplacian_mode:
            if self.naive_mode:
                self.sums['Laplacian (Naive)'] += value
                self.frames['Laplacian (Naive)'] += 1
            else:
                self.sums['Laplacian (Opt)'] += value
                self.frames['Laplacian (Opt)'] += 1

        elif self.bilateral_mode:
            if self.naive_mode:
                self.sums['Bilateral (Naive)'] += value
                self.frames['Bilateral (Naive)'] += 1
            else:
                self.sums['Bilateral (Opt)'] += value
                self.frames['Bilateral (Opt)'] += 1

        elif self.yolo_mode:

            if self.naive_mode:
                self.sums['YOLO (Naive)'] += value
                self.frames['YOLO (Naive)'] += 1
            else:
                self.sums['YOLO (Opt)'] += value
                self.frames['YOLO (Opt)'] += 1


    def process_video(self):
        self.pm_initilization()
        cap = VideoCapture(self.video_file)
        

        while(cap.isOpened()):
            ret, frame = cap.read()
            frameStart = clock()
            rows = frame.shape[0]
            cols = frame.shape[1]

            if self.harris_mode:
                res = self.harris_function(frame, rows, cols)

            elif self.unsharp_mode:
                res= self.unsharp_function(frame, rows, cols)

            elif self.laplacian_mode:
                res = self.laplacian_function(frame, rows, cols)

            elif self.bilateral_mode:
                res = self.bilateral_function(frame, rows, cols)

            elif self.yolo_mode:
                if self.naive_mode:
                    detections = self.yolo_object.process_image(frame)
                    res = frame
                else:
                    detections = self.yolo_object.process_image(frame)
                    res = frame

            else:
                res = frame

            frameEnd = clock()
            value = frameEnd*1000-frameStart*1000
            self.update_frame_sums(value)

            rectangle(res, (0, 0), (750, 150), (255, 255, 255), thickness=FILLED)
            draw_str(res, (40, 40),      "frame interval :  %.1f ms" % value)

            if self.yolo_mode:
                draw_str(res, (40, 120), "Benchmark    :  " + str("YOLO"))
                draw_rectangles(res, detections)
            else:
                res = self.draw_string(res)

            imshow('Video', res)

            ch = 0xFF & waitKey(1)
            if ch == ord('q'):
                break
            if ch == ord(' '):
                self.cv_mode = not self.cv_mode
            if ch == ord('n'):
                self.naive_mode = not self.naive_mode
            if ch == ord('h'):
                self.harris_mode = not self.harris_mode
                self.bilateral_mode = False
                self.unsharp_mode = False
                self.laplacian_mode = False
                self.yolo_mode = False

            if ch == ord('u'):
                self.unsharp_mode = not self.unsharp_mode
                self.bilateral_mode = False
                self.harris_mode = False
                self.laplacian_mode = False
                self.yolo_mode = False

            if ch == ord('l'):
                self.laplacian_mode = not self.laplacian_mode
                self.unsharp_mode = False
                self.bilateral_mode = False
                self.harris_mode = False
                self.yolo_mode = False

            if ch == ord('b'):
                self.bilateral_mode = not self.bilateral_mode
                self.harris_mode = False
                self.unsharp_mode = False
                self.laplacian_mode = False
                self.yolo_mode = False

            if ch == ord('d'):
                self.yolo_mode = not self.yolo_mode
                self.harris_mode = False
                self.unsharp_mode = False
                self.laplacian_mode = False
                self.bilateral_mode = False


        self.pm_destroy()
        cap.release()
        destroyAllWindows()

    def pm_destroy(self):
        self.libharris_naive.pool_destroy()
        self.libharris.pool_destroy()

        self.libunsharp_naive.pool_destroy()
        self.libunsharp.pool_destroy()

        self.liblaplacian_naive.pool_destroy()
        self.liblaplacian.pool_destroy()

        self.libbilateral_naive.pool_destroy()
        self.libbilateral.pool_destroy()



    def print_metrics(self):
        for mode in self.frames:
            if self.frames[mode]!=0:
                print(("Average frame delay for ",mode," is - ",self.sums[mode]/self.frames[mode],"ms"))


if __name__ == '__main__':


    video_file = './video/test.mp4'
    pm_object = PolyMage(video_file)
    pm_object.pm_construction()
    pm_object.process_video()
    pm_object.print_metrics()

    '''
    yolo_object = YOLO()
    yolo_object.yl_construction()
    yolo_object.process_image()
    '''
