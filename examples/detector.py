# Stupid python path shit.
# Instead just add darknet.py to somewhere in your python path
# OK actually that might not be a great idea, idk, work in progress
# Use at your own risk. or don't, i don't care

import sys, os
sys.path.append(os.path.join(os.getcwd(),'python/'))

import darknet as dn
import pdb


lib = CDLL("/home/e0358-3/asst-2/darknet/libdarknet.so", RTLD_GLOBAL)

dn.set_gpu(0)
net = dn.load_net("/home/e0358-3/asst-2/darknet/cfg/yolo.cfg", "/home/e0358-3/asst-2/darknet/yolo.weights", 0)
meta = dn.load_meta("/home/e0358-3/asst-2/darknet/cfg/coco.data")
r = dn.detect(net, meta, "data/dog.jpg")
print r

# And then down here you could detect a lot more images like:
# r = dn.detect(net, meta, "data/eagle.jpg")
# print r
# r = dn.detect(net, meta, "data/giraffe.jpg")
# print r
# r = dn.detect(net, meta, "data/horses.jpg")
# print r
# r = dn.detect(net, meta, "data/person.jpg")
# print r

