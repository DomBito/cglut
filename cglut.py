#!/usr/bin/python3
import sys
import os.path
import numpy as np
import colorlib
import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Script-based color-grading 3D LUT creator.")
parser.add_argument(metavar="script_file",
                        dest='sfile',
                        type=argparse.FileType('r'),
                        help='script file with the color-grading functions')
parser.add_argument('-s', '--lut_size',
                        dest='lsize', default=64, type=int,
                        help='set LUT size (32, 64, 128, or 256)')
parser.add_argument('-o', '--output',
                        dest='lutfile', default="lut", type=str,
                        help='Name of the lut file.')
parser.add_argument('-c', '--clean',
                    dest='clean', default=False, const=True,
                    action='store_const',
                    help="Doesn't save the preLUT file (slow down your LUT creations if you are exporting them multiple times).")
args = parser.parse_args()

script = args.sfile.readlines()

lsize = args.lsize
if lsize < 2 or lsize > 256:
    sys.exit("\nLUT size must be between 2 and 256!\n")

if os.path.isfile("preLUT"+str(lsize)+".npy"):
    lut = np.load("preLUT"+str(lsize)+".npy")
    print("\nUsing saved preLUT!")
else:
    lut = []
    for i in np.arange(lsize):
        for j in np.arange(lsize):
            for k in np.arange(lsize):
                lut.append(np.asarray([i,j,k])*255.0/(lsize-1))
    lut = np.asarray([lut])
    if not args.clean:
        np.save("preLUT"+str(lsize)+".npy",lut)


def curve1d(curve = [0,0,360,360], channel = 'lightness', smooth = 0,\
            r_range = [0,255], g_range = [0,255], b_range=[0,255],\
            c_range = [0,100], h_range = [0,360], l_range = [0,100],\
            u_range = [0,100], v_range = [0,100], relative_chroma=False):
    global lut
    lut = colorlib._curve1d(lut, curve, channel, smooth, r_range, g_range, b_range,\
            c_range, h_range, l_range, u_range, v_range, relative_chroma)

def rgbcurve(red=[0,0,255,255], green=[0,0,255,255], blue=[0,0,255,255],\
            mode = None, smooth = 0, gimpfile = None,\
            r_range = [0,255], g_range = [0,255], b_range=[0,255],\
            c_range = [0,100], h_range = [0,360], l_range = [0,100],\
            u_range = [0,100], v_range = [0,100], relative_chroma=False):
    global lut
    lut = colorlib._rgbcurve(lut, red, green, blue, mode, smooth, gimpfile,\
            r_range, g_range, b_range, c_range, h_range,\
            l_range, u_range, v_range, relative_chroma)

def local3dcontrast(cont=1.0, rgb1=[0,0,0], rgb2=[255,255,255], radius=10,\
                    mode = None, smooth = 0,center=0.5, relative_chroma=False):
    global lut
    lut = colorlib._3dlocalcontrast(lut,cont,rgb1, rgb2, radius,\
                    mode, smooth, center, relative_chroma)

def perturb(x=[0,100],y=[0,0],domain='lightness',codomain='lightness',\
            r_range = [0,255], g_range = [0,255], b_range=[0,255],\
            c_range = [0,100], h_range = [0,360], l_range = [0,100],\
            u_range = [0,100], v_range = [0,100], smooth=5,\
            relative_chroma = False, mode = 'additive'):
    global lut
    lut = colorlib._perturb(lut,x,y,domain,codomain,\
            r_range, g_range, b_range,c_range, h_range, l_range,\
            u_range, v_range, smooth,relative_chroma, mode)

def balance(grays = [[0,0,0],[255,255,255]]):
    global lut
    lut = colorlib._balance(lut, grays)

def tweak(hue=0.0, chroma=1.0, bright=0.0,\
          r_range = [0,255], g_range = [0,255], b_range=[0,255],\
          c_range = [0,100], h_range = [0,360], l_range = [0,100],\
          u_range = [0,100], v_range = [0,100], smooth=5):
    global lut
    lut = colorlib._tweak(lut, hue, chroma, bright,\
          r_range, g_range, b_range,c_range, h_range,\
          l_range, u_range, v_range, smooth)


for i in range(len(script)):
    exec(script[i].strip('\n'))

ofile = args.lutfile
if ofile[-5:] == ".cube":
    ofile = ofile[:-5]
colorlib.make_LUT(lut,lsize,ofile)
