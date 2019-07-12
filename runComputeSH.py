import glob
import os.path as osp
import argparse
import numpy as np
import cv2
import computeSH

parser = argparse.ArgumentParser()
parser.add_argument('--shapeRoot', default = '/home/ubuntu/CNN-BRDF-ArbitraryShape/Shapes/temp/', \
        help='the root to the shape of objects')
parser.add_argument('--camNum', type=int, default=12, help = 'the number of camera views')
parser.add_argument('--bounceNum', type=int, default=1, help = 'the number of bounce should be considered when render the image')
parser.add_argument('--envNum', type=int, default=1, help = 'the number of environment used for rendering dataset')
parser.add_argument('--matNum', type=int, default=5, help = 'the number of materials used for rendering dataset')
parser.add_argument('--sp', default = 122, type=int, help='the start point of shapes')
parser.add_argument('--ep', default = 600, type=int, help='the end point of shapes')
opt = parser.parse_args()

shapeNames = glob.glob(osp.join(opt.shapeRoot, 'Shape__*') )
shapeNames = sorted(shapeNames)
shapeNum = len(shapeNames)
for shapeCur in range(opt.sp, min(opt.ep, shapeNum) ):
    shapeName = shapeNames[shapeCur]
    print('%d/%d: %s'% (shapeCur, shapeNum, shapeName) )

    for camId in range(0, opt.camNum):
        theta, phi, thetaUp, phiUp = 0, 0, 0, 0
        with open(osp.join(shapeName, 'cam{0}.txt'.format(camId) ), 'r') as f:
            lines = f.readlines()
            angle = lines[0].strip().split(' ')
            angleUp = lines[1].strip().split(', ')
            theta = float(angle[0] )
            phi = float(angle[1] )
            thetaUp = float(angleUp[0] )
            phiUp = float(angleUp[1] )
            cameraLoc = np.array([np.sin(theta)*np.cos(phi),
                np.sin(theta)*np.sin(phi), np.cos(theta)], dtype=np.float32)
            cameraUp = np.array([np.sin(thetaUp) * np.cos(phiUp),
                np.sin(thetaUp) * np.sin(phiUp), np.cos(thetaUp)], dtype=np.float32)

        for envId in  range(0, opt.envNum):
            with open(osp.join(shapeName, 'env{0}.txt'.format(envId) ), 'r') as f:
                envName = f.readlines()[0]
                envName = envName.strip()
                envmap = cv2.imread(envName, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
                envmap = cv2.resize(envmap, (128, 256), interpolation = cv2.INTER_AREA)
                coef = computeSH.computeSHFromImageAfterRotate(envmap, cameraLoc, cameraUp)
                np.save(osp.join(shapeName, 'cam{0}_env{1}.npy'.format(camId, envId) ), coef)
