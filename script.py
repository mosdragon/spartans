import sys
import os
import dlib
import glob
from skimage import io
import cv2
import numpy as np
import math

if len(sys.argv) >= 2:
    predictor_path = sys.argv[1]
else:
    predictor_path = 'shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
# win = dlib.image_window()

def get_image(fname):
    img = cv2.imread(fname)
    # Convert to floating point
    img = np.float32(img) / 255.0;
    return img

def get_shape(fname):
    """
    :fname: string filename of an image

    :shape: the dlib.dlib.full_object_detection shape that contains all
            of the points used in the facial detection
    """
    img = io.imread(fname)
    if img is not None:
        faces = detector(img, upsample_num_times=1)
        print("Number of faces detected: {}".format(len(faces)))
        if faces:
            shape = predictor(img, faces[0])
            return shape

    return None

def scale_down_pic(img, shape):
    """
    Scale down from whatever dimensions to 600 x 600

    """
    if shape:

        points = list(shape.parts())  # convert the generator into a list
        clean_points = [ (p.x, p.y) for p in points]  # make x,y pairs tuples


        left_eye_dest = (180, 200)  # where the left eye will be in output
        right_eye_dest = (420, 200)  # where the right eye will be in output
        
        left_eye_landmark = clean_points[36]  # will be 36th landmark
        right_eye_landmark = clean_points[45]  # will be 45th landmark

        # Now, we will do a similarity transform

        # Must rotate and translate origin picture such that left eye and right
        # eye lie on the same line, and dimensions of img become 600 x 600
        # so face is centered

        # Third point is the first point rotated 60 degrees

        inPoints = [left_eye_landmark, right_eye_landmark]
        outPoints = [left_eye_dest, right_eye_dest]

        s60 = math.sin(60*math.pi/180) # sin of 60 degrees, in radians
        c60 = math.cos(60*math.pi/180)

        inPts = np.copy(inPoints).tolist()
        outPts = np.copy(outPoints).tolist()
        
        xin = c60*(inPts[0][0] - inPts[1][0]) - s60*(inPts[0][1] - inPts[1][1]) + inPts[1][0]
        yin = s60*(inPts[0][0] - inPts[1][0]) + c60*(inPts[0][1] - inPts[1][1]) + inPts[1][1]
        
        inPts.append([np.int(xin), np.int(yin)])
        
        xout = c60*(outPts[0][0] - outPts[1][0]) - s60*(outPts[0][1] - outPts[1][1]) + outPts[1][0]
        yout = s60*(outPts[0][0] - outPts[1][0]) + c60*(outPts[0][1] - outPts[1][1]) + outPts[1][1]
        
        outPts.append([np.int(xout), np.int(yout)])
        
        M = cv2.estimateRigidTransform(np.array([inPts]), np.array([outPts]), False)

        w, h = 600, 600
        smaller = cv2.warpAffine(img, M, (w, h))
        return smaller

    return None



if __name__ == '__main__':
    fname = "images/clinton.jpg"

    path = 'images/'
    w, h = 600, 600

    shapes = []
    smaller_pics = []

    for filePath in os.listdir(path):
        if filePath.endswith(".jpg"):
            fname = os.path.join(path,filePath)

            img = get_image(fname)

            if img is not None:
                shape = get_shape(fname)
                if shape:
                    smaller = scale_down_pic(img, shape)

                    shapes.append(shape)
                    smaller_pics.append(smaller)
                    # cv2.imshow("Smaller", smaller)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

                else:
                    print "No shape"

            else:
                print "No img"

    count = 0
    avg = np.zeros((600, 600, 3))
    for _, pic in zip(shapes, smaller_pics):
        count += 1
        avg += pic

    avg /= count
    cv2.imshow("Smaller", avg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




