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
    img = np.float32(img) / 255.0
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

def get_transform_matrix(source, dest):
    # Third point is the first point rotated 60 degrees from the
    # first vertex in the list, A, which ensures that we find a point that
    # makes all three an equilateral triangle.

    inPoints = np.array(source)
    outPoints = np.array(dest)

    sin60  = math.sin(60 * math.pi / 180) # sin of 60 degrees, in radians
    cos60 = math.cos(60 * math.pi / 180)

    inPts = np.copy(inPoints).tolist()
    outPts = np.copy(outPoints).tolist()
    
    # Before transform
    dx = source[0][0] - source[1][0]
    dy = source[0][1] - source[1][1]

    x3 = cos60 * dx - sin60 * dy + source[1][0]
    y3 = sin60 * dx + cos60 * dx + source[1][1]
    
    inPts.append([np.int(x3), np.int(y3)])

    # After transformation
    dx = dest[0][0] - dest[1][0]
    dy = dest[0][1] - dest[1][1]

    x3 = cos60 * dx - sin60 * dy + dest[1][0]
    y3 = sin60 * dx + cos60 * dx + dest[1][1]
    
    outPts.append([np.int(x3), np.int(y3)])
    
    M = cv2.estimateRigidTransform(np.array([inPts]), np.array([outPts]), False)

    return M

def transform_img(img, M, w, h):
    """
    Scale down from whatever dimensions to w x h
    """
    smaller = cv2.warpAffine(img, M, (w, h))
    return smaller

def transform_points(points, M, w, h):
    """
    """

    new_pts = np.reshape(np.array(points), (68,1,2))
    transformed_pts =  cv2.transform(np.array(new_pts), M)
    return transformed_pts


def rectContains(rect, point):
    """
    Check if a point is inside a rectangle
    """
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True

def calculateDelaunayTriangles(rect, points):
    """
    """
    # TODO: Dig into implementation details more
    subdiv = cv2.Subdiv2D(rect)

    for p in points:
        subdiv.insert((p[0], p[1]))

    triangles = subdiv.getTriangleList()

    delaunayTri = []
    
    for t in triangles:
        pt = []
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))
        
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])        
        
        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            ind = []
            for j in xrange(0, 3):
                for k in xrange(0, len(points)):                    
                    if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)                            
            if len(ind) == 3:                                                
                delaunayTri.append((ind[0], ind[1], ind[2]))

    return delaunayTri


def constrainPoint(p, w, h):
    p =  ( min( max( p[0], 0 ) , w - 1 ) , min( max( p[1], 0 ) , h - 1 ) )
    return p

# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst

def warpTriangle(img1, img2, t1, t2):
    """
    Warps and alpha blends triangular regions from img1 and img2 to img
    """

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = [] 
    t2Rect = []
    t2RectInt = []

    for i in xrange(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))


    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    
    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
    
    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
     
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect


def read_points(path):
    """
    """
    allPoints = []
    for filePath in os.listdir(path):
        if filePath.endswith(".jpg"):
            fname = os.path.join(path,filePath)
            img = get_image(fname)

            shape = get_shape(fname)
            points = [(p.x, p.y) for p in shape.parts()]
            allPoints.append(points)

    return allPoints




def main():
    path = 'images/'
    w, h = 600, 600


    allPoints = read_points(path)

    images = []  # the original images
    for filepath in os.listdir(path):
        if filepath.endswith(".jpg"):
            fname = os.path.join(path, filepath)
            img = cv2.imread(fname)
            images.append(img)
    
    # Destination for left and right eye corners
    left_dest = (int(0.3 * w), int(h / 3))
    right_dest = (int(0.7 * w), int(h / 3))
    eyes_dest = [left_dest, right_dest]

    images_norm = []
    points_norm = []

    boundaryPts = np.array([(0,0), (w/2,0), (w-1,0), (w-1,h/2), ( w-1, h-1 ), ( w/2, h-1 ), (0, h-1), (0,h/2) ])
    # Initialize location of average points to 0s
    points_avg = np.array([(0,0)]* ( len(allPoints[0]) + len(boundaryPts) ), np.float32())
    
    n = len(allPoints[0])
    num_images = len(images)

    for idx, img in enumerate(images):
        points = allPoints[idx]
        eyecornerSrc = [ points[36], points[45] ]

        tform = get_transform_matrix(eyecornerSrc, eyes_dest)

        img_norm = cv2.warpAffine(img, tform, (w, h))

        points2 = np.reshape(np.array(points), (68, 1, 2))
        points = cv2.transform(points2, tform)
        points = np.float32(np.reshape(points, (68, 2)))

        points_additional = np.append(points, boundaryPts, axis=0)

        # Add these to the averaged location of landmarks
        points_avg += points_additional / num_images

        points_norm.append(points_additional)
        images_norm.append(img_norm)


    # Delaunay Triangulation
    rect = (0, 0, w, h)
    dt = calculateDelaunayTriangles(rect, np.array(points_avg))

    output_img = np.zeros((h, w, 3), np.float32())

    # Warp input images to average image landmarks
    for i in xrange(0, len(images_norm)) :
        img = np.zeros((h,w,3), np.float32());
        # Transform triangles one by one
        for j in xrange(0, len(dt)) :
            tin = []; 
            tout = [];
            
            for k in xrange(0, 3) :                
                pIn = points_norm[i][dt[j][k]];
                pIn = constrainPoint(pIn, w, h);
                
                pOut = points_avg[dt[j][k]];
                pOut = constrainPoint(pOut, w, h);
                
                tin.append(pIn);
                tout.append(pOut);
            
            
            warpTriangle(images_norm[i], img, tin, tout);


        # Add image intensities for averaging
        output_img = output_img + img;



    output_img = output_img / num_images
    output = np.float32(output_img) / 255.0
    cv2.imshow('Averaged', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return output

if __name__ == '__main__':
    main()

