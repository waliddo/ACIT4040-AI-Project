import os
import cv2
import numpy as np
import imutils
from imutils import face_utils
import scipy 
from config import (
    TRI_FILE, FACESWAP_DIR, LANDMARKS_DIR, REFERENCE_IMG
)


def files_in_dirs(path):
    """Processes the dataset-directory and returns all files.

    Args:
        path (str): Path to dataset-directory.

    Returns:
        file_list (list): List of all files in the directory. 
        tot_files (int): Number of files in the dataset directory.
    """
    file_list = []
    files = sorted(os.listdir(path))
    tot_files = len(files)
    
    for i in range(tot_files):
        file_list.append(f'{path}/{files[i]}')
    return file_list, tot_files


def generate_facial_landmarks(img_path, detector, predictor):
    """Generates 68 facial landmark points for one input image.

    Args:
        img_path (str): Path to an image.
        detector (dlib_pybind11.fhog_object_detector): dlib default face detector.
        predictor (dlib_pybind11.shape_predictor): dlib facial landmark predictor.

    Returns:
        shape (np.ndarray): 68 (x, y)-coordinates of the facial landmarks.
    """
    # Load the input image, resize it, and convert it to grayscale
    image = cv2.imread(img_path)
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    rects = detector(gray)

    # Loop over the face detections
    for (i, rect) in enumerate(rects):
        # Determine the facial landmarks for the face region, then
        # Convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
    return shape


def get_txtfile(path1, path2):
    """Receives two image paths and returns the paths of their facial landmark txt-file. 

    Args:
        path1 (str)): Path to ref image.
        path2 (str): Path to an image.

    Returns:
        txt_path1 (str): Path to facial landmark txt-file of image in path1.
        txt_path2 (str): Path to facial landmark txt-file of image in path2.
    """
    separated_path1 = path1.split(os.sep)
    separated_path2 = path2.split(os.sep)
    path1 = separated_path1[-1]
    path2 = separated_path2[-1]

    extension1 = os.path.splitext(path1)[1]
    extension2 = os.path.splitext(path2)[1]
    txt_path1 = f'{LANDMARKS_DIR}/reference_img/{path1}'.replace(extension1, '.txt') # Ref image
    txt_path2 = f'{LANDMARKS_DIR}/{path2}'.replace(extension2, '.txt')

    return txt_path1, txt_path2


def write_landmarks_to_file(file_name_path, landmark_pos, dir):
    """Writes all 68 landmark positions to a new txt-file.

    Args:
        file_name_path (str): Path to the file.
        landmark_pos (np.ndarray): Array of all landmark positions.
        dir (str): Name of directory. 
    """
    if not os.path.exists(dir):
        os.mkdir(dir)
    
    file_name = os.path.splitext(os.path.basename(file_name_path))[0]
    
    fw = open(f'{dir}/{file_name}.txt', 'w')
    for value in range(68): 
        fw.write(f'{landmark_pos[value][0]}\t{landmark_pos[value][1]}\n')
    fw.close()
    
    
def read_points(path):
    """Reads all the (x, y) facial landmark coordinates from a given file, 
    and returns the points in the form of a list.

    Args:
        path (str): Path to a file containing facial landmarks.

    Returns:
        points (list): List of several (x, y) facial landmark coordinates.
    """
    points = []

    with open(path) as file:
        for line in file:
            x, y = line.split() 
            points.append((int(x), int(y)))
    return points


def read_tri_points(path):
    """Same as read_points(), but only used for reading data from tri.txt.

    Args:
        path (str): Path to tri.txt file

    Returns:
        points (list): List of several (x, y, z) triangilar (x, y) coordinates.
    """
    points = []

    with open(path) as file:
        for line in file:
            x, y, z = line.split() 
            points.append((int(x), int(y), int(z)))
    return points


def apply_affine_transform(src, src_tri, dst_tri, size):
    """Apply affine transform calculated using src_tri 
    and dst_tri to src and output an image of size.

    Args:
        src (np.ndarray): 
        src_tri (list): 
        dst_tri (list): 
        size (tuple): 

    Returns:
        np.ndarray: 
    """
    # Given a pair of triangles, find the affine transform.
    warp_mat = cv2.getAffineTransform( np.float32(src_tri), np.float32(dst_tri) )
    # Apply the Affine Transform just found to the src image
    return cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )


def rect_contains(rect, point):
    """Check if a point is inside a rectangle

    Args:
        rect (tuple): 
        point (tuple): One (x, y) coordinate from subdiv.getTriangleList().

    Returns:
        True if point is inside a rectangle or False if it is not. 
    """
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[0] + rect[2]:
        return False
    elif point[1] > rect[1] + rect[3]:
        return False
    return True


def warp_triangle(img1, img2, t1, t2):
    """Warps and alpha blends triangular regions from img1 and img2 to img2

    Args:
        img1 (np.ndarray): Pixel values of the reference image.
        img2 (np.ndarray): Copy of the pixel values of the source image.
        t1 (list): Triplet (x, y) coordinates from tri.txt.
        t2 (list): Triplet (x, y) coordinates from tri.txt.
    """
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1_rect = [] 
    t2_rect = []
    t2_rect_int = []

    for i in range(0, 3):
        t1_rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
        t2_rect_int.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))


    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    #img2Rect = np.zeros((r2[3], r2[2]), dtype = img1Rect.dtype)
    
    size = (r2[2], r2[3])

    img2_rect = apply_affine_transform(img1_rect, t1_rect, t2_rect, size)
    
    img2_rect = img2_rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
     
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2_rect 
    
    
def calculate_delaunay_triangles_cv2(rect, points):
    """Performs delaunay triangulation on the points using cv2 and 
    writes the result to a txt-file called tri.txt. The file consists 
    of three columns with each row forming a triangle. The values
    of the rows refers to the index of three (x, y) facial landmark point. 

    Args:
        rect (tuple): 
        points (list): List of the hull convex points. The coordinates of points to triangulate.
    """
    subdiv = cv2.Subdiv2D(rect) # Create subdiv

    # Insert points into subdiv
    for p in points:
        subdiv.insert(list(p)) 
    
    triangle_list = subdiv.getTriangleList()

    fw = open(TRI_FILE, 'w')

    for t in triangle_list:
        pt = []          
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))
        
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])        

        if rect_contains(rect, pt1) and rect_contains(rect, pt2) and rect_contains(rect, pt3):
            ind = []
            #Get face-points (from 68 face detector) by coordinates
            for j in range(0, 3):
                for k in range(0, len(points)):                    
                    if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)    
            # Three points form a triangle. Triangle array corresponds to the file tri.txt in FaceMorph 
            if len(ind) == 3:                                                
                fw.write(f'{ind[0]}\t{ind[1]}\t{ind[2]}\n')
    fw.close()
    

def calculate_delaunay_triangles_scipy(points):
    """Alternative to calculate_delaunay_triangles_cv2(). 

    Args:
        points (list): List of the hull convex points. The coordinates of points to triangulate.
    """
    triangleList = scipy.spatial.Delaunay(points)

    fw = open(TRI_FILE, 'w')
    for i in range(len(triangleList.simplices)):
        fw.write(f'{triangleList.simplices[i][0]}\t{triangleList.simplices[i][1]}\t{triangleList.simplices[i][2]}\n')
    fw.close()


def create_face_swap(path1, path2, method='cv2'):
    """Swaps the face of the reference image to the face of the source image.

    Args:
        path1 (str): Path to ref image.
        path2 (str): Path to source image.
        method (str, optional): Use cv2 or scipy for delaunay triangulation. Defaults to 'cv2'.

    Returns:
        np.ndarray: Face swapped image. 
    """
    # Read images
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    img1 = imutils.resize(img1, width = 500)
    img2 = imutils.resize(img2, width = 500)
    img1_warped = np.copy(img2)

    txt_path1, txt_path2 = get_txtfile(path1, path2)

    # Read array of corresponding points
    points1 = read_points(txt_path1)
    points2 = read_points(txt_path2)

    # Find convex hull
    hull1 = []
    hull2 = []

    hull_index = cv2.convexHull(np.array(points2), returnPoints = False)
            
    for i in range(0, len(hull_index)):
        hull1.append(points1[int(hull_index[i])])
        hull2.append(points2[int(hull_index[i])])

    # Find delanauy traingulation for convex hull points
    size_img2 = img2.shape    
    rect = (0, 0, size_img2[1], size_img2[0])
    
    if method == 'cv2':
        calculate_delaunay_triangles_cv2(rect, hull2)
    else:
        calculate_delaunay_triangles_scipy(hull2)

    # Apply affine transformation to Delaunay triangles
    pnt = read_tri_points(TRI_FILE)
    for i in range(len(pnt)):
        t1 = []
        t2 = []

        #get points for img1, img2 corresponding to the triangles
        for j in range(0, 3):
            t1.append(hull1[pnt[i][j]])
            t2.append(hull2[pnt[i][j]])

        warp_triangle(img1, img1_warped, t1, t2)

    # Calculate Mask
    hull8U = []
    for i in range(0, len(hull2)):
        hull8U.append((hull2[i][0], hull2[i][1]))

    mask = np.zeros(img2.shape, dtype = img2.dtype)  
    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))
    r = cv2.boundingRect(np.float32([hull2]))    
    center = ((r[0]+int(r[2]/2), r[1]+int(r[3]/2)))
    
    # Clone seamlessly.
    return cv2.seamlessClone(np.uint8(img1_warped), img2, mask, center, cv2.NORMAL_CLONE)


def save_face_swap(face_swap, path1, path2):
    """Saves the face swapped image to a directory.
    
    Args:
        face_swap (np.ndarray): Pixel values of the face swapped image.
        path1 (str): Path to reference image.
        path2 (str): Path to another image.
    """
    ext = os.path.splitext(REFERENCE_IMG)[1]
    filename1 = os.path.splitext(os.path.basename(path1))[0]
    filename2 = os.path.splitext(os.path.basename(path2))[0]
    
    if not os.path.exists(f'{FACESWAP_DIR}/{filename1}'):
        os.mkdir(f'{FACESWAP_DIR}/{filename1}')
        
    cv2.imwrite(f'{FACESWAP_DIR}/{filename1}/{filename1}_{filename2}{ext}', face_swap) 
    
    
def create_main_directories():
    """Creates two directories. One for storing facial-landmarks and one for storing the face swaps.
    """
    facial_landmarks_dir = os.path.join(os.getcwd(), LANDMARKS_DIR)
    if not os.path.exists(facial_landmarks_dir):
        os.mkdir(facial_landmarks_dir)
        
    faceswap_dir = os.path.join(os.getcwd(), FACESWAP_DIR)
    if not os.path.exists(faceswap_dir):
        os.mkdir(faceswap_dir)
    
