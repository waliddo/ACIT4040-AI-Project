#! /usr/bin/env python
import os
import cv2

from face_swap import face_swap
from face_detection import (
    select_face, select_all_faces, files_in_dirs
)
from config import (
    DATASET_DIR, REFERENCE_IMG, FACESWAP_DIR
)


def main():
    faceswap_dir = os.path.join(os.getcwd(), FACESWAP_DIR)
    if not os.path.exists(faceswap_dir):
        os.mkdir(faceswap_dir)
    
    src_img = cv2.imread(REFERENCE_IMG)
    filename1 = os.path.splitext(os.path.basename(REFERENCE_IMG))[0]
    ext = os.path.splitext(REFERENCE_IMG)[1]
    
    src_points, src_shape, src_face = select_face(src_img)
    imgs, tot_imgs = files_in_dirs(DATASET_DIR)
    
    for i in range(tot_imgs):
        # Read images
        dst_img = cv2.imread(imgs[i])

        # Select src face
        # Select dst face
        dst_faceBoxes = select_all_faces(dst_img)

        if dst_faceBoxes is None:
            print(f'Detect 0 Face !!! Image: {i} {imgs[i]}')
            continue
            #exit(-1)
        else:
            output = dst_img
            for k, dst_face in dst_faceBoxes.items():
                try:
                    output = face_swap(src_face, dst_face["face"], src_points,
                                    dst_face["points"], dst_face["shape"],
                                    output)
                except Exception as e:
                    print(f'{e} {i} {imgs[i]}')
            filename2 = os.path.splitext(os.path.basename(imgs[i]))[0]
            
            if not os.path.exists(f'{faceswap_dir}/{filename1}'):
                os.mkdir(f'{faceswap_dir}/{filename1}')
            cv2.imwrite(f'{FACESWAP_DIR}/{filename1}/{filename1}_{filename2}{ext}', output)
        
if __name__ == '__main__':
    main()

    
