import dlib
import sys

from config import (
    PREDICTOR_PATH, LANDMARKS_DIR, 
    DATASET_DIR, REFERENCE_IMG, REFERENCE_DIR
)

from face_swap_utils import (
    files_in_dirs, generate_facial_landmarks, create_main_directories,
    write_landmarks_to_file, create_face_swap, save_face_swap
)

def main():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    imgs, tot_imgs = files_in_dirs(DATASET_DIR)
    
    create_main_directories()
    
    try:
        ref_landmarks = generate_facial_landmarks(REFERENCE_IMG, detector, predictor)
        write_landmarks_to_file(REFERENCE_IMG, ref_landmarks, f'{LANDMARKS_DIR}{REFERENCE_DIR[1:]}') # Landmarks for ref-img
    except Exception as e:
        print(str(e))
        sys.exit(1)
    
    for i in range(tot_imgs):
        source_img = imgs[i]  
        try:
            source_landmarks = generate_facial_landmarks(source_img, detector, predictor)
            write_landmarks_to_file(source_img, source_landmarks, LANDMARKS_DIR)
        except Exception as e:
            print(str(e) + str(i))

        face_swap = create_face_swap(REFERENCE_IMG, source_img)
        save_face_swap(face_swap, REFERENCE_IMG, source_img)
        
if __name__ == '__main__':
    main()