import time
import cv2
import dlib
import numpy as np
import os
import shutil
import imutils
from imutils import face_utils
import tqdm
import pickle
import multiprocessing
import logging
import glob


# logging.basicConfig(filename='run.log', level=logging.DEBUG)
logging.basicConfig(filename='data_prep_run.log', level=logging.INFO)
logging.info("===============================")
logging.info("======= Started New Run =======")
logging.info("===============================")

IMG_SIZE = 64

# TODO: Add continue Algorithm

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

def crop_image(img):
    rects = detector(img, 1)
    if len(rects) != 1:
        logging.warning("ERROR: More than one or no faces detected")
        return
    
    for (i, rect) in enumerate(rects):
        shape = predictor(img, rect)
        shape = face_utils.shape_to_np(shape)
        name, i ,j = "mouth", 48, 68

        (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
        roi = img[y:y+h, x:x+w]
        if w >= h:
            roi = imutils.resize(roi, width=IMG_SIZE, inter=cv2.INTER_CUBIC)
        else:
            logging.info("Height bigger than width")
            roi = imutils.resize(roi, height=IMG_SIZE, inter=cv2.INTER_CUBIC)

    
        return roi

def add_padding(img):
    base_size = IMG_SIZE, IMG_SIZE
    height_top = (IMG_SIZE - img.shape[0]) // 2 
    width_left = (IMG_SIZE - img.shape[1]) // 2 
    base=np.zeros(base_size,dtype=np.uint8)
    base[height_top:height_top+img.shape[0], width_left:width_left+img.shape[1]]=img
    return base

def thread_main(i):
    person_id = i.split("/")[-2]
    video_id = i.split("/")[-1].split(".")[0]
    video_path = os.path.join("data/trainval", person_id, video_id+".mp4")
    label_path = os.path.join("data/trainval", person_id, video_id+".txt")
    os.mkdir(f"data/cropped/{person_id}_{video_id}")


    # Read and Crop Images
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    while success:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        image = crop_image(image)
        if image is not None:
            image = add_padding(image)
            cv2.imwrite(f"data/cropped/{person_id}_{video_id}/{count}.jpg", image)
        else:
            logging.warning(f"Frame {count} for id {person_id}/{video_id} could not be saved")
        success,image = vidcap.read()
        count += 1

    with open(label_path) as f:
        label = f.readline()[7:]

    logging.debug(f"Successfuly saved data/cropped/{person_id}_{video_id}/")
    return f"{person_id}_{video_id}", label

def main():
    paths = glob.glob("data/trainval/**/*.mp4")[:5000]
    pool = multiprocessing.Pool(10)
    labels_file = open("data/cropped/labels.csv", "a")

    pbar = tqdm.tqdm(total=len(paths))
    def update(results):
        pbar.update(1)
        labels_file.write(f"{results[0]}:  {results[1]}")

    for i in paths:
        pool.apply_async(thread_main, args=(i,), callback=update)
    pool.close()
    pool.join()

    logging.info(f"Finished!") 
                
if __name__=="__main__":
    if os.path.exists("data/cropped"):
        print("The cropped folder already exists.")
        answered = False
        while not answered:
            ans = input("Delete and Continue? (y/n)  ").lower()
            if ans == "y":
                shutil.rmtree("data/cropped")
                os.mkdir("data/cropped")
                answered = True
            elif ans == "n":
                exit()
    main()

