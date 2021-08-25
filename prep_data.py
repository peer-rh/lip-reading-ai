import cv2
import dlib
import numpy as np
import os
import shutil
import glob
import imutils
from imutils import face_utils
from tqdm import tqdm
import imageio

import logging

# logging.basicConfig(filename='run.log', level=logging.DEBUG)
logging.basicConfig(filename='data_prep_run.log', level=logging.INFO)
logging.info("===============================")
logging.info("======= Started New Run =======")
logging.info("===============================")

IMG_SIZE = 64

people = os.listdir("data/dataset/dataset")
folder_enum = ['01','02','03','04','05','06','07','08', '09', '10']
instances = ['01','02','03','04','05','06','07','08', '09', '10']

words = ['Begin', 'Choose', 'Connection', 'Navigation', 'Next', 'Previous', 'Start', 'Stop', 'Hello', 'Web']          
words_di = {i:words[i] for i in range(len(words))}

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



def main():
    for i, person_idx in tqdm(enumerate(people), desc="Person", total=len(people)):
        for word_idx, word in tqdm(enumerate(folder_enum), leave=False, desc="Word", total=len(folder_enum)):
            for j, instance in tqdm(enumerate(instances), leave=False, desc="Instance", total=len(instances)):
                path = os.path.join("data/dataset/dataset", person_idx, "words", word, instance)
                img_paths = glob.glob(path + "/color*")
                imgs = []
                for k, img_path in enumerate(img_paths):
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
                    img = crop_image(img)
                    if img is not None:
                        img = add_padding(img)
                        imgs.append(img)
                    else:
                        logging.warning(f"Frame {k} for instance {j} of word {word} for person {person_idx} not added to GIF")
                # Saving it as a gif for easier tf loading
                imageio.mimsave(f"data/prepared_data/{word}/{i*10+j+1}.gif", imgs)
                logging.debug(f"Successfuly saved data/prepared_data/{word}/{i*10+j+1}.gif")
            logging.info(f"Finished Word: {word} {words[word_idx]}") 
        logging.info(f"Finished Person: {person_idx}") 
                
if __name__=="__main__":
    if os.path.exists("data/prepared_data"):
        print("The prepared_data folder already exists.")
        answered = False
        while not answered:
            ans = input("Delete and Continue? (y/n)  ").lower()
            if ans == "y":
                shutil.rmtree("data/prepared_data")
                os.mkdir("data/prepared_data")
                answered = True
            elif ans == "n":
                exit()
    else:
        os.mkdir("data/prepared_data")
        answered = True

    for word in folder_enum:
        os.mkdir(f"data/prepared_data/{word}/") 

    main()

