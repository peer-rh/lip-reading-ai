from typing import Any, List, Tuple, Union
from numpy.typing import NDArray
import cv2
import dlib
import numpy as np
import os
from imutils import face_utils
import tqdm
import multiprocessing
import logging
import glob


logging.basicConfig(filename='data_prep_run.log', level=logging.DEBUG)
logging.info("===============================")
logging.info("======= Started New Run =======")
logging.info("===============================")

IMG_SIZE = 64
MIN_SEQ_LEN = 1

# TODO: Add continue Algorithm

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')


def get_face_landmarks(img: NDArray[Any]) -> Any:
    rects = detector(img, 1)
    if len(rects) != 1:
        logging.warning("ERROR: More than one or no faces detected")
        return None

    rect = rects[0]
    shape = predictor(img, rect)
    shape = face_utils.shape_to_np(shape)
    i, j = 48, 68
    face_landmarks = shape[i:j]
    fl_normalized = (face_landmarks - np.min(face_landmarks)) / \
        np.ptp(face_landmarks)
    return fl_normalized


def thread_main(i: str) -> Union[Tuple[str, str], None]:
    person_id = i.split("/")[-2]
    video_id = i.split("/")[-1].split(".")[0]
    video_path = os.path.join("data/trainval", person_id, video_id+".mp4")
    label_path = os.path.join("data/trainval", person_id, video_id+".txt")

    # Read and Crop Images
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    landmarks = []
    while success:
        landmark = get_face_landmarks(image)
        if landmark is not None:
            landmarks.append(landmark.reshape(-1))
        else:
            logging.warning(
                f"Frame {count} for id {person_id}/{video_id} could not be saved")
        success, image = vidcap.read()
        count += 1

    with open(label_path) as f:
        label = f.readline()[7:]
    if len(landmarks) > MIN_SEQ_LEN:
        np.savetxt(
            f"data/landmarks/{person_id}_{video_id}.csv", landmarks, fmt='%1.16f', delimiter=", ")
        logging.debug(
            f"Successfuly saved data/landmarks/{person_id}_{video_id}.csv")
        return f"{person_id}_{video_id}", label
    return None


def remove_already_computed(paths: List[str]) -> List[str]:
    if os.path.exists('data/landmarks/labels.csv'):
        with open("data/landmarks/labels.csv", "r") as f:
            already_computed = [i.split(":")[0] for i in f.readlines()]
            paths = [path for path in paths if "_".join(path.split(
                "/")[-2:]).replace(".mp4", "") not in already_computed]
    return paths


def main() -> None:
    paths = glob.glob("data/trainval/**/*.mp4")
    paths = remove_already_computed(paths)

    pool = multiprocessing.Pool(10)
    labels_file = open("data/landmarks/labels.csv", "a")

    pbar = tqdm.tqdm(total=len(paths))

    def update(results: Tuple[str, str]) -> None:
        pbar.update(1)
        if results is not None:
            labels_file.write(f"{results[0]}:  {results[1]}")

    for i in paths:
        # _= to solve mypy error
        _ = pool.apply_async(thread_main, args=(i,), callback=update)
    pool.close()
    pool.join()

    logging.info("Finished!")


if __name__ == "__main__":
    if not os.path.exists("data/landmarks"):
        os.mkdir("data/landmarks")
    main()
