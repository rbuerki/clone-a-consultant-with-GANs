import configparser
import json
from io import BytesIO
from typing import Dict, Optional, Tuple
from pathlib import Path

import cv2
import numpy as np
import PIL
import requests
from PIL import Image


def parse_config_img_data(path_to_cfg: str) -> Tuple[str, str, str, str, str]:
    """Retrieve necessary config stuff for image retreival
    and preprocessing.
    """
    config = configparser.ConfigParser()
    try:
        config.read(path_to_cfg)
    except FileNotFoundError as e:
        raise (f"Please check the path to the config file: {e}")

    full_url = config.get("SCRAPING", "FULL_URL")
    base_url = config.get("SCRAPING", "BASE_URL")
    path_to_fc = config.get("IMG_PROCESS", "FACE_CASCADE_RELPATH")
    scale_factor = float(config.get("IMG_PROCESS", "SCALE_FACTOR"))
    dsize = tuple(json.loads(config.get("IMG_PROCESS", "DSIZE")))
    return full_url, base_url, path_to_fc, scale_factor, dsize


def scrape_data_from_website(full_url: str) -> Dict:
    """Return a dict containing the target website's content. (Here
    from its `people` section. It's made up of personal information,
    containing a path to a picture for each employee.)
    """
    response_json = requests.get(full_url)
    response_dict = json.loads(response_json.text)
    return response_dict


def instantiate_OpenCV_face_detector(path_to_fc: str) -> cv2.CascadeClassifier:
    """Load pre-trained Haar feature-based cascade classifier
    from a stored XML file and instantiate it.
    """
    abs_path_to_fc = Path(path_to_fc).absolute()
    if not Path.is_file(abs_path_to_fc):
        raise AssertionError(
            f"No valid path for classifier.xml at {abs_path_to_fc}"
        )
    face_cascade = cv2.CascadeClassifier(str(abs_path_to_fc))
    return face_cascade


def generate_full_link_to_image(
    i: int, response_dict: Dict, base_url: str
) -> str:
    """Parse dict to get the path to the image for the employee at
    position i and return the full link (combined with the URL base).
    """
    # for i, employee in people_dict["modules"]:
    path_to_pic = response_dict["modules"][i]["data"]["media"]["source"]
    link_to_image = base_url + path_to_pic
    return link_to_image


def load_image_PIL(
    link_to_image: str,
) -> Optional[PIL.JpegImagePlugin.JpegImageFile]:
    """Return picture in PIL image format, with standard RGB color
    scale. (Note: I somehow could not read directly from link into
    OpenCV image format.) If picture cannot be loaded, return None.
    """
    r = requests.get(link_to_image)
    try:
        pil_rgb = Image.open(BytesIO(r.content))
        return pil_rgb
    except PIL.UnidentifiedImageError:
        return None


def convert_image_PIL_to_cv_gray_and_rgb(
    pil_rgb: PIL.JpegImagePlugin.JpegImageFile,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return grayscale image in form of an numpy array (the image
    format OpenCV uses). Grayscale is the default for face detection
    in OpenCV. Note also that, when there's color, OpenCV does use
    an inverted BGR scale, so we flip that first. And final note:
    No transformation back from grey to color, so we need both.
    """
    cv_rgb = np.array(pil_rgb)
    cv_bgr = np.flip(cv_rgb, axis=-1)  # faster subsitute for cv2.RBG2BGR
    cv_gray = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2GRAY)
    return cv_gray, cv_rgb


def detect_face_in_image(
    cv_gray: np.ndarray, face_cascade: cv2.CascadeClassifier,
) -> Tuple[Tuple[Optional[int]], int]:
    """Run the face detector, and return the coordinates (x,y,w,h) for
    the the FIRST rectangle containing a detected face in the list.
    (Note: There should never be more than one person on a picture,
    if more faces are detected we will simply use the first in the
    list.)If no face is detected, the tuple is empty. Also return the
    number of faces found.
    """
    faces = face_cascade.detectMultiScale(cv_gray)
    n_faces = len(faces)
    if n_faces == 0:
        cv_box = faces
    else:
        cv_box = tuple(faces[0])
    return cv_box, len(faces)


def crop_image(
    cv_gray_in: np.ndarray,
    cv_box: Tuple[int, int, int, int],
    scale_factor: float,
) -> np.ndarray:
    """Return a cropped square cutout of the original grayscale image
    centered around the face. The scale factor can be used to add
    some margin around the face (if set > 1).
    """
    x, y, w, h = cv_box
    x_delta = int(max((scale_factor - 1) * w, 0))
    y_delta = int(max((scale_factor - 1) * h, 0))

    # Use np.array slicing for the cropping
    cv_gray_out = cv_gray_in[
        y - y_delta : y + y_delta + h, x - x_delta : x + x_delta + w
    ].copy()
    return cv_gray_out


def resize_image_cv(
    cv_in: np.ndarray, dsize: Tuple[int, int] = None
) -> np.ndarray:
    """Resize image to given size. If no dsize tuple (width,
    height in pixels) is passed, the size does not change.
    """
    if dsize:
        cv_out = cv2.resize(cv_in, dsize)
    else:
        cv_out = cv_in
    return cv_out


def save_final_image(face_final: np.ndarray, dir_path: str, i: int):
    pass


def main(path_to_cfg):
    """[summary]
    """
    count_invalid_img = count_no_face_img = count_multi_face_img = 0

    full_url, base_url, path_to_fc, scale_factor, dsize = parse_config_img_data(
        path_to_cfg
    )
    employee_data = scrape_data_from_website(full_url)
    face_cascade = instantiate_OpenCV_face_detector(path_to_fc)

    for i, _ in enumerate(employee_data["modules"]):
        link_to_image = generate_full_link_to_image(i, employee_data, base_url)
        pil_rgb = load_image_PIL(link_to_image)
        if pil_rgb is None:
            count_invalid_img += 1
            continue
        else:
            cv_gray, cv_rgb = convert_image_PIL_to_cv_gray_and_rgb(pil_rgb)
            cv_box, n_faces = detect_face_in_image(cv_gray, face_cascade)
            if n_faces == 0:
                print(f"NO face detected for image nr {i}.")
                count_no_face_img += 1
                continue
            else:
                if n_faces > 1:
                    print(
                        f"More than 1 face detected in image nr {i},",
                        "will use first only.",
                    )
                    count_multi_face_img += 1
                face_rgb = crop_image(cv_rgb, cv_box, scale_factor=1.1)
                face_final = resize_image_cv(face_rgb, dsize)
                return face_final


if __name__ == "__main__":
    main("config.cfg")
