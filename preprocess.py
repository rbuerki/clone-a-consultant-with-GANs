import configparser
import json
import os
from io import BytesIO
from typing import Dict, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import requests
from bs4 import BeautifulSoup
from PIL import Image

# TODO this has to got in the executing function
# config = configparser.ConfigParser()
# config.read("config.cfg")

# FULL_URL = config.get("WEBSITE", "FULL_URL")
# BASE_URL = config.get("WEBSITE", "BASE_URL")
# FACE_CASCADE_LOCATION = config.get("RESOURCES", FACE_CASCADE_LOCATION)


def scrape_people_content_from_website(full_url: str) -> Dict:
    """Return a dict containing the website's content from the
    `people` section. It's made up of personal information and
    a path to a picture for each employee.
    """
    people_json = requests.get(full_url)
    people_dict = json.loads(people_json.text)
    return people_dict


def generate_full_link_to_picture(
    i: int, people_dict: Dict, base_url: str
) -> str:
    """Parse dict to get the path to the picture for the employee at
    position i and return the full link (combined with the URL base).
    """
    # for i, employee in people_dict["modules"]:
    path_to_pic = people_dict["modules"][i]["data"]["media"]["source"]
    link_to_image = base_url + path_to_pic
    return link_to_image


def load_image_PIL(link_to_image: str) -> PIL.JpegImagePlugin.JpegImageFile:
    """Return picture in PIL image format, with standard RGB color
    scale. (Note: I somehow could not read directly from link into
    OpenCV image format.)
    """
    r = requests.get(link_to_image)
    pil_rgb = Image.open(BytesIO(r.content))
    return pil_rgb


def convert_image_PIL_to_cv_gray(
    pil_rgb: PIL.JpegImagePlugin.JpegImageFile,
) -> np.ndarray:
    """Return grayscale image in form of an numpy array (the image
    format OpenCV uses). Grayscale is the default for face detection
    in OpenCV. Note also that, when there's color, OpenCV does use
    an inverted BGR scale, so we flip that first.
    """
    cv_rgb = np.array(pil_rgb)
    cv_bgr = np.flip(cv_rgb, axis=-1)  # faster subsitute for cv2.RBG2BGR
    cv_gray = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2GRAY)
    return cv_gray


def locate_face_in_image(
    cv_gray: np.array, i: int, face_cascade_loc: str
) -> Tuple(int, int, int, int):
    """Extract the pre-trained face detector, run it, and return the
    coordinates (x,y,w,h) for the the first rectangle containing a 
    detected face in the list. (Note: There should never be more than
    one person on a picture, if more faces are detected we will simply 
    use the first in the list.)
    """
    face_cascade = cv2.CascadeClassifier(face_cascade_loc)
    faces = face_cascade.detectMultiScale(cv_gray)

    # TODO what happens when no face is detected, what get's returned?
    if len(faces) == 0:
        print(f"NO face detected for image nr {i}")
    if len(faces) > 1:
        print(
            f"More than 1 face detected in image nr {i}, will use first only."
        )

    cv_box = tuple(faces[0])
    return cv_box


def convert_cv_box_to_crop_box():
    """Return a "crop box" in the right format [left, upper, right, lower] ?????????????????
    to crop an image.
    """
    pass
