!pip install pytesseract pillow pandas
!apt-get install -y tesseract-ocr
!pip install pytesseract
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
