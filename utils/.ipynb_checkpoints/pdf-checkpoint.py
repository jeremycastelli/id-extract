import numpy as np
from pdf2image import convert_from_path

def pdfToNumpyImages(file):
    images = convert_from_path(file, dpi=300)
    return [np.array(image) for image in images]

def pdfToImageFiles(file, dest):
    return convert_from_path(file, dpi=300, output_folder=dest, fmt='png', paths_only=True)