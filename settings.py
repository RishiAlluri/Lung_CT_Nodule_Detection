import os
WORKER_POOL_SIZE = 8

TARGET_VOXEL_MM = 1.00
MEAN_PIXEL_VALUE_NODULE = 41
LUNA_SUBSET_START_INDEX = 0
SEGMENTER_IMG_SIZE = 320

BASE_DIR_SSD = ""
BASE_DIR = ""
EXTRA_DATA_DIR = "resources/"
NDSB3_RAW_SRC_DIR = BASE_DIR + "ndsb_raw/stage12/"
LUNA16_RAW_SRC_DIR = 'input_data/'

NDSB3_EXTRACTED_IMAGE_DIR = BASE_DIR_SSD + "ndsb3_extracted_images/"
LUNA16_EXTRACTED_IMAGE_DIR = "Luna/luna16_extracted_images/"
LUNA_NODULE_DETECTION_DIR = BASE_DIR_SSD + "luna16_nodule_predictions/"

