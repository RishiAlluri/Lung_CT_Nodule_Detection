# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'CT_Viewer.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!

import os
from moviepy.editor import ImageSequenceClip
from PyQt5 import QtCore, QtGui, QtWidgets

import pylidc as pl
from matplotlib.patches import Circle
from PIL import Image
import glob
import settings
import helpers
import sys
import glob
import random
import pandas
import ntpath
import cv2
import numpy
from typing import List, Tuple
from keras.optimizers import Adam, SGD
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D, merge, Convolution3D, MaxPooling3D, UpSampling3D, LeakyReLU, BatchNormalization, Flatten, Dense, Dropout, ZeroPadding3D, AveragePooling3D, Activation
from keras.models import Model, load_model, model_from_json
from keras.metrics import binary_accuracy, binary_crossentropy, mean_squared_error, mean_absolute_error
from keras import backend as K
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import math
from step3_predict_nodules import *

# limit memory usage..
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import step2_train_nodule_detector
import shutil
import SimpleITK  # conda install -c https://conda.anaconda.org/simpleitk SimpleITK
#from bs4 import BeautifulSoup #  conda install beautifulsoup4, coda install lxml
from plotting_functions import *

random.seed(1321)
numpy.random.seed(1321)



config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1681, 1056)
        MainWindow.setStyleSheet("background-color: rgb(135,206,250)")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(310, -10, 240, 51))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(110, 40, 631, 591))
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(350, 700, 121, 61))
        self.pushButton_2.setStyleSheet("background-color :rgb(80,80,80)")
        self.pushButton_2.setObjectName("pushButton_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(1000, 40, 631, 591))
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        self.tableView = QtWidgets.QTableView(self.centralwidget)
        self.tableView.setGeometry(QtCore.QRect(680, 720, 431, 251))
        self.tableView.setObjectName("tableView")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(1140, -10, 240, 51))
        self.label_4.setObjectName("label_4")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(1300, 700, 121, 61))
        self.pushButton_4.setStyleSheet("background-color :rgb(80,80,80)")
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(1240, 900, 191, 71))
        self.pushButton.setStyleSheet("background-color :rgb(80,80,80)")
        self.pushButton.setObjectName("pushButton")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(350, 660, 101, 31))
        self.label_5.setText("")
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(1300, 660, 101, 31))
        self.label_6.setText("")
        self.label_6.setObjectName("label_6")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1681, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        self.pushButton.clicked.connect(self.browse_data1)
        self.pushButton_2.clicked.connect(self.plot_next_gif)
        
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "          Predicted Scan Nodules"))
        self.label.setFont(QtGui.QFont("Arial", 10))
        palette = self.label.palette()
        palette.setColor(self.label.foregroundRole(), QtGui.QColor(255,255,255))
        self.label.setPalette(palette)
        self.pushButton_2.setText(_translate("MainWindow", "Next Nodule"))
        self.pushButton_2.setStyleSheet('QPushButton {color: white;}')
        self.label_4.setText(_translate("MainWindow", "          Real Scan Nodules"))
        self.label_4.setPalette(palette)
        self.label_4.setFont(QtGui.QFont("Arial", 10))
        self.pushButton_4.setText(_translate("MainWindow", "Next Nodule"))
        self.pushButton_4.setStyleSheet('QPushButton {color: black;}')
        self.pushButton.setText(_translate("MainWindow", "Browse/Process CT"))
        self.pushButton.setStyleSheet('QPushButton {color: black;}')
    def browse_data1(self):
        
        data_path , _=QtWidgets.QFileDialog.getOpenFileName(None,'Open File',r"C:\Users\Ahmed\Desktop\CT_Gui")
        self.data_path = data_path
        
        #for magnification in [1, 1.5, 2]:
        if True:

            #for magnification in [1]:
            version = 2
            holdout = 0
            CONTINUE_JOB = True
            only_patient_id = None  # "ebd601d40a18634b100c92e7db39f585"
            magnification = 1
#                    predict_cubes("models/model_luna_posnegndsb_v" + str(version) + "__fs_h" + str(holdout) + "_end.hd5", CONTINUE_JOB, only_patient_id=only_patient_id, magnification=magnification, flip=False, train_data=True, holdout_no=holdout, ext_name="luna_posnegndsb_v" + str(version), fold_count=2)
#                    if holdout == 0:
            self.process_images(self.data_path)
            predict_cubes("models/model_luna_posnegndsb_v" + str(version) + "__fs_h" + str(holdout) + "_end.hd5", CONTINUE_JOB, only_patient_id=only_patient_id, magnification=magnification, flip=False, train_data=False, holdout_no=holdout, ext_name="luna_posnegndsb_v" + str(version), fold_count=2)
            
        
            #os.copy(self.data_path, 'Luna/luna16_extracted_images')
        
#            if True:
                #for magnification in [1]:  #
                    #predict_cubes("models/model_luna16_full__fs_best.hd5", CONTINUE_JOB, only_patient_id=only_patient_id, magnification=magnification, flip=False, train_data=True, holdout_no=None, ext_name="luna16_fs")
                #    predict_cubes("models/model_luna16_full__fs_best.hd5", CONTINUE_JOB, only_patient_id=only_patient_id, magnification=magnification, flip=False, train_data=False, holdout_no=None, ext_name="luna16_fs")
        
            
            #self.predict_cubes(self.data_path[:-4],"models/model_luna16_full__fs_best.hd5", magnification=magnification, holdout_no=None, ext_name="luna16_fs")
            radii , centroids,chances,image_array = get_nodules(self.data_path.split('\\')[-1])
            paths, self.sub_dfs = plot(radii,centroids,chances,image_array)
            #print("Done")
            self.sub_dfs_gen = (df_ for df_ in self.sub_dfs)
            frames = []
            for set_ in paths:
                frames = []
                for path in set_:
                    frames.append(cv2.imread(path))
                gif('{}_{}.gif'.format(set_[0].split('.')[0],set_[-1].split('.')[0]),np.array(frames))
            self.gif_ls = glob.iglob("*.gif")
            
            self.plot_next_gif()
            
            
                    
    def plot_next_gif(self):
        try:
            movie = QtGui.QMovie(next(self.gif_ls))
        except:
            self.gif_ls = glob.iglob("*.gif")
            self.sub_dfs_gen = (df_ for df_ in self.sub_dfs)
            movie = QtGui.QMovie(next(self.gif_ls))
        movie.setScaledSize(QtCore.QSize(631, 591))
        self.label_2.setMovie(movie)
        movie.start()
        df_tmp = next(self.sub_dfs_gen)
        chance = df_tmp['chances'].values[0]
        radius = df_tmp['rad'].values[0]
        print(df_tmp)
        self.textBrowser.setText('This nodule has a probability of {} to be Malignant, and its radius is {}'.format(chance,radius))
        
                    
    def prepare_image_for_net3D(self,img):
        img = img.astype(numpy.float32)
        img -= MEAN_PIXEL_VALUE
        img /= 255.
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2], 1)
        return img



    def filter_patient_nodules_predictions(self,df_nodule_predictions: pandas.DataFrame, patient_id, view_size):
        src_dir = ''
        patient_mask = helpers.load_patient_images('Luna\\luna16_extracted_images\\' + patient_id + '\\', src_dir, "*_m.png")
        delete_indices = []
        for index, row in df_nodule_predictions.iterrows():
            z_perc = row["coord_z"]
            y_perc = row["coord_y"]
            center_x = int(round(row["coord_x"] * patient_mask.shape[2]))
            center_y = int(round(y_perc * patient_mask.shape[1]))
            center_z = int(round(z_perc * patient_mask.shape[0]))

            mal_score = row["diameter_mm"]
            start_y = center_y - view_size / 2
            start_x = center_x - view_size / 2
            nodule_in_mask = False
            for z_index in [-1, 0, 1]:
                img = patient_mask[z_index + center_z]
                start_x = int(start_x)
                start_y = int(start_y)
                view_size = int(view_size)
                img_roi = img[start_y:start_y+view_size, start_x:start_x + view_size]
                if img_roi.sum() > 255:  # more than 1 pixel of mask.
                    nodule_in_mask = True

            if not nodule_in_mask:
                print("Nodule not in mask: ", (center_x, center_y, center_z))
                if mal_score > 0:
                    mal_score *= -1
                df_nodule_predictions.loc[index, "diameter_mm"] = mal_score
            else:
                if center_z < 30:
                    print("Z < 30: ", patient_id, " center z:", center_z, " y_perc: ",  y_perc)
                    if mal_score > 0:
                        mal_score *= -1
                    df_nodule_predictions.loc[index, "diameter_mm"] = mal_score


                if (z_perc > 0.75 or z_perc < 0.25) and y_perc > 0.85:
                    print("SUSPICIOUS FALSEPOSITIVE: ", patient_id, " center z:", center_z, " y_perc: ",  y_perc)

                if center_z < 50 and y_perc < 0.30:
                    print("SUSPICIOUS FALSEPOSITIVE OUT OF RANGE: ", patient_id, " center z:", center_z, " y_perc: ",  y_perc)

        df_nodule_predictions.drop(df_nodule_predictions.index[delete_indices], inplace=True)
        return df_nodule_predictions


    def filter_nodule_predictions(self,only_patient_id=None):
        src_dir = settings.LUNA_NODULE_DETECTION_DIR
        for csv_index, csv_path in enumerate(glob.glob(src_dir + "*.csv")):
            file_name = ntpath.basename(csv_path)
            patient_id = file_name.replace(".csv", "")
            print(csv_index, ": ", patient_id)
            if only_patient_id is not None and patient_id != only_patient_id:
                continue
            df_nodule_predictions = pandas.read_csv(csv_path)
            self.filter_patient_nodules_predictions(df_nodule_predictions, patient_id, CUBE_SIZE)
            df_nodule_predictions.to_csv(csv_path, index=False)


    def make_negative_train_data_based_on_predicted_luna_nodules():
        src_dir = settings.LUNA_NODULE_DETECTION_DIR
        pos_labels_dir = settings.LUNA_NODULE_LABELS_DIR
        keep_dist = CUBE_SIZE + CUBE_SIZE / 2
        total_false_pos = 0
        for csv_index, csv_path in enumerate(glob.glob(src_dir + "*.csv")):
            file_name = ntpath.basename(csv_path)
            patient_id = file_name.replace(".csv", "")
            # if not "273525289046256012743471155680" in patient_id:
            #     continue
            df_nodule_predictions = pandas.read_csv(csv_path)
            pos_annos_manual = None
            manual_path = settings.MANUAL_ANNOTATIONS_LABELS_DIR + patient_id + ".csv"
            if os.path.exists(manual_path):
                pos_annos_manual = pandas.read_csv(manual_path)

            self.filter_patient_nodules_predictions(df_nodule_predictions, patient_id, CUBE_SIZE, luna16=True)
            pos_labels = pandas.read_csv(pos_labels_dir + patient_id + "_annos_pos_lidc.csv")
            print(csv_index, ": ", patient_id, ", pos", len(pos_labels))
            patient_imgs = helpers.load_patient_images(patient_id, settings.LUNA16_EXTRACTED_IMAGE_DIR, "*_m.png")
            for nod_pred_index, nod_pred_row in df_nodule_predictions.iterrows():
                if nod_pred_row["diameter_mm"] < 0:
                    continue
                nx, ny, nz = helpers.percentage_to_pixels(nod_pred_row["coord_x"], nod_pred_row["coord_y"], nod_pred_row["coord_z"], patient_imgs)
                diam_mm = nod_pred_row["diameter_mm"]
                for label_index, label_row in pos_labels.iterrows():
                    px, py, pz = helpers.percentage_to_pixels(label_row["coord_x"], label_row["coord_y"], label_row["coord_z"], patient_imgs)
                    dist = math.sqrt(math.pow(nx - px, 2) + math.pow(ny - py, 2) + math.pow(nz- pz, 2))
                    if dist < keep_dist:
                        if diam_mm >= 0:
                            diam_mm *= -1
                        df_nodule_predictions.loc[nod_pred_index, "diameter_mm"] = diam_mm
                        break

                if pos_annos_manual is not None:
                    for index, label_row in pos_annos_manual.iterrows():
                        px, py, pz = helpers.percentage_to_pixels(label_row["x"], label_row["y"], label_row["z"], patient_imgs)
                        diameter = label_row["d"] * patient_imgs[0].shape[1]
                        # print((pos_coord_x, pos_coord_y, pos_coord_z))
                        # print(center_float_rescaled)
                        dist = math.sqrt(math.pow(px - nx, 2) + math.pow(py - ny, 2) + math.pow(pz - nz, 2))
                        if dist < (diameter + 72):  #  make sure we have a big margin
                            if diam_mm >= 0:
                                diam_mm *= -1
                            df_nodule_predictions.loc[nod_pred_index, "diameter_mm"] = diam_mm
                            print("#Too close",  (nx, ny, nz))
                            break

            df_nodule_predictions.to_csv(csv_path, index=False)
            df_nodule_predictions = df_nodule_predictions[df_nodule_predictions["diameter_mm"] >= 0]
            df_nodule_predictions.to_csv(pos_labels_dir + patient_id + "_candidates_falsepos.csv", index=False)
            total_false_pos += len(df_nodule_predictions)
        print("Total false pos:", total_false_pos)


    def predict_cubes(self,path,model_path, magnification=1, holdout_no=-1, ext_name="", fold_count=2):

        dst_dir = settings.LUNA_NODULE_DETECTION_DIR

        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        holdout_ext = ""

        dst_dir += "predictions" + str(int(magnification * 10)) + holdout_ext  + "_" + ext_name + "/"
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        sw = helpers.Stopwatch.start_new()
        model = step2_train_nodule_detector.get_net(input_shape=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 1), load_weight_path=model_path)
    
        patient_id = path

        all_predictions_csv = []
    
        
        if holdout_no is not None:
            patient_fold = helpers.get_patient_fold(patient_id)
            patient_fold %= fold_count

        print( ": ", patient_id)
        csv_target_path = dst_dir + patient_id.split('/')[-1] + ".csv"
        print(patient_id)
    
        try:
            patient_img = helpers.load_patient_images('Luna\\luna16_extracted_images\\' + patient_id + '\\', '', "*_i.png", [])
        except:
            print('Please Re-Process the dicom file again')
    
        if magnification != 1:
            patient_img = helpers.rescale_patient_images(patient_img, (1, 1, 1), magnification)

        patient_mask = helpers.load_patient_images('Luna\\luna16_extracted_images\\' + patient_id + '\\','', "*_m.png", [])
        if magnification != 1:
            patient_mask = helpers.rescale_patient_images(patient_mask, (1, 1, 1), magnification, is_mask_image=True)

            # patient_img = patient_img[:, ::-1, :]
            # patient_mask = patient_mask[:, ::-1, :]

        step = PREDICT_STEP
        CROP_SIZE = CUBE_SIZE
        # CROP_SIZE = 48

        predict_volume_shape_list = [0, 0, 0]
        for dim in range(3):
            dim_indent = 0
            while dim_indent + CROP_SIZE < patient_img.shape[dim]:
                predict_volume_shape_list[dim] += 1
                dim_indent += step

        predict_volume_shape = (predict_volume_shape_list[0], predict_volume_shape_list[1], predict_volume_shape_list[2])
        predict_volume = numpy.zeros(shape=predict_volume_shape, dtype=float)
        print("Predict volume shape: ", predict_volume.shape)
        done_count = 0
        skipped_count = 0
        batch_size = 128
        batch_list = []
        batch_list_coords = []
        patient_predictions_csv = []
        cube_img = None
        annotation_index = 0

        for z in range(0, predict_volume_shape[0]):
            for y in range(0, predict_volume_shape[1]):
                for x in range(0, predict_volume_shape[2]):
                    #if cube_img is None:
                    cube_img = patient_img[z * step:z * step+CROP_SIZE, y * step:y * step + CROP_SIZE, x * step:x * step+CROP_SIZE]
                    cube_mask = patient_mask[z * step:z * step+CROP_SIZE, y * step:y * step + CROP_SIZE, x * step:x * step+CROP_SIZE]

                    if cube_mask.sum() < 2000:
                        skipped_count += 1

                        if CROP_SIZE != CUBE_SIZE:
                            cube_img = helpers.rescale_patient_images2(cube_img, (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))
                            # helpers.save_cube_img("c:/tmp/cube.png", cube_img, 8, 4)
                            # cube_mask = helpers.rescale_patient_images2(cube_mask, (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))
                    
                        img_prep = self.prepare_image_for_net3D(cube_img)
                        batch_list.append(img_prep)
                        batch_list_coords.append((z, y, x))
                        if len(batch_list) % batch_size == 0:
                            batch_data = numpy.vstack(batch_list)
                        
                            p = model.predict(batch_data, batch_size=batch_size)
                            for i in range(len(p[0])):
                                p_z = batch_list_coords[i][0]
                                p_y = batch_list_coords[i][1]
                                p_x = batch_list_coords[i][2]
                                nodule_chance = p[0][i][0]
                                predict_volume[p_z, p_y, p_x] = nodule_chance
                                if nodule_chance > P_TH:
                                    p_z = p_z * step + CROP_SIZE / 2
                                    p_y = p_y * step + CROP_SIZE / 2
                                    p_x = p_x * step + CROP_SIZE / 2

                                    p_z_perc = round(p_z / patient_img.shape[0], 4)
                                    p_y_perc = round(p_y / patient_img.shape[1], 4)
                                    p_x_perc = round(p_x / patient_img.shape[2], 4)
                                    diameter_mm = round(p[1][i][0], 4)
                                    # diameter_perc = round(2 * step / patient_img.shape[2], 4)
                                    diameter_perc = round(2 * step / patient_img.shape[2], 4)
                                    diameter_perc = round(diameter_mm / patient_img.shape[2], 4)
                                    nodule_chance = round(nodule_chance, 4)
                                    patient_predictions_csv_line = [annotation_index, p_x_perc, p_y_perc, p_z_perc, diameter_perc, nodule_chance, diameter_mm]
                                    patient_predictions_csv.append(patient_predictions_csv_line)
                                    all_predictions_csv.append([patient_id] + patient_predictions_csv_line)
                                    annotation_index += 1

                            batch_list = []
                            batch_list_coords = []
                    done_count += 1
                    if done_count % 10000 == 0:
                        print("Done: ", done_count, " skipped:", skipped_count)

        df = pandas.DataFrame(patient_predictions_csv, columns=["anno_index", "coord_x", "coord_y", "coord_z", "diameter", "nodule_chance", "diameter_mm"])
        print("Started Filtering")
        print(all_predictions_csv)
        #print(batch_data)
        self.filter_patient_nodules_predictions(df, patient_id, CROP_SIZE * magnification)
        df.to_csv(csv_target_path, index=False)

        print(predict_volume.mean())
        print("Done in : ", sw.get_elapsed_seconds(), " seconds")


    def find_mhd_file(self,patient_id):
        for subject_no in range(settings.LUNA_SUBSET_START_INDEX, 10):
            src_dir = settings.LUNA16_RAW_SRC_DIR + "subset" + str(subject_no) + "/"
            for src_path in glob.glob(src_dir + "*.mhd"):
                if patient_id in src_path:
                    return src_path
        return None


    def load_lidc_xml(self,xml_path, agreement_threshold=0, only_patient=None, save_nodules=False):
        pos_lines = []
        neg_lines = []
        extended_lines = []
        with open(xml_path, 'r') as xml_file:
            markup = xml_file.read()
        xml = BeautifulSoup(markup, features="xml")
        if xml.LidcReadMessage is None:
            return None, None, None
        patient_id = xml.LidcReadMessage.ResponseHeader.SeriesInstanceUid.text

        if only_patient is not None:
            if only_patient != patient_id:
                return None, None, None

        src_path = self.find_mhd_file(patient_id)
        if src_path is None:
            return None, None, None

        print(patient_id)
        itk_img = SimpleITK.ReadImage(src_path)
        img_array = SimpleITK.GetArrayFromImage(itk_img)
        num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane
        origin = numpy.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
        spacing = numpy.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
        rescale = spacing / settings.TARGET_VOXEL_MM

        reading_sessions = xml.LidcReadMessage.find_all("readingSession")
        for reading_session in reading_sessions:
            # print("Sesion")
            nodules = reading_session.find_all("unblindedReadNodule")
            for nodule in nodules:
                nodule_id = nodule.noduleID.text
                # print("  ", nodule.noduleID)
                rois = nodule.find_all("roi")
                x_min = y_min = z_min = 999999
                x_max = y_max = z_max = -999999
                if len(rois) < 2:
                    continue

                for roi in rois:
                    z_pos = float(roi.imageZposition.text)
                    z_min = min(z_min, z_pos)
                    z_max = max(z_max, z_pos)
                    edge_maps = roi.find_all("edgeMap")
                    for edge_map in edge_maps:
                        x = int(edge_map.xCoord.text)
                        y = int(edge_map.yCoord.text)
                        x_min = min(x_min, x)
                        y_min = min(y_min, y)
                        x_max = max(x_max, x)
                        y_max = max(y_max, y)
                    if x_max == x_min:
                        continue
                    if y_max == y_min:
                        continue

                x_diameter = x_max - x_min
                x_center = x_min + x_diameter / 2
                y_diameter = y_max - y_min
                y_center = y_min + y_diameter / 2
                z_diameter = z_max - z_min
                z_center = z_min + z_diameter / 2
                z_center -= origin[2]
                z_center /= spacing[2]

                x_center_perc = round(x_center / img_array.shape[2], 4)
                y_center_perc = round(y_center / img_array.shape[1], 4)
                z_center_perc = round(z_center / img_array.shape[0], 4)
                diameter = max(x_diameter , y_diameter)
                diameter_perc = round(max(x_diameter / img_array.shape[2], y_diameter / img_array.shape[1]), 4)

                if nodule.characteristics is None:
                    print("!!!!Nodule:", nodule_id, " has no charecteristics")
                    continue
                if nodule.characteristics.malignancy is None:
                    print("!!!!Nodule:", nodule_id, " has no malignacy")
                    continue

                malignacy = nodule.characteristics.malignancy.text
                sphericiy = nodule.characteristics.sphericity.text
                margin = nodule.characteristics.margin.text
                spiculation = nodule.characteristics.spiculation.text
                texture = nodule.characteristics.texture.text
                calcification = nodule.characteristics.calcification.text
                internal_structure = nodule.characteristics.internalStructure.text
                lobulation = nodule.characteristics.lobulation.text
                subtlety = nodule.characteristics.subtlety.text

                line = [nodule_id, x_center_perc, y_center_perc, z_center_perc, diameter_perc, malignacy]
                extended_line = [patient_id, nodule_id, x_center_perc, y_center_perc, z_center_perc, diameter_perc, malignacy, sphericiy, margin, spiculation, texture, calcification, internal_structure, lobulation, subtlety ]
                pos_lines.append(line)
                extended_lines.append(extended_line)

            nonNodules = reading_session.find_all("nonNodule")
            for nonNodule in nonNodules:
                z_center = float(nonNodule.imageZposition.text)
                z_center -= origin[2]
                z_center /= spacing[2]
                x_center = int(nonNodule.locus.xCoord.text)
                y_center = int(nonNodule.locus.yCoord.text)
                nodule_id = nonNodule.nonNoduleID.text
                x_center_perc = round(x_center / img_array.shape[2], 4)
                y_center_perc = round(y_center / img_array.shape[1], 4)
                z_center_perc = round(z_center / img_array.shape[0], 4)
                diameter_perc = round(max(6 / img_array.shape[2], 6 / img_array.shape[1]), 4)
                # print("Non nodule!", z_center)
                line = [nodule_id, x_center_perc, y_center_perc, z_center_perc, diameter_perc, 0]
                neg_lines.append(line)

        if agreement_threshold > 1:
            filtered_lines = []
            for pos_line1 in pos_lines:
                id1 = pos_line1[0]
                x1 = pos_line1[1]
                y1 = pos_line1[2]
                z1 = pos_line1[3]
                d1 = pos_line1[4]
                overlaps = 0
                for pos_line2 in pos_lines:
                    id2 = pos_line2[0]
                    if id1 == id2:
                        continue
                    x2 = pos_line2[1]
                    y2 = pos_line2[2]
                    z2 = pos_line2[3]
                    d2 = pos_line1[4]
                    dist = math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2) + math.pow(z1 - z2, 2))
                    if dist < d1 or dist < d2:
                        overlaps += 1
                if overlaps >= agreement_threshold:
                    filtered_lines.append(pos_line1)
                # else:
                #     print("Too few overlaps")
            pos_lines = filtered_lines

        df_annos = pandas.DataFrame(pos_lines, columns=["anno_index", "coord_x", "coord_y", "coord_z", "diameter", "malscore"])
        df_annos.to_csv(settings.LUNA16_EXTRACTED_IMAGE_DIR + "_labels/" + patient_id + "_annos_pos_lidc.csv", index=False)
        df_neg_annos = pandas.DataFrame(neg_lines, columns=["anno_index", "coord_x", "coord_y", "coord_z", "diameter", "malscore"])
        df_neg_annos.to_csv(settings.LUNA16_EXTRACTED_IMAGE_DIR + "_labels/" + patient_id + "_annos_neg_lidc.csv", index=False)

        # return [patient_id, spacing[0], spacing[1], spacing[2]]
        return pos_lines, neg_lines, extended_lines


    def normalize(self,image):
        MIN_BOUND = -1000.0
        MAX_BOUND = 400.0
        image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        image[image > 1] = 1.
        image[image < 0] = 0.
        return image


    def process_image(self,src_path):
        patient_id = ntpath.basename(src_path).replace(".mhd", "")
        print("Patient: ", patient_id)

        dst_dir = 'Luna\\luna16_extracted_images\\' + patient_id + "\\"
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)

        itk_img = SimpleITK.ReadImage(src_path)
        img_array = SimpleITK.GetArrayFromImage(itk_img)
        print("Img array: ", img_array.shape)

        origin = numpy.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
        print("Origin (x,y,z): ", origin)

        direction = numpy.array(itk_img.GetDirection())      # x,y,z  Origin in world coordinates (mm)
        print("Direction: ", direction)


        spacing = numpy.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
        print("Spacing (x,y,z): ", spacing)
        rescale = spacing / settings.TARGET_VOXEL_MM
        print("Rescale: ", rescale)

        img_array = helpers.rescale_patient_images(img_array, spacing, settings.TARGET_VOXEL_MM)

        img_list = []
        for i in range(img_array.shape[0]):
            img = img_array[i]
            seg_img, mask = helpers.get_segmented_lungs(img.copy())
            img_list.append(seg_img)
            img = self.normalize(img)
            cv2.imwrite(dst_dir + "img_" + str(i).rjust(4, '0') + "_i.png", img * 255)
            cv2.imwrite(dst_dir + "img_" + str(i).rjust(4, '0') + "_m.png", mask * 255)


    def process_pos_annotations_patient(src_path, patient_id):
        df_node = pandas.read_csv("resources/luna16_annotations/annotations.csv")
        dst_dir = settings.LUNA16_EXTRACTED_IMAGE_DIR + "_labels/"
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)
        dst_dir = dst_dir + patient_id + "/"
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)

        itk_img = SimpleITK.ReadImage(src_path)
        img_array = SimpleITK.GetArrayFromImage(itk_img)
        print("Img array: ", img_array.shape)
        df_patient = df_node[df_node["seriesuid"] == patient_id]
        print("Annos: ", len(df_patient))

        num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane
        origin = numpy.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
        print("Origin (x,y,z): ", origin)
        spacing = numpy.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
        print("Spacing (x,y,z): ", spacing)
        rescale = spacing /settings.TARGET_VOXEL_MM
        print("Rescale: ", rescale)

        direction = numpy.array(itk_img.GetDirection())      # x,y,z  Origin in world coordinates (mm)
        print("Direction: ", direction)
        flip_direction_x = False
        flip_direction_y = False
        if round(direction[0]) == -1:
            origin[0] *= -1
            direction[0] = 1
            flip_direction_x = True
            print("Swappint x origin")
        if round(direction[4]) == -1:
            origin[1] *= -1
            direction[4] = 1
            flip_direction_y = True
            print("Swappint y origin")
        print("Direction: ", direction)
        assert abs(sum(direction) - 3) < 0.01

        patient_imgs = helpers.load_patient_images(patient_id, settings.LUNA16_EXTRACTED_IMAGE_DIR, "*_i.png")

        pos_annos = []
        df_patient = df_node[df_node["seriesuid"] == patient_id]
        anno_index = 0
        for index, annotation in df_patient.iterrows():
            node_x = annotation["coordX"]
            if flip_direction_x:
                node_x *= -1
            node_y = annotation["coordY"]
            if flip_direction_y:
                node_y *= -1
            node_z = annotation["coordZ"]
            diam_mm = annotation["diameter_mm"]
            print("Node org (x,y,z,diam): ", (round(node_x, 2), round(node_y, 2), round(node_z, 2), round(diam_mm, 2)))
            center_float = numpy.array([node_x, node_y, node_z])
            center_int = numpy.rint((center_float-origin) / spacing)
            # center_int = numpy.rint((center_float - origin) )
            print("Node tra (x,y,z,diam): ", (center_int[0], center_int[1], center_int[2]))
            # center_int_rescaled = numpy.rint(((center_float-origin) / spacing) * rescale)
            center_float_rescaled = (center_float - origin) / settings.TARGET_VOXEL_MM
            center_float_percent = center_float_rescaled / patient_imgs.swapaxes(0, 2).shape
            # center_int = numpy.rint((center_float - origin) )
            print("Node sca (x,y,z,diam): ", (center_float_rescaled[0], center_float_rescaled[1], center_float_rescaled[2]))
            diameter_pixels = diam_mm / settings.TARGET_VOXEL_MM
            diameter_percent = diameter_pixels / float(patient_imgs.shape[1])

            pos_annos.append([anno_index, round(center_float_percent[0], 4), round(center_float_percent[1], 4), round(center_float_percent[2], 4), round(diameter_percent, 4), 1])
            anno_index += 1

        df_annos = pandas.DataFrame(pos_annos, columns=["anno_index", "coord_x", "coord_y", "coord_z", "diameter", "malscore"])
        df_annos.to_csv(settings.LUNA16_EXTRACTED_IMAGE_DIR + "_labels/" + patient_id + "_annos_pos.csv", index=False)
        return [patient_id, spacing[0], spacing[1], spacing[2]]


    def process_excluded_annotations_patient(src_path, patient_id):
        df_node = pandas.read_csv("resources/luna16_annotations/annotations_excluded.csv")
        dst_dir = settings.LUNA16_EXTRACTED_IMAGE_DIR + "_labels/"
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)
        dst_dir = dst_dir + patient_id + "/"
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)

        # pos_annos_df = pandas.read_csv(TRAIN_DIR + "metadata/" + patient_id + "_annos_pos_lidc.csv")
        pos_annos_df = pandas.read_csv(settings.LUNA16_EXTRACTED_IMAGE_DIR + "_labels/" + patient_id + "_annos_pos.csv")
        pos_annos_manual = None
        manual_path = settings.EXTRA_DATA_DIR + "luna16_manual_labels/" + patient_id + ".csv"
        if os.path.exists(manual_path):
            pos_annos_manual = pandas.read_csv(manual_path)
            dmm = pos_annos_manual["dmm"]  # check

        itk_img = SimpleITK.ReadImage(src_path)
        img_array = SimpleITK.GetArrayFromImage(itk_img)
        print("Img array: ", img_array.shape)
        df_patient = df_node[df_node["seriesuid"] == patient_id]
        print("Annos: ", len(df_patient))

        num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane
        origin = numpy.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
        print("Origin (x,y,z): ", origin)
        spacing = numpy.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
        print("Spacing (x,y,z): ", spacing)
        rescale = spacing / settings.TARGET_VOXEL_MM
        print("Rescale: ", rescale)

        direction = numpy.array(itk_img.GetDirection())      # x,y,z  Origin in world coordinates (mm)
        print("Direction: ", direction)
        flip_direction_x = False
        flip_direction_y = False
        if round(direction[0]) == -1:
            origin[0] *= -1
            direction[0] = 1
            flip_direction_x = True
            print("Swappint x origin")
        if round(direction[4]) == -1:
            origin[1] *= -1
            direction[4] = 1
            flip_direction_y = True
            print("Swappint y origin")
        print("Direction: ", direction)
        assert abs(sum(direction) - 3) < 0.01

        patient_imgs = helpers.load_patient_images(patient_id, settings.LUNA16_EXTRACTED_IMAGE_DIR, "*_i.png")

        neg_annos = []
        df_patient = df_node[df_node["seriesuid"] == patient_id]
        anno_index = 0
        for index, annotation in df_patient.iterrows():
            node_x = annotation["coordX"]
            if flip_direction_x:
                node_x *= -1
            node_y = annotation["coordY"]
            if flip_direction_y:
                node_y *= -1
            node_z = annotation["coordZ"]
            center_float = numpy.array([node_x, node_y, node_z])
            center_int = numpy.rint((center_float-origin) / spacing)
            center_float_rescaled = (center_float - origin) / settings.TARGET_VOXEL_MM
            center_float_percent = center_float_rescaled / patient_imgs.swapaxes(0, 2).shape
            # center_int = numpy.rint((center_float - origin) )
            # print("Node sca (x,y,z,diam): ", (center_float_rescaled[0], center_float_rescaled[1], center_float_rescaled[2]))
            diameter_pixels = 6 / settings.TARGET_VOXEL_MM
            diameter_percent = diameter_pixels / float(patient_imgs.shape[1])

            ok = True

            for index, row in pos_annos_df.iterrows():
                pos_coord_x = row["coord_x"] * patient_imgs.shape[2]
                pos_coord_y = row["coord_y"] * patient_imgs.shape[1]
                pos_coord_z = row["coord_z"] * patient_imgs.shape[0]
                diameter = row["diameter"] * patient_imgs.shape[2]
                print((pos_coord_x, pos_coord_y, pos_coord_z))
                print(center_float_rescaled)
                dist = math.sqrt(math.pow(pos_coord_x - center_float_rescaled[0], 2) + math.pow(pos_coord_y - center_float_rescaled[1], 2) + math.pow(pos_coord_z - center_float_rescaled[2], 2))
                if dist < (diameter + 64):  #  make sure we have a big margin
                    ok = False
                    print("################### Too close", center_float_rescaled)
                    break

            if pos_annos_manual is not None and ok:
                for index, row in pos_annos_manual.iterrows():
                    pos_coord_x = row["x"] * patient_imgs.shape[2]
                    pos_coord_y = row["y"] * patient_imgs.shape[1]
                    pos_coord_z = row["z"] * patient_imgs.shape[0]
                    diameter = row["d"] * patient_imgs.shape[2]
                    print((pos_coord_x, pos_coord_y, pos_coord_z))
                    print(center_float_rescaled)
                    dist = math.sqrt(math.pow(pos_coord_x - center_float_rescaled[0], 2) + math.pow(pos_coord_y - center_float_rescaled[1], 2) + math.pow(pos_coord_z - center_float_rescaled[2], 2))
                    if dist < (diameter + 72):  #  make sure we have a big margin
                        ok = False
                        print("################### Too close", center_float_rescaled)
                        break

            if not ok:
                continue

            neg_annos.append([anno_index, round(center_float_percent[0], 4), round(center_float_percent[1], 4), round(center_float_percent[2], 4), round(diameter_percent, 4), 1])
            anno_index += 1

        df_annos = pandas.DataFrame(neg_annos, columns=["anno_index", "coord_x", "coord_y", "coord_z", "diameter", "malscore"])
        df_annos.to_csv(settings.LUNA16_EXTRACTED_IMAGE_DIR + "_labels/" + patient_id + "_annos_excluded.csv", index=False)
        return [patient_id, spacing[0], spacing[1], spacing[2]]


    def process_luna_candidates_patient(src_path, patient_id):
        dst_dir = settings.LUNA16_EXTRACTED_IMAGE_DIR + "/_labels/"
        img_dir = dst_dir + patient_id + "/"
        df_pos_annos = pandas.read_csv(dst_dir + patient_id + "_annos_pos_lidc.csv")
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)

        pos_annos_manual = None
        manual_path = settings.EXTRA_DATA_DIR + "luna16_manual_labels/" + patient_id + ".csv"
        if os.path.exists(manual_path):
            pos_annos_manual = pandas.read_csv(manual_path)

        itk_img = SimpleITK.ReadImage(src_path)
        img_array = SimpleITK.GetArrayFromImage(itk_img)
        print("Img array: ", img_array.shape)
        print("Pos annos: ", len(df_pos_annos))

        num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane
        origin = numpy.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
        print("Origin (x,y,z): ", origin)
        spacing = numpy.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
        print("Spacing (x,y,z): ", spacing)
        rescale = spacing / settings.TARGET_VOXEL_MM
        print("Rescale: ", rescale)

        direction = numpy.array(itk_img.GetDirection())      # x,y,z  Origin in world coordinates (mm)
        print("Direction: ", direction)
        flip_direction_x = False
        flip_direction_y = False
        if round(direction[0]) == -1:
            origin[0] *= -1
            direction[0] = 1
            flip_direction_x = True
            print("Swappint x origin")
        if round(direction[4]) == -1:
            origin[1] *= -1
            direction[4] = 1
            flip_direction_y = True
            print("Swappint y origin")
        print("Direction: ", direction)
        assert abs(sum(direction) - 3) < 0.01

        src_df = pandas.read_csv("resources/luna16_annotations/" + "candidates_V2.csv")
        src_df = src_df[src_df["seriesuid"] == patient_id]
        src_df = src_df[src_df["class"] == 0]
        patient_imgs = helpers.load_patient_images(patient_id, settings.LUNA16_EXTRACTED_IMAGE_DIR, "*_i.png")
        candidate_list = []

        for df_index, candiate_row in src_df.iterrows():
            node_x = candiate_row["coordX"]
            if flip_direction_x:
                node_x *= -1
            node_y = candiate_row["coordY"]
            if flip_direction_y:
                node_y *= -1
            node_z = candiate_row["coordZ"]
            candidate_diameter = 6
            # print("Node org (x,y,z,diam): ", (round(node_x, 2), round(node_y, 2), round(node_z, 2), round(candidate_diameter, 2)))
            center_float = numpy.array([node_x, node_y, node_z])
            center_int = numpy.rint((center_float-origin) / spacing)
            # center_int = numpy.rint((center_float - origin) )
            # print("Node tra (x,y,z,diam): ", (center_int[0], center_int[1], center_int[2]))
            # center_int_rescaled = numpy.rint(((center_float-origin) / spacing) * rescale)
            center_float_rescaled = (center_float - origin) / settings.TARGET_VOXEL_MM
            center_float_percent = center_float_rescaled / patient_imgs.swapaxes(0, 2).shape
            # center_int = numpy.rint((center_float - origin) )
            # print("Node sca (x,y,z,diam): ", (center_float_rescaled[0], center_float_rescaled[1], center_float_rescaled[2]))
            coord_x = center_float_rescaled[0]
            coord_y = center_float_rescaled[1]
            coord_z = center_float_rescaled[2]

            ok = True

            for index, row in df_pos_annos.iterrows():
                pos_coord_x = row["coord_x"] * patient_imgs.shape[2]
                pos_coord_y = row["coord_y"] * patient_imgs.shape[1]
                pos_coord_z = row["coord_z"] * patient_imgs.shape[0]
                diameter = row["diameter"] * patient_imgs.shape[2]
                dist = math.sqrt(math.pow(pos_coord_x - coord_x, 2) + math.pow(pos_coord_y - coord_y, 2) + math.pow(pos_coord_z - coord_z, 2))
                if dist < (diameter + 64):  #  make sure we have a big margin
                    ok = False
                    print("################### Too close", (coord_x, coord_y, coord_z))
                    break

            if pos_annos_manual is not None and ok:
                for index, row in pos_annos_manual.iterrows():
                    pos_coord_x = row["x"] * patient_imgs.shape[2]
                    pos_coord_y = row["y"] * patient_imgs.shape[1]
                    pos_coord_z = row["z"] * patient_imgs.shape[0]
                    diameter = row["d"] * patient_imgs.shape[2]
                    print((pos_coord_x, pos_coord_y, pos_coord_z))
                    print(center_float_rescaled)
                    dist = math.sqrt(math.pow(pos_coord_x - center_float_rescaled[0], 2) + math.pow(pos_coord_y - center_float_rescaled[1], 2) + math.pow(pos_coord_z - center_float_rescaled[2], 2))
                    if dist < (diameter + 72):  #  make sure we have a big margin
                        ok = False
                        print("################### Too close", center_float_rescaled)
                        break

            if not ok:
                continue

            candidate_list.append([len(candidate_list), round(center_float_percent[0], 4), round(center_float_percent[1], 4), round(center_float_percent[2], 4), round(candidate_diameter / patient_imgs.shape[0], 4), 0])

        df_candidates = pandas.DataFrame(candidate_list, columns=["anno_index", "coord_x", "coord_y", "coord_z", "diameter", "malscore"])
        df_candidates.to_csv(dst_dir + patient_id + "_candidates_luna.csv", index=False)


    def process_auto_candidates_patient(src_path, patient_id, sample_count=1000, candidate_type="white"):
        dst_dir = settings.LUNA16_EXTRACTED_IMAGE_DIR + "/_labels/"
        img_dir = settings.LUNA16_EXTRACTED_IMAGE_DIR + patient_id + "/"
        df_pos_annos = pandas.read_csv(dst_dir + patient_id + "_annos_pos_lidc.csv")

        pos_annos_manual = None
        manual_path = settings.EXTRA_DATA_DIR + "luna16_manual_labels/" + patient_id + ".csv"
        if os.path.exists(manual_path):
            pos_annos_manual = pandas.read_csv(manual_path)

        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)

        itk_img = SimpleITK.ReadImage(src_path)
        img_array = SimpleITK.GetArrayFromImage(itk_img)
        print("Img array: ", img_array.shape)
        print("Pos annos: ", len(df_pos_annos))

        num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane
        origin = numpy.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
        print("Origin (x,y,z): ", origin)
        spacing = numpy.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
        print("Spacing (x,y,z): ", spacing)
        rescale = spacing / settings.TARGET_VOXEL_MM
        print("Rescale: ", rescale)

        if candidate_type == "white":
            wildcard = "*_c.png"
        else:
            wildcard = "*_m.png"

        src_files = glob.glob(img_dir + wildcard)
        src_files.sort()
        src_candidate_maps = [cv2.imread(src_file, cv2.IMREAD_GRAYSCALE) for src_file in src_files]

        candidate_list = []
        tries = 0
        while len(candidate_list) < sample_count and tries < 10000:
            tries += 1
            coord_z = int(numpy.random.normal(len(src_files) / 2, len(src_files) / 6))
            coord_z = max(coord_z, 0)
            coord_z = min(coord_z, len(src_files) - 1)
            candidate_map = src_candidate_maps[coord_z]
            if candidate_type == "edge":
                candidate_map = cv2.Canny(candidate_map.copy(), 100, 200)

            non_zero_indices = numpy.nonzero(candidate_map)
            if len(non_zero_indices[0]) == 0:
                continue
            nonzero_index = random.randint(0, len(non_zero_indices[0]) - 1)
            coord_y = non_zero_indices[0][nonzero_index]
            coord_x = non_zero_indices[1][nonzero_index]
            ok = True
            candidate_diameter = 6
            for index, row in df_pos_annos.iterrows():
                pos_coord_x = row["coord_x"] * src_candidate_maps[0].shape[1]
                pos_coord_y = row["coord_y"] * src_candidate_maps[0].shape[0]
                pos_coord_z = row["coord_z"] * len(src_files)
                diameter = row["diameter"] * src_candidate_maps[0].shape[1]
                dist = math.sqrt(math.pow(pos_coord_x - coord_x, 2) + math.pow(pos_coord_y - coord_y, 2) + math.pow(pos_coord_z - coord_z, 2))
                if dist < (diameter + 48): #  make sure we have a big margin
                    ok = False
                    print("# Too close", (coord_x, coord_y, coord_z))
                    break

            if pos_annos_manual is not None:
                for index, row in pos_annos_manual.iterrows():
                    pos_coord_x = row["x"] * src_candidate_maps[0].shape[1]
                    pos_coord_y = row["y"] * src_candidate_maps[0].shape[0]
                    pos_coord_z = row["z"] * len(src_files)
                    diameter = row["d"] * src_candidate_maps[0].shape[1]
                    # print((pos_coord_x, pos_coord_y, pos_coord_z))
                    # print(center_float_rescaled)
                    dist = math.sqrt(math.pow(pos_coord_x - coord_x, 2) + math.pow(pos_coord_y - coord_y, 2) + math.pow(pos_coord_z - coord_z, 2))
                    if dist < (diameter + 72):  #  make sure we have a big margin
                        ok = False
                        print("#Too close",  (coord_x, coord_y, coord_z))
                        break

            if not ok:
                continue


            perc_x = round(coord_x / src_candidate_maps[coord_z].shape[1], 4)
            perc_y = round(coord_y / src_candidate_maps[coord_z].shape[0], 4)
            perc_z = round(coord_z / len(src_files), 4)
            candidate_list.append([len(candidate_list), perc_x, perc_y, perc_z, round(candidate_diameter / src_candidate_maps[coord_z].shape[1], 4), 0])

        if tries > 9999:
            print("****** WARING!! TOO MANY TRIES ************************************")
        df_candidates = pandas.DataFrame(candidate_list, columns=["anno_index", "coord_x", "coord_y", "coord_z", "diameter", "malscore"])
        df_candidates.to_csv(dst_dir + patient_id + "_candidates_" + candidate_type + ".csv", index=False)


    def process_images(self,path):
        print(path)
        self.process_image(path)


    def process_pos_annotations_patient2():
        candidate_index = 0
        only_patient = "197063290812663596858124411210"
        only_patient = None
        for subject_no in range(settings.LUNA_SUBSET_START_INDEX, 10):
            src_dir = settings.LUNA16_RAW_SRC_DIR + "subset" + str(subject_no) + "/"
            for src_path in glob.glob(src_dir + "*.mhd"):
                if only_patient is not None and only_patient not in src_path:
                    continue
                patient_id = ntpath.basename(src_path).replace(".mhd", "")
                print(candidate_index, " patient: ", patient_id)
                process_pos_annotations_patient(src_path, patient_id)
                candidate_index += 1


    def process_excluded_annotations_patients(only_patient=None):
        candidate_index = 0
        for subject_no in range(settings.LUNA_SUBSET_START_INDEX, 10):
            src_dir = settings.LUNA16_RAW_SRC_DIR + "subset" + str(subject_no) + "/"
            for src_path in glob.glob(src_dir + "*.mhd"):
                if only_patient is not None and only_patient not in src_path:
                    continue
                patient_id = ntpath.basename(src_path).replace(".mhd", "")
                print(candidate_index, " patient: ", patient_id)
                process_excluded_annotations_patient(src_path, patient_id)
                candidate_index += 1


    def process_auto_candidates_patients():
        for subject_no in range(settings.LUNA_SUBSET_START_INDEX, 10):
            src_dir = settings.LUNA16_RAW_SRC_DIR + "subset" + str(subject_no) + "/"
            for patient_index, src_path in enumerate(glob.glob(src_dir + "*.mhd")):
                # if not "100621383016233746780170740405" in src_path:
                #     continue
                patient_id = ntpath.basename(src_path).replace(".mhd", "")
                print("Patient: ", patient_index, " ", patient_id)
                # process_auto_candidates_patient(src_path, patient_id, sample_count=500, candidate_type="white")
                process_auto_candidates_patient(src_path, patient_id, sample_count=200, candidate_type="edge")


    def process_luna_candidates_patients(only_patient_id=None):
        for subject_no in range(settings.LUNA_SUBSET_START_INDEX, 10):
            src_dir = settings.LUNA16_RAW_SRC_DIR + "subset" + str(subject_no) + "/"
            for patient_index, src_path in enumerate(glob.glob(src_dir + "*.mhd")):
                # if not "100621383016233746780170740405" in src_path:
                #     continue
                patient_id = ntpath.basename(src_path).replace(".mhd", "")
                if only_patient_id is not None and patient_id != only_patient_id:
                    continue
                print("Patient: ", patient_index, " ", patient_id)
                process_luna_candidates_patient(src_path, patient_id)


    def process_lidc_annotations(self,only_patient=None, agreement_threshold=0):
        # lines.append(",".join())
        file_no = 0
        pos_count = 0
        neg_count = 0
        all_lines = []
        for anno_dir in [d for d in glob.glob("resources/luna16_annotations/*") if os.path.isdir(d)]:
            xml_paths = glob.glob(anno_dir + "/*.xml")
            for xml_path in xml_paths:
                print(file_no, ": ",  xml_path)
                pos, neg, extended = self.load_lidc_xml(xml_path=xml_path, only_patient=only_patient, agreement_threshold=agreement_threshold)
                if pos is not None:
                    pos_count += len(pos)
                    neg_count += len(neg)
                    print("Pos: ", pos_count, " Neg: ", neg_count)
                    file_no += 1
                    all_lines += extended
                # if file_no > 10:
                #     break

                # extended_line = [nodule_id, x_center_perc, y_center_perc, z_center_perc, diameter_perc, malignacy, sphericiy, margin, spiculation, texture, calcification, internal_structure, lobulation, subtlety ]
        df_annos = pandas.DataFrame(all_lines, columns=["patient_id", "anno_index", "coord_x", "coord_y", "coord_z", "diameter", "malscore", "sphericiy", "margin", "spiculation", "texture", "calcification", "internal_structure", "lobulation", "subtlety"])
        df_annos.to_csv(settings.BASE_DIR + "lidc_annotations.csv", index=False)


    #process_images('/content/1.3.6.1.4.1.14519.5.2.1.6279.6001.219087313261026510628926082729.mhd')
    

if __name__ == "__main__":
    K.set_image_data_format("channels_last")
    CUBE_SIZE = step2_train_nodule_detector.CUBE_SIZE
    MEAN_PIXEL_VALUE = settings.MEAN_PIXEL_VALUE_NODULE
    NEGS_PER_POS = 20
    P_TH = 0.6

    PREDICT_STEP = 12
    USE_DROPOUT = False
    import sys
    app = QtWidgets.QApplication(sys.argv)
    CT_Viewer = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(CT_Viewer)
    CT_Viewer.show()
    sys.exit(app.exec_())
