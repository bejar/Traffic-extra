"""
.. module:: Generate_Dataset

Generate_Dataset
*************

:Description: Generate_Dataset

    Different functions for Traffic dataset generation

:Authors: bejar
    

:Version: 

:Created on: 14/11/2016 12:53 

"""

import glob
import os.path
from collections import Counter
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
from Utilities.Cameras import Cameras_ok
from Utilities.Constants import cameras_path, data_path, dataset_path, process_path
from Utilities.Constants import cameras_path, status_path
from Utilities.DataTram import DataTram
from numpy.random import shuffle
from scipy.ndimage import zoom
from sklearn.decomposition import IncrementalPCA
from Process.CamTram import CamTram
import pickle
import h5py

__author__ = 'bejar'


def get_day_images_data(day, cpatt=None):
    """
    Return a dictionary with all the camera identifiers that exist for all the timestamps of the day
    :param cpatt:
    :return:
    """
    camdic = {}

    if cpatt is not None:
        ldir = sorted(glob.glob(cameras_path + day + '/*' + cpatt + '*.gif'))
    else:
        ldir = sorted(glob.glob(cameras_path + day + '/*.gif'))

    camdic = {}

    for f in sorted(ldir):
        name = f.split('.')[0].split('/')[-1]
        time, place = name.split('-')
        if place in Cameras_ok:
            if int(time) in camdic:
                camdic[int(time)].append(place)
            else:
                camdic[int(time)] = [place]

    return camdic

def get_day_predictions(day):
    """
    Returns all the predictions for a day

    :param day:
    :return:
    """
    ldir = glob.glob(status_path + day + '/*-dadestram.data')
    ldata = []
    for f in sorted(ldir):
        ldata.append(DataTram(f))
    return ldata


def dist_time(time1, time2):
    """
    distance between two hours

    :param time1:
    :param time2:
    :return:
    """
    t1 = (time1 % 100) + (60 * ((time1 // 100) % 100))
    t2 = (time2 % 100) + (60 * ((time2 // 100) % 100))
    return t2 - t1


def generate_classification_dataset_one(day, cpatt=None):
    """
    Generates a dictionary with the dates of the images with lists that contain the camera name and current and predicted
    traffic status using only the nearest prediction in space and time

    :param day:
    :return:
    """
    camdic = get_day_images_data(day, cpatt=cpatt)
    ldata = get_day_predictions(day)
    CTram = CamTram()
    assoc = {}

    for imgtime in sorted(camdic):
        # Look for the status and forecast closer to the image but always in the future
        dmin = None
        dmin2 = None
        vmin2 = 70
        vmin = 60
        for d in ldata:
            diff = dist_time(imgtime, d.date)
            if vmin > np.abs(diff): # Only if it is ahead in time
                if imgtime - d.date > 0:
                    vmin2 = vmin
                    vmin = diff
                    dmin2 = dmin
                    dmin = d
            elif vmin2 > np.abs(diff):
                if diff > 0 and diff != vmin:
                    vmin2 = diff
                    dmin2 = d

        if dmin is not None and dmin2 is not None and vmin < 60 and vmin2 < 60:
            lclass = []
            for img in camdic[imgtime]:
                tram = CTram.ct[img][0]
                # print(imgtime, dmin.dt[tram], img)
                # store for an image of that time the name, closest status, prediction and next status
                lclass.append((img, dmin.dt[tram][0], dmin.dt[tram][1], dmin2.dt[tram][0]))
            assoc[imgtime] = lclass

    return assoc


def generate_classification_dataset_two(day, cpatt=None, mxdelay=60):
    """
    Generates a dictionary with the dates of the images with lists that contain the camera name and current and predicted
    traffic status using the two nearest prediction in space and time

    :param day:
    :param mxdelay: Maximum delay distance between image and status label
    :return:
    """

    camdic = get_day_images_data(day, cpatt=cpatt)
    ldata = get_day_predictions(day)
    assoc = {}
    CTram = CamTram()

    for imgtime in sorted(camdic):
        # Look for the status and forecast closer to the image but always in the future
        dmin = None
        dmin2 = None
        vmin = 60
        vmin2 = 70
        for d in ldata:
            diff = dist_time(imgtime, d.date)
            if vmin > np.abs(diff): # Only if it is ahead in time
                if diff >= 0:
                    vmin2 = vmin
                    vmin = diff
                    dmin2 = dmin
                    dmin = d
            elif vmin2 > np.abs(diff):
                if diff >= 0 and diff != vmin:
                    vmin2 = diff
                    dmin2 = d
        if dmin is not None and dmin2 is not None and vmin < mxdelay and vmin2 < mxdelay:
            print(vmin, vmin2)
            lclass = []
            for img in camdic[imgtime]:
                tram1 = CTram.ct[img][0]
                tram2 = CTram.ct[img][1]
                # store for an image of that time the name, closest status, prediction and next status
                lclass.append((img,
                               max(dmin.dt[tram1][0], dmin.dt[tram2][0]),
                               max(dmin.dt[tram1][1], dmin.dt[tram2][1]),
                               max(dmin2.dt[tram1][0], dmin2.dt[tram2][0])))
            assoc[imgtime] = lclass

    return assoc


def generate_dataset_PCA(ldaysTr, ldaysTs, z_factor, PCA=True, ncomp=100, method='one', cpatt=None, reshape=False):
    """
    Generates a training and test datasets from the days in the parameters
    z_factor is the zoom factor to rescale the images
    :param ldaysTr:
    :param ldaysTs:
    :param z_factor:
    :param PCA:
    :param method:
    :return:

    """
    # -------------------- Train Set ------------------
    ldataTr = []
    llabelsTr = []

    for day in ldaysTr:
        if method == 'one':
            dataset = generate_classification_dataset_one(day, cpatt=cpatt)
        else:
            dataset = generate_classification_dataset_two(day, cpatt=cpatt)
        for t in dataset:
            for cam, l, _, _ in dataset[t]:
                if l != 0 and l != 6:
                    image = mpimg.imread(cameras_path + day + '/' + str(t) + '-' + cam + '.gif')
                    if np.sum(image == 254) < 100000: # This avoids the "not Available data" image
                        del image
                        im = Image.open(cameras_path + day + '/' + str(t) + '-' + cam + '.gif').convert('RGB')
                        data = np.asarray(im)
                        data = data[5:235, 5:315, :].astype('float32')
                        data /= 255.0
                        if z_factor is not None:
                            data = np.dstack((zoom(data[:, :, 0], z_factor), zoom(data[:, :, 1], z_factor),
                                              zoom(data[:, :, 2], z_factor)))
                        if reshape:
                            data = np.reshape(data, (data.shape[0] * data.shape[1] * data.shape[2]))
                        ldataTr.append(data)
                        llabelsTr.append(l)

    # ------------- Test Set ------------------
    ldataTs = []
    llabelsTs = []

    for day in ldaysTs:
        if method == 'one':
            dataset = generate_classification_dataset_one(day, cpatt=cpatt)
        else:
            dataset = generate_classification_dataset_two(day, cpatt=cpatt)
        for t in dataset:
            for cam, l, _, _ in dataset[t]:
                # print(cameras_path + day + '/' + str(t) + '-' + cam + '.gif')
                if l != 0 and l != 6:
                    image = mpimg.imread(cameras_path + day + '/' + str(t) + '-' + cam + '.gif')
                    if np.sum(image == 254) < 100000: # This avoids the "not Available data" image
                        del image
                        im = Image.open(cameras_path + day + '/' + str(t) + '-' + cam + '.gif').convert('RGB')
                        data = np.asarray(im)
                        data = data[5:235, 5:315, :].astype('float32')
                        data /= 255.0
                        if z_factor is not None:
                            data = np.dstack((zoom(data[:, :, 0], z_factor), zoom(data[:, :, 1], z_factor),
                                              zoom(data[:, :, 2], z_factor)))
                        if reshape:
                            data = np.reshape(data, (data.shape[0] * data.shape[1] * data.shape[2]))
                        ldataTs.append(data)
                        llabelsTs.append(l)

    if reshape or z_factor is not None:
        del data

    print(Counter(llabelsTr))
    print(Counter(llabelsTs))

    X_train = np.array(ldataTr)
    del ldataTr
    X_test = np.array(ldataTs)
    del ldataTs

    if PCA:
        pca = IncrementalPCA(n_components=ncomp)
        pca.fit(X_train)
        print(np.sum(pca.explained_variance_ratio_[:ncomp]))
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)

    y_train = llabelsTr
    y_test = llabelsTs
    print(X_train.shape, X_test.shape)

    return X_train, y_train, X_test, y_test

def generate_dataset(ldaysTr, z_factor, method='one', cpatt=None):
    """
    Generates a training and test datasets from the days in the parameters
    z_factor is the zoom factor to rescale the images
    :param ldaysTr:
    :param ldaysTs:
    :param z_factor:
    :param PCA:
    :param method:
    :return:

    """
    ldataTr = []
    llabelsTr = []

    for day in ldaysTr:
        if method == 'one':
            dataset = generate_classification_dataset_one(day, cpatt=cpatt)
        else:
            dataset = generate_classification_dataset_two(day, cpatt=cpatt)
        for t in dataset:
            for cam, l, _, _ in dataset[t]:
                if l != 0 and l != 6:
                    image = mpimg.imread(cameras_path + day + '/' + str(t) + '-' + cam + '.gif')
                    if np.sum(image == 254) < 100000: # This avoids the "not Available data" image
                        del image
                        im = Image.open(cameras_path + day + '/' + str(t) + '-' + cam + '.gif').convert('RGB')
                        data = np.asarray(im)
                        data = data[5:235, 5:315, :].astype('float32')
                        data /= 255.0
                        if z_factor is not None:
                            data = np.dstack((zoom(data[:, :, 0], z_factor), zoom(data[:, :, 1], z_factor),
                                              zoom(data[:, :, 2], z_factor)))

                        ldataTr.append(data)
                        llabelsTr.append(l)


    print(Counter(llabelsTr))

    X_train = np.array(ldataTr)
    y_train = llabelsTr

    return X_train, y_train


def save_daily_dataset(ldaysTr, ldaysTs, z_factor, PCA=True, ncomp=100, method='one', cpatt=None, reshape=False):
    """
    Computes the PCA transformation using the days in ldaysTr
    Generates and save datasets from the days in the ldaysTs
    z_factor is the zoom factor to rescale the images
    :param trdays:
    :param tsdays:
    :return:
    """

    # -------------------- Train Set ------------------
    ldataTr = []
    llabelsTr = []

    for day in ldaysTr:
        if method == 'one':
            dataset = generate_classification_dataset_one(day, cpatt=cpatt)
        else:
            dataset = generate_classification_dataset_two(day, cpatt=cpatt)
        for t in dataset:
            for cam, l, _, _ in dataset[t]:
                if l != 0 and l != 6:
                    image = mpimg.imread(cameras_path + day + '/' + str(t) + '-' + cam + '.gif')
                    if np.sum(image == 254) < 100000:
                        del image
                        im = Image.open(cameras_path + day + '/' + str(t) + '-' + cam + '.gif').convert('RGB')
                        data = np.asarray(im)
                        data = data[5:235, 5:315, :].astype('float32')
                        data /= 255.0
                        if z_factor is not None:
                            data = np.dstack((zoom(data[:, :, 0], z_factor), zoom(data[:, :, 1], z_factor),
                                              zoom(data[:, :, 2], z_factor)))
                        if reshape:
                            data = np.reshape(data, (data.shape[0] * data.shape[1] * data.shape[2]))

                        ldataTr.append(data)
                        llabelsTr.append(l)

    print(Counter(llabelsTr))
    X_train = np.array(ldataTr)
    pca = IncrementalPCA(n_components=ncomp)
    pca.fit(X_train)
    print(np.sum(pca.explained_variance_ratio_[:ncomp]))
    del X_train

    # ------------- Test Set ------------------
    for day in ldaysTs:
        ldataTs = []
        llabelsTs = []
        if method == 'one':
            dataset = generate_classification_dataset_one(day, cpatt=cpatt)
        else:
            dataset = generate_classification_dataset_two(day, cpatt=cpatt)
        for t in dataset:
            for cam, l, _, _ in dataset[t]:
                # print(cameras_path + day + '/' + str(t) + '-' + cam + '.gif')
                if l != 0 and l != 6:
                    image = mpimg.imread(cameras_path + day + '/' + str(t) + '-' + cam + '.gif')
                    if np.sum(image == 254) < 100000:
                        del image
                        im = Image.open(cameras_path + day + '/' + str(t) + '-' + cam + '.gif').convert('RGB')
                        data = np.asarray(im)
                        data = data[5:235, 5:315, :].astype('float32')
                        data /= 255.0
                        if z_factor is not None:
                            data = np.dstack((zoom(data[:, :, 0], z_factor), zoom(data[:, :, 1], z_factor),
                                              zoom(data[:, :, 2], z_factor)))
                        if reshape:
                            data = np.reshape(data, (data.shape[0] * data.shape[1] * data.shape[2]))
                        ldataTs.append(data)
                        llabelsTs.append(l)
        X_test = pca.transform(np.array(ldataTs))
        y_test = llabelsTs
        print(Counter(llabelsTs))
        np.save(dataset_path + 'data-D%s-Z%0.2f-C%d.npy' % (day, z_factor, ncomp), X_test)
        np.save(dataset_path + 'labels-D%s-Z%0.2f-C%d.npy' % (day, z_factor, ncomp), np.array(y_test))


def generate_rebalanced_dataset(ldaysTr, ndays, z_factor, PCA=True, ncomp=100):
    """
    Generates a training dataset with a rebalance of the classes using a specific number of days of
    the input files for the training dataset

    :param ldaysTr:
    :param z_factor:
    :param PCA:
    :param ncomp:
    :return:
    """
    ldata = []
    y_train = []

    for cl, nd in ndays:
        for i in range(nd):
            day = ldaysTr[i]
            data = np.load(dataset_path + 'data-D%s-Z%0.2f-C%d.npy' % (day, z_factor, ncomp))
            labels = np.load(dataset_path + 'labels-D%s-Z%0.2f-C%d.npy' % (day, z_factor, ncomp))
            ldata.append(data[labels==cl,:])
            y_train.extend(labels[labels==cl])
    X_train = np.concatenate(ldata)
    print(X_train.shape)
    print(Counter(y_train))
    np.save(dataset_path + 'data-RB-Z%0.2f-C%d.npy' % (z_factor, ncomp), X_train)
    np.save(dataset_path + 'labels-RB-Z%0.2f-C%d.npy' % (z_factor, ncomp), np.array(y_train))


def generate_data_day(day, z_factor, method='two', mxdelay=60, log=False):
    """
    Generates a raw dataset for a day with a zoom factor (data and labels)
    :param z_factor:
    :return:
    """
    ldata = []
    llabels = []
    limages = []
    if method == 'one':
        dataset = generate_classification_dataset_one(day)
    else:
        dataset = generate_classification_dataset_two(day, mxdelay=mxdelay)
    for t in dataset:
        for cam, l, _, _ in dataset[t]:
            if l != 0 and l != 6:
                if log:
                    print(cameras_path + day + '/' + str(t) + '-' + cam + '.gif')
                image = mpimg.imread(cameras_path + day + '/' + str(t) + '-' + cam + '.gif')
                if np.sum(image == 254) < 100000:
                    del image
                    im = Image.open(cameras_path + day + '/' + str(t) + '-' + cam + '.gif').convert('RGB')
                    data = np.asarray(im)
                    data = data[5:235, 5:315, :].astype('float32')
                    data /= 255.0
                    if z_factor is not None:
                        data = np.dstack((zoom(data[:, :, 0], z_factor), zoom(data[:, :, 1], z_factor),
                                          zoom(data[:, :, 2], z_factor)))
                    ldata.append(data)
                    llabels.append(l)
                    limages.append(day + '/' + str(t) + '-' + cam)

    X_train = np.array(ldata)
    X_train = X_train.transpose((0,3,1,2)) # Theano ordering
    llabels = [i - 1 for i in llabels]  # change labels from 1-5 to 0-4
    np.save(dataset_path + 'data-D%s-Z%0.2f.npy' % (day, z_factor), X_train)
    np.save(dataset_path + 'labels-D%s-Z%0.2f.npy' % (day, z_factor), np.array(llabels))
    output = open(dataset_path + 'images-D%s-Z%0.2f.pkl' % (day, z_factor), 'wb')
    pickle.dump(limages, output)
    output.close()


def generate_splitted_data_day(day, z_factor, method='two', log=False):
    """
    Generates a raw dataset for a day with a zoom factor splitted in as many files as classes
    :param z_factor:
    :return:
    """
    ldata = []
    llabels = []
    if method == 'one':
        dataset = generate_classification_dataset_one(day)
    else:
        dataset = generate_classification_dataset_two(day)
    for t in dataset:
        for cam, l, _, _ in dataset[t]:
            if l != 0 and l != 6:
                if log:
                    print(cameras_path + day + '/' + str(t) + '-' + cam + '.gif')
                image = mpimg.imread(cameras_path + day + '/' + str(t) + '-' + cam + '.gif')
                if np.sum(image == 254) < 100000:
                    del image
                    im = Image.open(cameras_path + day + '/' + str(t) + '-' + cam + '.gif').convert('RGB')

                    # if l == 1:
                    #     fig = plt.figure()
                    #     fig.set_figwidth(30)
                    #     fig.set_figheight(30)
                    #     sp1 = fig.add_subplot(1, 1, 1)
                    #     sp1.imshow(im)
                    #     plt.title(str(t) + ' ' + cam)
                    #     plt.show()
                    #     plt.close()

                    data = np.asarray(im)
                    data = data[5:235, 5:315, :].astype('float32')
                    data /= 255.0
                    if z_factor is not None:
                        data = np.dstack((zoom(data[:, :, 0], z_factor), zoom(data[:, :, 1], z_factor),
                                          zoom(data[:, :, 2], z_factor)))

                    ldata.append(data)
                    llabels.append(l)

    llabels = np.array(llabels) -1  # labels in range [0,max classes]
    data = np.array(ldata)
    for l in np.unique(llabels):
        sel = llabels == l
        np.save(process_path + 'data-D%s-Z%0.2f-L%d.npy' % (day, z_factor, l), data[sel])


def generate_rebalanced_data_day(day, z_factor, pclasses):
    """
    Generates a rebalanced dataset using the probability of the examples indicated in the parameter pclasses

    :param day:
    :param z_factor:
    :param nclasses:
    :return:
    """
    ddata = {}
    for c in pclasses:
        if os.path.exists(process_path + 'data-D%s-Z%0.2f-L%d.npy' % (day, z_factor, c)):
            ddata[c] = np.load(process_path + 'data-D%s-Z%0.2f-L%d.npy' % (day, z_factor, c))

    ldata = []
    llabels = []
    for c in pclasses:
        if c in ddata:
            nex = np.array(range(ddata[c].shape[0]))
            shuffle(nex)
            nsel = int(ddata[c].shape[0] * pclasses[c])
            sel = nex[0:nsel]
            ldata.append(ddata[c][sel])
            llabels.append(np.zeros(nsel)+c)

    np.save(dataset_path + 'rdata-D%s-Z%0.2f.npy' % (day, z_factor), np.concatenate(ldata))
    np.save(dataset_path + 'rlabels-D%s-Z%0.2f.npy' % (day, z_factor), np.concatenate(llabels))


def load_generated_dataset(datapath, ldaysTr, z_factor):
    """
    Load the already generated datasets

    :param ldaysTr:
    :param ldaysTs:
    :param z_factor:
    :return:
    """
    ldata = []
    y_train = []
    for day in ldaysTr:
        data = np.load(datapath + 'data-D%s-Z%0.2f.npy' % (day, z_factor))
        ldata.append(data)
        y_train.extend(np.load(datapath + 'labels-D%s-Z%0.2f.npy' % (day, z_factor)))
    X_train = np.concatenate(ldata)

    return X_train, y_train

def list_days_generator(year, month, iday, fday):
    """
    Generates a list of days
    :param year:
    :param month:
    :param iday:
    :param fday:
    :return:
    """
    ldays = []
    for v in range(iday, fday+1):
        ldays.append("%d%d%02d" % (year, month, v))
    return ldays


def info_dataset(ldaysTr, z_factor, reb=False):
    """
    Prints counts of the labels of the dataset

    :param ldaysTr:
    :param z_factor:
    :return:
    """

    y_train = []
    fname = 'labels'
    if reb:
        fname= 'r' + fname
    for day in ldaysTr:
        data = np.load(dataset_path + fname + '-D%s-Z%0.2f.npy' % (day, z_factor))
        print(day, Counter(data))
        y_train.extend(data)
    print('TOTAL=', Counter(list(y_train)))


# --------------------------------------------------------------------------------------
# New functions for generating the datasets

def generate_image_labels(day, mxdelay=30, onlyfuture=True):
    """
    Generates a dictionary with the dates of the images with lists that contain the camera name and current
    traffic status using the two nearest prediction in space

    :param day:
    :param mxdelay: Maximum delay distance between image and status label
    :return:
    """

    camdic = get_day_images_data(day)
    ldata = get_day_predictions(day)
    assoc = {}
    CTram = CamTram()

    for imgtime in sorted(camdic):
        # Look for the status and forecast closer to the image but always in the future
        dmin = None
        vmin = 100

        # Find the closest prediction in time for the day
        for d in ldata:
            diff = dist_time(imgtime, d.date)
            if vmin > np.abs(diff): # Only if it is ahead in time
                if onlyfuture:
                    if diff >= 0:
                        vmin = np.abs(diff)
                        dmin = d
                else:
                    vmin = np.abs(diff)
                    dmin = d




        if dmin is not None and vmin < mxdelay:
            # print vmin, imgtime, dmin.date
            lclass = []
            for img in camdic[imgtime]:
                # Two closest positions to the camera
                tram1 = CTram.ct[img][0]
                tram2 = CTram.ct[img][1]

                # store for an image of that time the name, worst status from the two closest positions
                lclass.append((img, max(dmin.dt[tram1][0], dmin.dt[tram2][0])))
            assoc[imgtime] = lclass

    return assoc


def generate_labeled_dataset_day(day, z_factor, mxdelay=60, onlyfuture=True, log=False, imgordering='th'):
    """
    Generates a raw dataset for a day with a zoom factor (data and labels)
    :param z_factor:
    :return:
    """
    ldata = []
    llabels = []
    limages = []
    dataset = generate_image_labels(day, mxdelay=mxdelay, onlyfuture=onlyfuture)
    for t in dataset:
        for cam, l in dataset[t]:
            if l != 0 and l != 6:
                if log:
                    print(cameras_path + day + '/' + str(t) + '-' + cam + '.gif')
                image = mpimg.imread(cameras_path + day + '/' + str(t) + '-' + cam + '.gif')
                if np.sum(image == 254) < 100000:
                    del image
                    im = Image.open(cameras_path + day + '/' + str(t) + '-' + cam + '.gif').convert('RGB')
                    data = np.asarray(im)
                    data = data[5:235, 5:315, :].astype('float32')
                    data /= 255.0
                    if z_factor is not None:
                        data = np.dstack((zoom(data[:, :, 0], z_factor), zoom(data[:, :, 1], z_factor),
                                          zoom(data[:, :, 2], z_factor)))
                    ldata.append(data)
                    llabels.append(l)
                    limages.append(day + '/' + str(t) + '-' + cam)

    X_train = np.array(ldata)
    if imgordering == 'th':
        X_train = X_train.transpose((0,3,1,2)) # Theano image ordering
    print(X_train.shape)

    llabels = [i - 1 for i in llabels]  # change labels from 1-5 to 0-4
    print(Counter(llabels))
    np.save(process_path + 'data-D%s-Z%0.2f.npy' % (day, z_factor), X_train)
    np.save(process_path + 'labels-D%s-Z%0.2f.npy' % (day, z_factor), np.array(llabels))
    output = open(process_path + 'images-D%s-Z%0.2f.pkl' % (day, z_factor), 'wb')
    pickle.dump(limages, output)
    output.close()


def load_generated_day(datapath, day, z_factor):
    """
    Load the already generated datasets

    :param ldaysTr:
    :param ldaysTs:
    :param z_factor:
    :return:
    """

    X_train = np.load(datapath + 'data-D%s-Z%0.2f.npy' % (day, z_factor))
    y_train = np.load(datapath + 'labels-D%s-Z%0.2f.npy' % (day, z_factor))
    output = open(datapath + 'images-D%s-Z%0.2f.pkl' % (day, z_factor), 'rb')
    img_path = pickle.load(output)
    output.close()

    return X_train, y_train, img_path



def chunkify(lchunks, size):
    """
    Returns the saving list for the data with chunks of size = size
    :param lchunks:
    :param size:
    :return:
    """

    accum = 0
    csize = size
    i = 0
    quant = lchunks[0]
    lcut = []
    lpos = []

    while i < len(lchunks):
        if accum + quant <= size:
            accum += quant
            lpos.append((i, quant))
            i += 1
            if i < len(lchunks):
                quant = lchunks[i]
            else:
                if accum == size:
                    lcut.append(lpos)
        else:
            lpos.append((i, size - accum))
            lcut.append(lpos)
            lpos = []
            quant = quant - (size - accum)
            accum = 0
            csize += size

    return lcut




def generate_training_dataset(datapath, ldays, chunk=1024, z_factor=0.25):
    """
    Generates an hdf5 file with blocks of data for training
    :param ldays:
    :param zfactor:
    :return:
    """

    nlabels = []
    for i, day in enumerate(ldays):
        labels = np.load(datapath + 'labels-D%s-Z%0.2f.npy' % (day, z_factor))
        nlabels.append(len(labels))

    lsave = chunkify(nlabels, chunk)

    sfile = h5py.File(datapath + '/train-Z%0.2f.hdf5'% z_factor, 'w')


    prev = {}
    for nchunk, save in enumerate(lsave):

        curr = {}
        for nday, nex in save:
            curr[ldays[nday]] = [load_generated_day(datapath, ldays[nday], z_factor), nex, 0]
            if ldays[nday] in prev:
                curr[ldays[nday]][2] += prev[ldays[nday]][1]

        X_train = []
        y_train = []
        imgpath = []
        for day in curr:

            indi = int(curr[day][2])
            indf = int(curr[day][2] + curr[day][1])

            X_train.append(curr[day][0][0][indi:indf])
            y_train.extend(curr[day][0][1][indi:indf])
            imgpath.extend(curr[day][0][2][indi:indf])

        X_train = np.concatenate(X_train)
        y_train = np.array(y_train)
        imgpath = [n.encode("ascii", "ignore") for n in imgpath]
        prev = curr

        namechunk = 'chunk%03d' % nchunk
        sfile.require_dataset(namechunk + '/' + 'data', X_train.shape, dtype='f',
                              data=X_train, compression='gzip')

        sfile.require_dataset(namechunk + '/' + 'labels', y_train.shape, dtype='f',
                              data=y_train, compression='gzip')

        sfile.require_dataset(namechunk + '/' + 'imgpath', (len(imgpath),1), dtype='S100',
                              data=imgpath, compression='gzip')
        sfile.flush()

    sfile.close()

if __name__ == '__main__':

    days = list_days_generator(2016, 11, 1, 30) + list_days_generator(2016, 12, 1, 2)
    # days = list_days_generator(2016, 11, 1, 5)
    z_factor = 0.25

    # for day in days:
    #     generate_data_day(day, z_factor, method='two', mxdelay=60)
    #
    # for day in days:
    #     generate_splitted_data_day(day, z_factor)

    # for day in days:
    #     print(day)
    #     generate_rebalanced_data_day(day, z_factor, {0:0.4, 1:0.5, 2:1, 3:1, 4:1})
    #
    # info_dataset(days, z_factor, reb=False)

    # data, labels = load_generated_dataset(dataset_path, days, z_factor)
    #
    # print(data.shape)
    # for day in days:
    #     generate_labeled_dataset_day(day, z_factor, mxdelay=15, onlyfuture=False, imgordering='th')


    generate_training_dataset(process_path, days, z_factor=z_factor)

