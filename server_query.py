# -*- coding: utf-8 -*-
# Author: zhleternity

import os
# os.environ['KERAS_BACKEND']='theano'
from extract_cnn_vgg16_keras import VGGNet
import numpy as np
from PIL import Image
from flask import Flask, request, render_template, url_for, send_from_directory
import json
import base64
from werkzeug import secure_filename
import h5py
import time

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'bmp'])
app = Flask(__name__)

model = VGGNet()
h5f = h5py.File("facefeatureCNN.h5", 'r')
feats = h5f['dataset_1'][:]
imgNames = h5f['dataset_2'][:]
h5f.close()


database_path = 'static/img/'# "F:\\hailing\\database\\retrieve\\oxbuild_images\\"



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def description_search(path):
    queryVec = model.extract_feat(path)
    # print(queryVec.shape)
    scores = np.dot(queryVec, feats.T)
    # print(len(scores))
    maxres = 5
    answers = []
    rank_ID = np.argsort(scores)[::-1]
    # print(len(rank_ID))
    # print(rank_ID)
    # number of top retrieved images to show
    imlist = [database_path + imgNames[index] for index in rank_ID[0:maxres]]
    print("top %d images in order are: " % maxres, imlist)
    des = [scores[id] for id in rank_ID[0:maxres]]
    answers = imlist
    scores = des
    return answers, scores


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/upload/" + file.filename
        img.save(uploaded_img_path)
        tt = time.time()
        answers, scores = description_search(uploaded_img_path)
        print("Using time: {} s".format(time.time() - tt))
        return render_template('index.html',
                               query_path=uploaded_img_path,
                               answers=answers,
                               scores=scores)
    else:
        return render_template('index.html')


if __name__=="__main__":
    # try:
    #     index_data()
    # except Exception as e:
    #     pass
    app.run("127.0.0.1", debug=False)
