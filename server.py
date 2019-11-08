# Agrifix : Detect land conditions through satellite imagery with a single click.

import json
import requests
import logging
import reverse_geocoder
import requests
from requests.auth import HTTPBasicAuth
import glob, os
from subprocess import Popen
import cv2
import numpy as np
import time

from logging.handlers import RotatingFileHandler

from flask import Flask, jsonify, request

app = Flask(__name__, static_url_path='')
usr = 'agrifix'
tkn = 'a_-_-**_-_-9'

@app.route("/", methods=['GET', 'POST'])
def root():
    return app.send_static_file('index.html')

@app.route('/reverse_geocode', methods=['POST'])
def get_reverse_geocode():
    try:
        longitude = request.form.get("longitude")
        latitude = request.form.get("latitude")
    except Exception:
        return jsonify({"response" : "Bad request!"}), 400

    return json.dumps(reverse_geocoder.search((longitude, latitude))), 200

@app.route('/coordinates', methods=['POST'])
def get_images():
    try:
        longitude = request.form.get("longitude")
        latitude = request.form.get("latitude")
    except Exception:
        return jsonify({"response" : "Bad request!"}), 400

    return getRadasat1Images(longitude, latitude), 200

def getRadasat1Images(longitude, latitude):
    radarsat1_api_url = "https://data.eodms-sgdot.nrcan-rncan.gc.ca/api/dhus/v1/products/Radarsat1/search?q=footprint:Intersects((" + latitude + "," + longitude + "))"#" AND sensoroperationalmode:S4"
    response = requests.get(radarsat1_api_url, auth=(usr, tkn)).json()
    localdir=os.path.dirname(os.path.abspath(__file__))
    test = localdir+"/static/images/radarsat1/original/*"
    r = glob.glob(test)
    for i in r:
           os.remove(i)

    response_array = []
    for entry in range(len(response['entry'])):
        image_date = response["entry"][entry]['beginposition']
        linklist=response["entry"][entry]['link']
        for el in range(len(linklist)):
            if linklist[el].get('rel')=='alternative':
                img=requests.get(linklist[el].get('href'), auth=HTTPBasicAuth(usr, tkn), verify=False)
                title=img.url.split("FeatureID=")[1].split("&")[0]+".jpeg"
                image_pair = {"date" : image_date, "image_url" : title}
                if img.status_code == 200:
                    with open(localdir+"/static/images/radarsat1/original/"+title, 'wb') as f:
                        f.write(img.content)
                    #Popen(['cd ./modules/colorization && python3 bw2color_image.py --image ../../static/images/radarsat1/original/' + title + ' --prototxt models/colorization_deploy_v2.prototxt --model models/colorization_release_v2.caffemodel --points models/pts_in_hull.npy'], shell=True)
        response_array.append(image_pair)
    # Sort the images by date.
    response_array.sort(key = lambda x:x['date'])

    caliberImgs(response_array)

    return json.dumps(response_array)

def caliberImgs(response_array):

    localdir=os.path.dirname(os.path.abspath(__file__))
    test = localdir+"/static/images/radarsat1/subset/*"
    r = glob.glob(test)
    for i in r:
           os.remove(i)

    #setup feature extractor
    surf = cv2.xfeatures2d.SURF_create()
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    fstimg = None
    warpedimgs=[]
    try:
        #get first img
        fstimg=cv2.imread(localdir+"/static/images/radarsat1/original/"+response_array[0]['image_url'])
        warpedimgs=[{'img':cv2.resize(fstimg,(613, 685)), 'image_url':response_array[0]['image_url']}]
    except:
        return None
    #iterate through images
    for idx in range(1,len(response_array)):
        try:
            img=cv2.resize(cv2.imread(localdir+"/static/images/radarsat1/original/"+response_array[idx]['image_url']),(613, 685))
        except:
            pass
        #find relationship between the two images
        H=match(surf, flann,warpedimgs[-1]['img'], img )

        if H is None or H.any()==None:
            continue
        #check that the resolution difference is not too high
        if np.linalg.det(H) < 0.1 or np.linalg.det(H) > 10:
            continue
        #check which of new or previous image is lower resolution
        if np.linalg.det(H) < 1:
            H = np.linalg.inv(H)
            #check that perpective difference is not too big
            if abs(H[2][2]) < 0.9 or abs(H[2][2]) > 1.1:
                continue
            for idx in range(len(warpedimgs)):
                #apply new homography to previous imgs
                 warpedimgs[idx]['img'] = cv2.warpPerspective(warpedimgs[idx]['img'], H, (613,685))
        else:
            #check that perpective difference is not too big
            if abs(H[2][2]) < 0.9 or abs(H[2][2]) > 1.1:
                continue
            #apply previous homography to new img
            img = cv2.warpPerspective(img, H, (613,685))
        warpedimgs.append({"img":img, "image_url":response_array[idx]['image_url']})
    #store calibrated images
    for idx in range(len(warpedimgs)):
        cv2.imwrite(localdir+"/static/images/radarsat1/subset/"+warpedimgs[idx]['image_url']+".jpeg", warpedimgs[idx]['img'])


def match(surf, flann, i1, i2):
    #find features in two images
    imageSet1 = getSURFFeatures(surf,i1)
    imageSet2 = getSURFFeatures(surf,i2)
    matches = flann.knnMatch(
        imageSet2['des'],
        imageSet1['des'],
        k=2
        )
    good = []
    #retain good features that are present in both
    for i , (m, n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            good.append((m.trainIdx, m.queryIdx))

    if len(good) > 4:
        pointsCurrent = imageSet2['kp']
        pointsPrevious = imageSet1['kp']

        matchedPointsCurrent = np.float32(
            [pointsCurrent[i].pt for (__, i) in good]
        )
        matchedPointsPrev = np.float32(
            [pointsPrevious[i].pt for (i, __) in good]
            )
        #find relationship between the features of both images
        H, s = cv2.findHomography(matchedPointsCurrent, matchedPointsPrev, cv2.RANSAC, 4)
        return H
    return None

def getSURFFeatures(surf, im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    kp, des = surf.detectAndCompute(gray, None)
    return {'kp':kp, 'des':des}

if __name__ == '__main__':
    handler = RotatingFileHandler('server.log', maxBytes=10000, backupCount=1)
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)
    app.run(host="127.0.0.1",port=5000)
