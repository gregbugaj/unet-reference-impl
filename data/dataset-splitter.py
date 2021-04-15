import shutil
import os
import argparse
import cv2
import numpy as np


def imwrite(path, img):
    try:
        cv2.imwrite(path, img)
    except Exception as ident:
        print(ident)


def split(src_mulitpage_tif, target_dir):
    print("Spliting : %s" % (src_mulitpage_tif))

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    loaded, frames = cv2.imreadmulti(src_mulitpage_tif, [], cv2.IMREAD_ANYCOLOR)
    if not loaded:
        raise Exception('Unable to load source file : %s' % (src_mulitpage_tif))

    print("Total frames : %s" % (len(frames)))

    for idx, frame in enumerate(frames):
        target = os.path.join(target_dir, "%s.png" %(idx))
        print(target)

        img = frame
        print('Original Dimensions : ',img.shape)
        scale_percent = 75 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        frame = resized
        
        print('Resized Dimensions : ',resized.shape)
        imwrite(target, frame)

    # filename = str(uuid.uuid4())
    # imwrite(os.path.join(hash_dir, "%s.tif" % (filename)), snip)
    # cv2.imwrite(hash_dir, segment)
    # showAndDestroy('frame ', snip)


if __name__ == '__main__':
    split('./raw/train-volume.tif', './nerve/train/image')
    split('./raw/train-labels.tif', './nerve/train/mask')
    # split('./raw/train-labels.tif', './ma')
    # extractSnippetsFromCache()
