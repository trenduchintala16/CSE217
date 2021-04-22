import numpy as np
import pandas as pd
import skimage
import skimage.transform
import skimage.io
import warnings
import pathlib
from tensorflow import keras
from tqdm import tqdm
from tensorflow.keras import utils

def mktrte(df,N,sz):
    '''Creates a training or testing dataset of size N that can be used in the NN'''
    X = []
    y = []
    if df.shape[0]<N:
        N = df.shape[0]
    rand_ids = np.random.permutation(df.shape[0])
    for i in range(N):
        r = df.loc[rand_ids[i]]
        l=r["label"]
        im=r["image"]
        im=transform_simple(im,sz)
        X.append(im)
        y.append(l)
    X=np.array(X).astype('float32')
    y=np.array(y)
    y=utils.to_categorical(y,3)
    return X,y

def mkbatch(df,N,sz):
    X = []
    y = []
    for i in range(N):
        im,l=sample(df,sz)
        X.append(im)
        y.append(l)
    X=np.array(X).astype('float32')
    y=np.array(y)
    y=utils.to_categorical(y,3)
    return X,y


def sample(df,sz):
    r=df.sample(n=1)
    l=r["label"].iloc[0]
    im=r["image"].iloc[0]
    im=transform_simple(im,sz)
    return im,l

def generator(df,batch_size,sz):
    while True:
        X,y = mkbatch(df,batch_size,sz)
        yield (X,y)

# Take image and resize to a specified size
def transform_simple(im,sz):
    imr = skimage.transform.resize(im, (sz,sz))
    return imr

def create_user_testdata(path2folder, foldername):
    dataset_directory = pathlib.Path(path2folder)

    # Now check the data
    ddir=dataset_directory/foldername
    cdirs={}
    cdirs.update({ddir/"c0":0,
                  ddir/"c1":1,
                  ddir/"c2":2})

    names = ["rock", "paper", "scissors"]

    for cdir,cdir_class in cdirs.items():
        assert cdir.exists()==1, str(cdir)+' does not exist'
        print("Found directory {} containing class {}".format(cdir,names[cdir_class]))

    imagesize = 500
    dataset1=[]
    for cdir,cn in reversed(list(cdirs.items())):

        for f in tqdm(list(cdir.glob("*"))):
            try:
                im=skimage.io.imread(f)
                h,w=im.shape[0:2] # height, width
                sz=min(h,w)
                im=im[(h//2-sz//2):(h//2+sz//2),(w//2-sz//2):(w//2+sz//2),:] # defines the central square
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    im=skimage.img_as_ubyte(skimage.transform.resize(im,(imagesize,imagesize))) # resize it to 500x500, whatever the original resolution
            except:
                warnings.warn("ignoring "+str(f))
                continue

            dataset1.append({
                "file": f,
                "label": cn,
                "image": im
            })

    print("Done")

    dataset1 = pd.DataFrame(dataset1)
    dataset1["dn"] = dataset1["file"].apply(lambda x: x.parent.parts[-2])
    return dataset1


def load_image(path, target_size=500):
    '''Loads and resizes image located at PATH'''

    # load the image located at PATH
    image = skimage.io.imread(path)

    # ignore possible precision loss warning due to float64 conversion to uint8
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        return skimage.img_as_ubyte(image)


def square_image(image):
    '''
    Crops image into a square using the length of the shortest side
    '''
    # compute image and central square dimensions
    height, width = image.shape[:2]
    square_dimension = min(height, width)

    # compute boundary indices of central square
    square_top = height // 2 + square_dimension // 2
    square_bot = height // 2 - square_dimension // 2
    square_left = width // 2 - square_dimension // 2
    square_right = width // 2 + square_dimension // 2

    # retrieve the content from the central square
    return image[square_bot:square_top, square_left:square_right, :]


def resize_image(image, target_size=500):
    '''
    Resizes given image to have the target dimensions
    '''

    # resize the image to be our desired target size
    return skimage.transform.resize(image, (target_size, target_size))


def save_image(path, image):
    '''
    Writes given IMAGE to PATH
    '''

    skimage.io.imsave(path, image)


def show_image(image):
    '''
    Displays supplied image. Image can be supplied as an skimage array
    or a file path
    '''

    skimage.io.imshow(image)
