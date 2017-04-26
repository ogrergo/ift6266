import os, sys
import glob
import pickle as pkl
import numpy as np
import PIL.Image as Image
from skimage.transform import resize



MODELS_FOLDER = "models"

def resize_mscoco():
    '''
    function used to create the dataset,
    Resize original MS_COCO Image into 64x64 images
    '''

    ### PATH need to be fixed
    data_path="/dataset/inpainting/train2014"
    save_dir = "/tmp/64_64/train2014/"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    preserve_ratio = True
    image_size = (64, 64)
    #crop_size = (32, 32)

    imgs = glob.glob(data_path+"/*.jpg")


    for i, img_path in enumerate(imgs):
        img = Image.open(img_path)
        print( i, len(imgs), img_path)

        if img.size[0] != image_size[0] or img.size[1] != image_size[1] :
            if not preserve_ratio:
                img = img.resize((image_size), Image.ANTIALIAS)
            else:
                ### Resize based on the smallest dimension
                scale = image_size[0] / float(np.min(img.size))
                new_size = (int(np.floor(scale * img.size[0]))+1, int(np.floor(scale * img.size[1])+1))
                img = img.resize((new_size), Image.ANTIALIAS)

                ### Crop the 64/64 center
                tocrop = np.array(img)
                center = (int(np.floor(tocrop.shape[0] / 2.)), int(np.floor(tocrop.shape[1] / 2.)))
                print(tocrop.shape, center, (center[0]-32,center[0]+32), (center[1]-32,center[1]+32))
                if len(tocrop.shape) == 3:
                    tocrop = tocrop[center[0]-32:center[0]+32, center[1] - 32:center[1]+32, :]
                else:
                    tocrop = tocrop[center[0]-32:center[0]+32, center[1] - 32:center[1]+32]
                img = Image.fromarray(tocrop)

        img.save(save_dir + os.path.basename(img_path))



def load_data(start, n):

    if n <= 0:
        return

    mscoco = "dataset/inpainting"
    split = "train2014"
    caption_path = "dict_key_imgID_value_caps_train_and_valid.pkl"

    data_path = os.path.join(mscoco, split)
    caption_path = os.path.join(mscoco, caption_path)
    with open(caption_path, 'rb') as fd:
        caption_dict = pkl.load(fd)

    print(data_path + "/*.jpg")
    imgs = glob.glob(data_path + "/*.jpg")

    imgs = imgs[start: start + n]
    data = np.empty((n, 3, 64, 64), dtype='uint8')

    for i, img_path in enumerate(imgs):
        __t = np.array(Image.open(img_path))
        if len(__t.shape) == 2:
            __t = np.repeat(__t[None, :, :], axis=0, repeats=3)
        else:
            __t = __t.transpose((2,0,1))
        data[i, :, :, :] = __t

    return data


def get_border(imgs):
    shape = imgs.shape[0:2] + (32, 32)
    result = np.copy(imgs)
    result[:,:,16:48,16:48] = np.zeros(shape)
    return result


def get_center(imgs):
    return imgs[:, :, 16:48, 16:48]



def get_dataset(size_train, size_valid, size_test=100, min=0., max=1.):
    scale = lambda i: (i/255.) * (max - min) + min

    train_imgs = scale(load_data(0, size_train)) if size_train > 0 else None
    valid_imgs = scale(load_data(size_train, size_valid)) if size_valid > 0 else None


    test_imgs = np.concatenate((train_imgs[0:size_test//2],
                                scale(load_data(size_train + size_valid, size_test - size_test//2))), axis=0) if size_test > 0 else None


    return (train_imgs, valid_imgs, test_imgs)



def load_data2(batch_idx, batch_size,
                  ### PATH need to be fixed
                  mscoco="dataset/inpainting", split="train2014", caption_path="dict_key_imgID_value_caps_train_and_valid.pkl"):
    '''
    Show an example of how to read the dataset
    '''

    data_path = os.path.join(mscoco, split)
    caption_path = os.path.join(mscoco, caption_path)
    with open(caption_path, 'rb') as fd:
        caption_dict = pkl.load(fd)

    print(data_path + "/*.jpg")
    imgs = glob.glob(data_path + "/*.jpg")
    batch_imgs = imgs[batch_idx*batch_size:(batch_idx+1)*batch_size]

    input_data = np.empty((batch_size, 64, 64, 3), dtype='uint8')
    target_data = np.empty((batch_size, 32, 32, 3), dtype='uint8')

    for i, img_path in enumerate(batch_imgs):
        img = Image.open(img_path)
        img_array = np.array(img)

        cap_id = os.path.basename(img_path)[:-4]

        ### Get input/target from the images
        center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))
        if len(img_array.shape) == 3:
            input = np.copy(img_array)
            input[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = 0
            target = img_array[center[0]-16:center[0]+16, center[1] - 16:center[1]+16, :]
        else:
            input = np.copy(img_array)
            input = np.repeat(input[:, :, np.newaxis], 3, axis=2)
            input[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = 0
            target = np.repeat(img_array[center[0]-16:center[0]+16, center[1] - 16:center[1]+16, np.newaxis], 3, axis=2)

        input_data[i, :,:,:] = input
        target_data[i, :,:,:] = target
        #Image.fromarray(img_array).show()
        # Image.fromarray(input).show()
        # Image.fromarray(target).show()

    return input_data, target_data

# return dataset for train, valid, test


def save_image(model_name, image_name, img):
    dir = os.path.join('output', model_name)
    try:
        os.makedirs(dir)
    except FileExistsError:
        pass

    Image.fromarray(img).save(os.path.join(dir, str(image_name) + '.bmp'))





if __name__ == '__main__':
    #resize_mscoco()
    train_set = load_data(0, 1000)
    print(train_set)
