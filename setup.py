import wget, os
from subprocess import call

DATASET_URL = "http://lisaweb.iro.umontreal.ca/transfert/lisa/datasets/mscoco_inpaiting/inpainting.tar.bz2"
DATASET_FILE = "dataset.tar.bz2"
DATASET_FOLDER = "dataset"


def install_dataset():
    try:
        os.makedirs(DATASET_FOLDER)
    except FileExistsError:
        pass


    if not os.path.isfile(os.path.join(DATASET_FOLDER, DATASET_FILE)):
        print("Downloading dataset ...")
        filename = wget.download(DATASET_URL, out=os.path.join(DATASET_FOLDER, DATASET_FILE))

        print("done. File saved at %s"%filename)

    if not os.path.isdir(os.path.join(DATASET_FOLDER, 'inpainting')):
        print("Extracting dataset ...")
        os.chdir(DATASET_FOLDER)
        call(["tar", "xjvf", DATASET_FILE])

    print("Done.")


if __name__ == '__main__':
    install_dataset()