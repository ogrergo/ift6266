# Predicting the center of MSCOCO images given the border and the captions

This code is a model written in tensorflow to predict the *32x32* center of a *64x64* image given the border and some captions.



## Installation

    pip install -r requirements.txt

## Dataset

I have used the MSCOCO dataset and preprocessed the data into `tf.Examples` files. You can get my preprocessed dataset [here](https://s3.amazonaws.com/akiaj36ibbq3myh2imfa-dump/dataset_ift6366_images_captions.tar.gz?response-content-disposition=attachment&X-Amz-Security-Token=AgoGb3JpZ2luEHsaCXVzLWVhc3QtMSKAAhGD6RDJAAhz7%2FtsV01ypbwtG8eA66maaHpDjf6cULVZ8r5TUtkcsaAJdX7elYE9Bmwc4XTKBwnDGsgP8mhBuoVpzk%2FsiXQ9Wp4yj665hHaJvQEWatIKW5srQug9kNC0OrYA0xlE%2BvU5PnAlDESudDi2gneLuwOO0SMeC4JjilZdseFHXNgi7%2BWNHTd5cwpaZqsWBQNURsbSOeYtWhjjXFw9e5oeLUe5TtJTPMsmy1qkGCqRuTx9YwrcxvanZr8dL%2FNS%2FSV0Ef6zcTh42mV1fEpYYKKGO8RIHGZThRNUDcddzwixHqXgtEI1Y64vWQlI9dPmJh7BaEsTat6KZaVWhsAqpgIIgP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAAGgw0NDY3MjE5MDY3MTYiDI34tr%2BUVxmSHGdP2ir6AVAy3dnzGIem6koy9IP49IlxLVhzYvf73oK77KJlwwYhD%2FAFosbVaxzAxEJ69wwATwxmv6Av14NiCx8MUT4vBNOC%2Bo%2BcSXnV3wN7FTkNuAK9%2BnWP6fwXt%2BlDf9L9Zg4V9NcFWuEkt4j5TU8x5jjlCdQOg00bsetw2Uz9kDWITBolXTA7FHJmhUW7odsEVBGqVD9EpxgpAOvRn1rx%2F5Qh1pIKkc3sNlk2DLnz%2BQZ7bvhDe8wh6jX%2Bfc8hgyvYHU3CRznSIpFAXFrs0oOQ4xYhMAd%2FdMaJ%2B%2F2Xm8PigPjOYo2fzxf0EOO%2B8vqn8NmA1gpOX0gdOt6NIu31SWEwgb%2BXyAU%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20170430T225000Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAIV5OPGTFQQVZUKZA%2F20170430%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=7e7369d4efa5dbf4bd4ac3dcd9ad6180f1feb0dba8a9de412df48fc57f1f038b).

    mkdir dataset
    cd dataset
    tar xvf dataset_ift6366_images_captions.tar.gz .

To run the model:

    python3 main.py

You can start a tensorboard instance to monitor the training:

    tensorboard --logdir=logs

## architecture
The architecture of the generator is resumed in this schema:
![generator](https://raw.githubusercontent.com/ogrergo/ift6266/master/docs/static_files/Archi gen.jpg "archi generator")
