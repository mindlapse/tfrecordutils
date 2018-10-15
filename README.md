# TFRecordUtils

Making life easier for working with TFRecord files and TensorFlow's data pipeline API.

## Installation

```
pip install tfrecordutils
```


### HOWTO
### Create a TFRecord file from a directory of images

* crops images by default to 500x500, but you can override this

```
jpg_glob     = './jpg/*.jpg'
tfrecord_file = 'images.tfrecords'

TFRecordUtils.jpgToTFRecord(jpg_glob)
```



### HOWTO
### Get an image batch iterator from a TFRecord file

```
    tfrecord_file = 'images.tfrecords'

    with tf.Session() as sess:

        iterator = TFRecordUtils.get_image_batch_iterator(sess, tfrecord_file)

        batch = sess.run(iterator)

        print(batch.shape)
```
