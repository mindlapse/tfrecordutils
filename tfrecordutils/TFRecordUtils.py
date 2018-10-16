import sys
import glob
import tensorflow as tf

class TFRecordUtils:


    @staticmethod
    def jpgToTFRecord(jpg_glob, tfrecord_file, crop_to=500):

        filenames = tf.constant(glob.glob(jpg_glob))
        dataset = tf.data.Dataset.from_tensor_slices(filenames)

        def preprocess_image(filename):
            image_string = tf.read_file(filename)
            image = tf.image.decode_jpeg(image_string)
            image = tf.image.resize_image_with_crop_or_pad(image, crop_to, crop_to)
            return image, filename

        dataset = dataset.map(preprocess_image)
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        with tf.Session() as sess:
            writer = tf.python_io.TFRecordWriter(tfrecord_file)
            num_files = filenames.shape[0].value
            for i in range(num_files):

                img, filename = sess.run(next_element)
                feature = {
                    'image': tf.train.Feature(int64_list=tf.train.Int64List(value=img.reshape(-1)))
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())

                if (i+1) % 100 == 0:
                    print("Wrote {} of {} images".format(i+1, num_files))
            writer.close()



    @staticmethod
    def get_image_batch_iterator(sess, filename, shape=(500,500,3), resize=None, 
                                 mapping=None, shuffle_buffer=1024, flip_horiz=True,
                                 batch_size=32, prefetch=64, device=None):
        """
        shape: The shape of the data within the TFRecord
        resize: The dimensions to resize to, or None to disable
        mapping: An arbitrary mapping to apply to the image after it was resized.  Must be a function that accepts a numpy array.
        flip_horiz: Enable/disable random horizontal flips for the image
        batch_size: The number of items the iterator will return per iteration
        prefetch:   The number of elements to prefetch (or 0)
        device:     The device to prefetch to
        """
        
        fn_ph = tf.placeholder(tf.string, shape=[None])
        dataset = tf.data.TFRecordDataset(fn_ph)
        features = {'image': tf.FixedLenFeature(shape, tf.int64)}

        def to_numpy(tfrecord):
            example = tf.parse_single_example(tfrecord, features)
            image = tf.cast(example['image'], tf.float32) / 256.
            image = tf.r
            image = tf.image.random_flip_left_right(image, seed=None)
            return mapping(image) if mapping is not None else image
        
        dataset = dataset.map(to_numpy).repeat()
        
        if shuffle_buffer > 0:
            dataset = dataset.shuffle(buffer=shuffle_buffer)
        
        dataset = dataset.batch(batch_size)
        
        if prefetch > 0 and device is not None:
            dataset = dataset.apply(tf.contrib.data.prefetch_to_device(device, buffer_size=prefetch))
        
        iterator = dataset.make_initializable_iterator()
        sess.run(iterator.initializer, feed_dict={fn_ph : [filename]})
        next_element = iterator.get_next()
        return next_element
        



"""
Here's an example of how to use it
"""
if __name__ == '__main__':
    jpg_glob     = './jpg/*.jpg'
    tfrecord_file = 'images.tfrecords'

    # TFRecordUtils.jpgToTFRecord(jpg_glob, tfrecord_file)

    graph = tf.Graph()
    with graph.as_default():
        with tf.Session() as sess:

            iterator = TFRecordUtils.get_image_batch_iterator(sess, tfrecord_file)

            batch = sess.run(iterator)

            print(batch.shape)
