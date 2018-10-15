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
    def get_image_batch_iterator(sess, filename, shape=(500,500,3), mapping=None, batch_size=32):
        fn_ph = tf.placeholder(tf.string, shape=[None])
        dataset = tf.data.TFRecordDataset(fn_ph)
        features = {'image': tf.FixedLenFeature(shape, tf.int64)}

        def to_numpy(tfrecord):
            example = tf.parse_single_example(tfrecord, features)
            image = tf.cast(example['image'], tf.float32) / 256.
            image = tf.image.random_flip_left_right(image, seed=None)
            return mapping(image) if mapping is not None else image

        dataset = dataset.map(to_numpy).repeat().batch(batch_size)
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
