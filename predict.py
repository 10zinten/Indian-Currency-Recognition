import argparse
import logging
import os
import random
import numpy as np
import tensorflow as tf

from model.utils import Params
from model.model_fn import build_model

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/test',
                    help="Experiment directory containing params.json")
parser.add_argument('--image',
                    help="path to image to predict")
parser.add_argument('--restore_from', default='best_weights',
                    help="Subdirecitory of model dir or file containing the weights")


if __name__ == "__main__":

    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    params = Params(json_path)

    assert os.path.isfile(args.image), "Image {} not found".format(args.iamge)

    image_string = tf.read_file(args.image)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image_decoded, tf.float32)
    resized_image = tf.image.resize_images(image, [params.image_size, params.image_size])
    image = tf.clip_by_value(resized_image, 0.0, 0.1)
    image = tf.expand_dims(image, 0)

    inputs = {"images": image}

    print(image.get_shape().as_list())
    
    # Building model
    with tf.variable_scope('model'):
        logits = build_model(True, inputs, params) # logits shape: (1, 6)
   
    predictions = tf.argmax(logits, 1)             # min max in col=1
    probs = tf.nn.softmax(logits=logits)            
    
    # list all the variables of graphs
    # for var in tf.all_variables():
    #    print(var)

    # from tensorflow.contrib.framework.python.framework import checkpoint_utils
    # var_list = checkpoint_utils.list_variables(os.path.join(args.model_dir, args.restore_from))
    # for var in var_list:
    #    print(var)


    # Initialize tf.Saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Initialize the lookup table
        sess.run(tf.global_variables_initializer())
        
        # Reload weights from the weights subdirectory
        save_path = os.path.join(args.model_dir, args.restore_from)
        if os.path.isdir(save_path):
            print("Best weight found")
            save_path = tf.train.latest_checkpoint(save_path)
        saver.restore(sess, save_path)
        
        # Get predicted labels and probs distribution
        pred, probs = sess.run([predictions, probs])

    print(pred, np.sum(probs))
