"""Evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import numpy as np
import scipy.misc
import os

from datasets import dataset_split
from nets import nets_factory
from preprocessing import preprocessing_factory
from tensorflow.contrib.tensorboard.plugins import projector

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
	'batch_size', 100, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
	'max_num_batches', None,
	'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
	'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
	'checkpoint_path', '/tmp/tfmodel/',
	'The directory where the model was written to or an absolute path to a '
	'checkpoint file.')

tf.app.flags.DEFINE_string(
	'vis_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
	'num_preprocessing_threads', 4,
	'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
	'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
	'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
	'labels_offset', 0,
	'An offset for the labels in the dataset. This flag is primarily used to '
	'evaluate the VGG and ResNet architectures which do not use a background '
	'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
	'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
	'preprocessing_name', None, 'The name of the preprocessing to use. If left '
	'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
	'moving_average_decay', None,
	'The decay to use for the moving average.'
	'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
	'eval_image_size', None, 'Eval image size')

FLAGS = tf.app.flags.FLAGS

def main(_):
	if not FLAGS.dataset_dir:
		raise ValueError('You must supply the dataset directory with --dataset_dir')
	FLAGS.vis_dir = FLAGS.vis_dir + FLAGS.model_name + '/'
	FLAGS.checkpoint_path = FLAGS.checkpoint_path + FLAGS.model_name + '/training/'
	compute_saliency = False
	max_saliency = 200
	###################################
	# Dictionnary for label inference #
	###################################       
	keys_9 = []
	vals_9 = []
	with open(FLAGS.dataset_dir + "labels_to_labels_9.txt") as f:
		for line in f:
			 (key, val) = line.split(":")
			 keys_9.append(int(key))
			 vals_9.append(int(val))
                
	if FLAGS.model_name == "inception_v4":
		embedding_name = 'PreLogitsFlatten'
		embedding_dim = 1536
	elif FLAGS.model_name == "inception_v3":
		embedding_name = 'PreLogits'
		embedding_dim = 2048
        
	tf.logging.set_verbosity(tf.logging.INFO)

	tf.reset_default_graph()
	graph = tf.Graph()
	with graph.as_default():
		tf_global_step = slim.get_or_create_global_step()

		label_dict_9 = tf.contrib.lookup.HashTable(
			tf.contrib.lookup.KeyValueTensorInitializer(
				keys_9, vals_9, key_dtype=tf.int64, value_dtype=tf.int64), -1)

		step = tf.Variable(0, dtype=tf.int32,
			collections=[tf.GraphKeys.LOCAL_VARIABLES], trainable=False)

		######################
		# Select the dataset #
		######################
		dataset = dataset_split.get_split(
			FLAGS.dataset_split_name, FLAGS.dataset_dir)

		#########################################
		# Define different variable of interest #
		#########################################
		labels_total = tf.Variable([0] * dataset.num_samples, dtype=tf.int64,
			collections=[tf.GraphKeys.LOCAL_VARIABLES], trainable=False)
		labels_total_9 = tf.Variable([0] * dataset.num_samples, dtype=tf.int64,
			collections=[tf.GraphKeys.LOCAL_VARIABLES], trainable=False)
        
		embeddings_total = tf.Variable([[0] * embedding_dim] * dataset.num_samples, dtype=tf.float32,
			collections=[tf.GraphKeys.LOCAL_VARIABLES], trainable=False)
        
		if compute_saliency:
			print("Wait...")
			images_total = tf.Variable([np.zeros((299, 299, 3)).tolist()] * max_saliency, dtype=tf.float32,
				collections=[tf.GraphKeys.LOCAL_VARIABLES], trainable=False)
			print("Ok.")
			print("Wait...")
			saliency_total = tf.Variable([np.zeros((299, 299)).tolist()] * max_saliency, dtype=tf.float32,
				collections=[tf.GraphKeys.LOCAL_VARIABLES], trainable=False)
			print("Ok.")
        
		####################
		# Select the model #
		####################
		network_fn = nets_factory.get_network_fn(
			FLAGS.model_name,
			num_classes=(dataset.num_classes - FLAGS.labels_offset),
			is_training=False)
	
		##############################################################
		# Create a dataset provider that loads data from the dataset #
		##############################################################
		provider = slim.dataset_data_provider.DatasetDataProvider(
			dataset,
			shuffle=False,
			common_queue_capacity=2 * FLAGS.batch_size,
			common_queue_min=FLAGS.batch_size)
		[image, label] = provider.get(['image', 'label'])
		label -= FLAGS.labels_offset
	
		#####################################
		# Select the preprocessing function #
		#####################################
		preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
		image_preprocessing_fn = preprocessing_factory.get_preprocessing(
			preprocessing_name,
			is_training=False)

		eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size
	
		image = image_preprocessing_fn(image, eval_image_size, eval_image_size)
		
		images, labels = tf.train.batch(
			[image, label],
			batch_size=FLAGS.batch_size,
			num_threads=FLAGS.num_preprocessing_threads,
			capacity=5 * FLAGS.batch_size)
	
		####################
		# Define the model #
		####################
		logits, end_points = network_fn(images)
		probs = tf.nn.softmax(logits, axis=1)
        
		if FLAGS.moving_average_decay:
			variable_averages = tf.train.ExponentialMovingAverage(
				FLAGS.moving_average_decay, tf_global_step)
			variables_to_restore = variable_averages.variables_to_restore(
				slim.get_model_variables())
			variables_to_restore[tf_global_step.op.name] = tf_global_step
		else:
			variables_to_restore = slim.get_variables_to_restore()

		labels = tf.squeeze(labels)
        
		embeddings = tf.squeeze(end_points[embedding_name])

		labels_9 = label_dict_9.lookup(labels)

		# Saliency map
		if compute_saliency:
			correct_scores = tf.gather_nd(logits,
                                  tf.stack((tf.range(images.shape[0]),
                                            tf.cast(labels, tf.int32)), axis=1))
        
			loss_ = tf.reduce_sum(correct_scores, axis=0)
			grads_ = tf.gradients(loss_, images)[0]
			saliency = tf.reduce_max(tf.abs(grads_), axis=-1)
        
		op = []
		with tf.control_dependencies([labels, labels_9, embeddings]):
			labels_assign_op = tf.assign(
				labels_total[step * FLAGS.batch_size:(step + 1) * FLAGS.batch_size],
				tf.identity(labels))
			op.append(labels_assign_op)
			labels_9_assign_op = tf.assign(
				labels_total_9[step * FLAGS.batch_size:(step + 1) * FLAGS.batch_size],
				tf.identity(labels_9))
			op.append(labels_9_assign_op)
			embeddings_assign_op = tf.assign(
				embeddings_total[step * FLAGS.batch_size:(step + 1) * FLAGS.batch_size],
				tf.identity(embeddings))
			op.append(embeddings_assign_op)
            
		if compute_saliency:
			with tf.control_dependencies([saliency]):
				#step_ = tf.cond(step <= max_saliency, lambda: tf.identity(step), lambda: tf.identity(step_))
				images_assign_op = tf.assign(
					images_total[step * FLAGS.batch_size:(step + 1) * FLAGS.batch_size],
					tf.identity(images))
				op.append(images_assign_op)
				saliency_assign_op = tf.assign(
					saliency_total[step * FLAGS.batch_size:(step + 1) * FLAGS.batch_size],
					tf.identity(saliency))
				op.append(saliency_assign_op)

		with tf.control_dependencies(op):
			step_update_op = tf.assign(step, step + 1)

		# TODO(sguada) use num_epochs=1
		if FLAGS.max_num_batches:
			num_batches = FLAGS.max_num_batches
		else:
			# This ensures that we make a single pass over all of the data.
			num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))
	
		if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
			checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
		else:
			checkpoint_path = FLAGS.checkpoint_path
	
		tf.logging.info('Evaluating %s' % checkpoint_path)
		if compute_saliency:
			final_op = [embeddings_total, labels_total_9, saliency_total, images_total]
		else:
			final_op = [embeddings_total, labels_total_9]
            
		output = slim.evaluation.evaluate_once(
			master=FLAGS.master,
			checkpoint_path=checkpoint_path,
			logdir=FLAGS.vis_dir,
			num_evals=num_batches,
			eval_op=[step_update_op],
			final_op=final_op,
			variables_to_restore=variables_to_restore)
        
	embed, lab = output[0], output[1]
	log_dir = os.path.join(FLAGS.vis_dir, 'tsne_%s/' % FLAGS.dataset_split_name)
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	metadata = os.path.join(log_dir, 'metadata.tsv')
	open(metadata, 'a').close()
	images = tf.Variable(embed, name='images')
	with open(metadata, 'w') as metadata_file:
		for row in lab:
			metadata_file.write('%d\n' % row)
	with tf.Session() as sess:
		saver = tf.train.Saver([images])

		sess.run(images.initializer)
		saver.save(sess, os.path.join(log_dir, 'images.ckpt'))

		config = projector.ProjectorConfig()
		# One can add multiple embeddings.
		embedding = config.embeddings.add()
		embedding.tensor_name = images.name
		# Link this tensor to its metadata file (e.g. labels).
		embedding.metadata_path = metadata
		# Saves a config file that TensorBoard will read during startup.
		projector.visualize_embeddings(tf.summary.FileWriter(log_dir), config)

	if compute_saliency:
		sal, im = output[2], output[3]
		saliency_dir = os.path.join(FLAGS.vis_dir, 'saliency_%s/' % FLAGS.dataset_split_name)
		if not os.path.exists(saliency_dir):
			os.makedirs(saliency_dir)
            
		im = im / 2 + 0.5
        
		for i, img in enumerate(im[:max_saliency]):
			scipy.misc.imsave(saliency_dir + 'img_%i_label_%i.jpg'% (i, lab[i]), img)
			scipy.misc.imsave(saliency_dir + 'sal_%i_label_%i.jpg'% (i, lab[i]), sal[i])
            

if __name__ == '__main__':
	tf.app.run()
