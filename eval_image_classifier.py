"""Evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import numpy as np

from datasets import dataset_split
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
	'usage', "training", 'Usage')

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
	'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

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
	FLAGS.eval_dir = FLAGS.eval_dir + FLAGS.model_name + '/general/'
	if FLAGS.usage == 'training':
		FLAGS.eval_dir = FLAGS.checkpoint_path + FLAGS.model_name + '/validation/'
	FLAGS.checkpoint_path = FLAGS.checkpoint_path + FLAGS.model_name + '/training/'
    
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
                
	keys_3 = []
	vals_3 = []
	with open(FLAGS.dataset_dir + "labels_9_to_labels_3.txt") as f:
		for line in f:
			 (key, val) = line.split(":")
			 keys_3.append(int(key))
			 vals_3.append(int(val))

	tf.logging.set_verbosity(tf.logging.INFO)

	tf.reset_default_graph()

	with tf.Graph().as_default():
		tf_global_step = slim.get_or_create_global_step()

		label_dict_9 = tf.contrib.lookup.HashTable(
			tf.contrib.lookup.KeyValueTensorInitializer(
				keys_9, vals_9, key_dtype=tf.int64, value_dtype=tf.int64), -1)
        
		label_dict_3 = tf.contrib.lookup.HashTable(
			tf.contrib.lookup.KeyValueTensorInitializer(
				keys_3, vals_3, key_dtype=tf.int64, value_dtype=tf.int64), -1)

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
		predictions_total = tf.Variable([0] * dataset.num_samples, dtype=tf.int64,
			collections=[tf.GraphKeys.LOCAL_VARIABLES], trainable=False)

		labels_total_9 = tf.Variable([0] * dataset.num_samples, dtype=tf.int64,
			collections=[tf.GraphKeys.LOCAL_VARIABLES], trainable=False)
		predictions_total_9 = tf.Variable([0] * dataset.num_samples, dtype=tf.int64,
			collections=[tf.GraphKeys.LOCAL_VARIABLES], trainable=False)
		predictions_total_9_sum = tf.Variable([0] * dataset.num_samples, dtype=tf.int64,
			collections=[tf.GraphKeys.LOCAL_VARIABLES], trainable=False)
        
		labels_total_3 = tf.Variable([0] * dataset.num_samples, dtype=tf.int64,
			collections=[tf.GraphKeys.LOCAL_VARIABLES], trainable=False)
		predictions_total_3 = tf.Variable([0] * dataset.num_samples, dtype=tf.int64,
			collections=[tf.GraphKeys.LOCAL_VARIABLES], trainable=False)
		predictions_total_3_sum = tf.Variable([0] * dataset.num_samples, dtype=tf.int64,
			collections=[tf.GraphKeys.LOCAL_VARIABLES], trainable=False)
        
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
		logits, _ = network_fn(images)
		probs = tf.nn.softmax(logits, axis=1)

		if FLAGS.moving_average_decay:
			variable_averages = tf.train.ExponentialMovingAverage(
				FLAGS.moving_average_decay, tf_global_step)
			variables_to_restore = variable_averages.variables_to_restore(
				slim.get_model_variables())
			variables_to_restore[tf_global_step.op.name] = tf_global_step
		else:
			variables_to_restore = slim.get_variables_to_restore()
	
		predictions = tf.argmax(logits, 1)
		labels = tf.squeeze(labels)

		labels_9 = label_dict_9.lookup(labels)
		predictions_9 = label_dict_9.lookup(predictions)
        
		probs_9 = tf.transpose(tf.stack([
            tf.reduce_sum(
                tf.transpose(tf.boolean_mask(
                    tf.transpose(probs), tf.equal(vals_9, i))),
                axis=1)
            for i in range(9)]))
        
		predictions_9_sum = tf.argmax(probs_9, axis=1)
        
		labels_3 = label_dict_3.lookup(labels_9)
		predictions_3 = label_dict_3.lookup(predictions_9)

		probs_3 = tf.transpose(tf.stack([
            tf.reduce_sum(
                tf.transpose(tf.boolean_mask(
                    tf.transpose(probs_9), tf.equal(vals_3, i))),
                axis=1)
            for i in range(3)]))

		predictions_3_sum = tf.argmax(probs_3, axis=1)

		with tf.control_dependencies([labels, predictions]):
			batch_accuracy = tf.reduce_mean(tf.to_float(tf.equal(labels, predictions)))
			batch_accuracy_9 = tf.reduce_mean(tf.to_float(tf.equal(labels_9, predictions_9)))
			batch_accuracy_3 = tf.reduce_mean(tf.to_float(tf.equal(labels_3, predictions_3)))
			batch_accuracy_9_sum = tf.reduce_mean(tf.to_float(tf.equal(labels_9, predictions_9_sum)))
			batch_accuracy_3_sum = tf.reduce_mean(tf.to_float(tf.equal(labels_3, predictions_3_sum)))
			batch_accuracy = tf.Print(batch_accuracy, [batch_accuracy],
				'batch_accuracy')
			batch_accuracy_9 = tf.Print(batch_accuracy_9, [batch_accuracy_9],
				'batch_accuracy_9')
			batch_accuracy_3 = tf.Print(batch_accuracy_3, [batch_accuracy_3],
				'batch_accuracy_3')
			batch_accuracy_9_sum = tf.Print(batch_accuracy_9_sum, [batch_accuracy_9_sum],
				'batch_accuracy_9_sum')
			batch_accuracy_3_sum = tf.Print(batch_accuracy_3_sum, [batch_accuracy_3_sum],
				'batch_accuracy_3_sum')
			labels_assign_op = tf.assign(
				labels_total[step * FLAGS.batch_size:(step + 1) * FLAGS.batch_size],
				tf.identity(labels))
			predictions_assign_op = tf.assign(
				predictions_total[step * FLAGS.batch_size:(step + 1) * FLAGS.batch_size],
				tf.identity(predictions))
			labels_9_assign_op = tf.assign(
				labels_total_9[step * FLAGS.batch_size:(step + 1) * FLAGS.batch_size],
				tf.identity(labels_9))
			predictions_9_assign_op = tf.assign(
				predictions_total_9[step * FLAGS.batch_size:(step + 1) * FLAGS.batch_size],
				tf.identity(predictions_9))
			predictions_9_sum_assign_op = tf.assign(
				predictions_total_9_sum[step * FLAGS.batch_size:(step + 1) * FLAGS.batch_size],
				tf.identity(predictions_9_sum))
			labels_3_assign_op = tf.assign(
				labels_total_3[step * FLAGS.batch_size:(step + 1) * FLAGS.batch_size],
				tf.identity(labels_3))
			predictions_3_assign_op = tf.assign(
				predictions_total_3[step * FLAGS.batch_size:(step + 1) * FLAGS.batch_size],
				tf.identity(predictions_3))
			predictions_3_sum_assign_op = tf.assign(
				predictions_total_3_sum[step * FLAGS.batch_size:(step + 1) * FLAGS.batch_size],
				tf.identity(predictions_3_sum))

		with tf.control_dependencies([labels_assign_op, predictions_assign_op,
                                      labels_9_assign_op, predictions_9_assign_op,
                                      labels_3_assign_op, predictions_3_assign_op,
                                      predictions_9_sum_assign_op, predictions_3_sum_assign_op,
                                      batch_accuracy, batch_accuracy_9, batch_accuracy_3,
                                      batch_accuracy_9_sum, batch_accuracy_3_sum]):
			step_update_op = tf.assign(step, step + 1)

		# Define the metrics:
		names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
			'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
			'Recall': slim.metrics.streaming_recall(predictions, labels),
			'Accuracy_9': slim.metrics.streaming_accuracy(predictions_9, labels_9),
			'Recall_9': slim.metrics.streaming_recall(predictions_9, labels_9),
			'Accuracy_3': slim.metrics.streaming_accuracy(predictions_3, labels_3),
			'Recall_3': slim.metrics.streaming_recall(predictions_3, labels_3),
			'Accuracy_9_sum': slim.metrics.streaming_accuracy(predictions_9_sum, labels_9),
			'Recall_9_sum': slim.metrics.streaming_recall(predictions_9_sum, labels_9),
			'Accuracy_3_sum': slim.metrics.streaming_accuracy(predictions_3_sum, labels_3),
			'Recall_3_sum': slim.metrics.streaming_recall(predictions_3_sum, labels_3),
		})

		# Print the summaries to screen.
		for name, value in names_to_values.items():
			summary_name = 'eval/%s' % name
			op = tf.summary.scalar(summary_name, value, collections=[])
			op = tf.Print(op, [value], summary_name)
			tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

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

		confusion_matrix = tf.confusion_matrix(labels_total, predictions_total)
		confusion_matrix_9 = tf.confusion_matrix(labels_total_9, predictions_total_9)
		confusion_matrix_3 = tf.confusion_matrix(labels_total_3, predictions_total_3)
		confusion_matrix_9_sum = tf.confusion_matrix(labels_total_9, predictions_total_9_sum)
		confusion_matrix_3_sum = tf.confusion_matrix(labels_total_3, predictions_total_3_sum)

		if FLAGS.usage == "testing":
			output = slim.evaluation.evaluate_once(
				master=FLAGS.master,
				checkpoint_path=checkpoint_path,
				logdir=FLAGS.eval_dir,
				num_evals=num_batches,
				eval_op=list(names_to_updates.values()) + [step_update_op],
				final_op=[confusion_matrix, confusion_matrix_9, confusion_matrix_3,
                      confusion_matrix_9_sum, confusion_matrix_3_sum],
				variables_to_restore=variables_to_restore)

			np.save(FLAGS.eval_dir + "output.npy", output)
            
		elif FLAGS.usage == "training":
			slim.evaluation.evaluation_loop(
				master=FLAGS.master,
				checkpoint_dir=FLAGS.checkpoint_path,
				logdir=FLAGS.eval_dir,
				num_evals=num_batches,
				eval_op=list(names_to_updates.values()),
				variables_to_restore=variables_to_restore,
				eval_interval_secs=250)

if __name__ == '__main__':
	tf.app.run()
