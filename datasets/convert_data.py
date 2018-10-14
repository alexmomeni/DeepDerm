"""
Usage:
```shell

$ python convert_data.py \
		--dataset_dir=data/
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import json
import numpy as np
import tensorflow as tf
import dataset_utils

# The number of images in the validation set.
_NUM_VALIDATION = 1000

_NUM_VALIDATION_PER_CLASS = 100

_NUM_TEST_ISIC_PER_CLASS = 200

_NUM_TEST_ISIC = 500

# Seed for repeatability.
_RANDOM_SEED = 0

# The number of shards per dataset split.
_NUM_SHARDS = 5

# Flag 
tf.app.flags.DEFINE_string(
		'dataset_dir',
		None,
		'The directory where the output TFRecords and temporary files are saved.')

FLAGS = tf.app.flags.FLAGS

dico_isic = {'actinic keratosis': 'epidermal-tumors-pre-malignant-and-malignant/cutaneous-squamous-cell-carcinoma-precursors-and-mimics/actinic-keratosis/actinic-keratosis/dermoscopy/',
 'angiofibroma or fibrous papule': None,
 'angioma': 'benign-dermal-tumors-cysts-sinuses/vascular-tumors-and-malformations/benign-vascular-lesions/hemangioma/angioma/dermoscopy/',
 'atypical melanocytic proliferation': None,
 'basal cell carcinoma': 'epidermal-tumors-pre-malignant-and-malignant/basal-cell-carcinoma/basal-cell-carcinoma/dermoscopy/',
 'dermatofibroma': 'benign-dermal-tumors-cysts-sinuses/dermatofibroma/dermoscopy/',
 'lentigo NOS': None,
 'lentigo simplex': 'pigmented-lesions-benign/lentigo-simplex/dermoscopy/',
 'lichenoid keratosis': None,
 'melanoma': 'pigmented-lesions-malignant/melanoma/dermoscopy/',
 'nevus': 'pigmented-lesions-benign/melanocytic-nevi/dermoscopy/',
 'other': None,
 'pigmented benign keratosis': 'epidermal-tumors-hamartomas-milia-and-growths-benign/seborrheic-keratosis/benign-keratosis/dermoscopy/',
 'scar': None,
 'seborrheic keratosis': 'epidermal-tumors-hamartomas-milia-and-growths-benign/seborrheic-keratosis/seborrheic-keratosis/dermoscopy/',
 'solar lentigo': 'pigmented-lesions-benign/solar-lentigo/dermoscopy/',
 'squamous cell carcinoma': 'epidermal-tumors-pre-malignant-and-malignant/cutaneous-squamous-cell-carcinoma-precursors-and-mimics/squamous-cell-carcinoma/squamous-cell-carcinoma/dermoscopy/',
 'vascular lesion': None}

class ImageReader(object):
	"""Helper class that provides TensorFlow image coding utilities."""

	def __init__(self):
		# Initializes function that decodes RGB JPEG data.
		self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
		self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

	def read_image_dims(self, sess, image_data):
		image = self.decode_jpeg(sess, image_data)
		return image.shape[0], image.shape[1]

	def decode_jpeg(self, sess, image_data):
		image = sess.run(self._decode_jpeg,
			feed_dict={self._decode_jpeg_data: image_data})
		assert len(image.shape) == 3
		assert image.shape[2] == 3
		return image

def _count_files(dir):
	return sum([len(files) for r, d, files in os.walk(dir)]) 

def _partitioning(rootdir, max_class_size=1000):
	classes = []
	for subdir, dirs, files in os.walk(rootdir):
		subdir = subdir + '/'
		if _count_files(subdir) < max_class_size and \
			(len(classes)==0 or classes[-1] not in subdir):
			classes.append(subdir)
		elif _count_files(subdir) >= max_class_size and len(dirs)==0:
			classes.append(subdir)
	return classes

def _get_filenames_and_classes(dataset_dir, max_class_size=1000):
	"""Returns a list of filenames and inferred class names.

	Args:
		dataset_dir: A directory containing a set of subdirectories representing
			class names. Each subdirectory should contain PNG or JPG encoded images.

	Returns:
		A list of image file paths, relative to `dataset_dir` and the list of
		subdirectories, representing class names.
	"""
	rootdir = os.path.join(dataset_dir, 'images')
	classes = _partitioning(rootdir, max_class_size)
	fileclasses = []
	filepaths = []
	for _dir in classes:
		count = 0
		for subdir, dirs, files in os.walk(_dir):
			if files:
				for file in files:
					if os.path.splitext(file)[1] == '.jpg':
						file_path = os.path.join(subdir, file)
						file_class = _dir.replace(os.path.join(dataset_dir, 'images/'), '')
						filepaths.append(file_path)
						fileclasses.append(file_class)
						count += 1
					if count >= max_class_size:
						break

	return filepaths, fileclasses

def _get_filenames_and_classes_isic(dataset_dir):
	"""Returns a list of filenames and inferred class names.

	Args:
		dataset_dir: A directory containing a set of subdirectories representing
			class names. Each subdirectory should contain PNG or JPG encoded images.

	Returns:
		A list of image file paths, relative to `dataset_dir` and the list of
		subdirectories, representing class names.
	"""
	rootdir = os.path.join(dataset_dir, 'isic_images')
	filepaths = []
	fileclasses = []
	for subdir, dirs, files in os.walk(rootdir):
		if files:
			for file in files:
				file_name, file_extension = os.path.splitext(file)
				if file_extension == '.jpg':
					file_path = os.path.join(subdir, file)
					file_info = json.load(open(os.path.join(subdir, file_name + ".json")))
					try:
						if file_info["meta"]["acquisition"]["image_type"] == "dermoscopic":
							diagnosis = file_info["meta"]["clinical"]["diagnosis"]
							file_class = dico_isic[diagnosis]
							if diagnosis == "nevus" and file_info["meta"]["clinical"]["benign_malignant"] != "benign":
								continue
							if diagnosis == "melanoma" and file_info["meta"]["clinical"]["benign_malignant"] != "malignant":
								continue
							if file_class is None:
								continue
						else:
							continue
					except:
						continue
					filepaths.append(file_path)
					fileclasses.append(file_class)

	return filepaths, fileclasses

def _get_dataset_filename(dataset_dir, split_name, shard_id):
	output_filename = 'images_%s_%05d-of-%05d.tfrecord' % (
			split_name, shard_id, _NUM_SHARDS)
	return os.path.join(dataset_dir + "dataset_test/", output_filename)

def _convert_dataset(split_name, filenames, fileclassnames, class_names_to_ids, dataset_dir):
	"""Converts the given filenames to a TFRecord dataset.

	Args:
		split_name: The name of the dataset, either 'train' or 'validation'.
		filenames: A list of absolute paths to png or jpg images.
		class_names_to_ids: A dictionary from class names (strings) to ids
			(integers).
		dataset_dir: The directory where the converted datasets are stored.
	"""
	assert split_name in ['train', 'validation', "test_isic"]

	num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

	with tf.Graph().as_default():
		image_reader = ImageReader()

		with tf.Session('') as sess:

			for shard_id in range(_NUM_SHARDS):
				output_filename = _get_dataset_filename(
						dataset_dir, split_name, shard_id)

				with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
					start_ndx = shard_id * num_per_shard
					end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
					for i in range(start_ndx, end_ndx):
						sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
								i+1, len(filenames), shard_id))
						sys.stdout.flush()

						# Read the filename:
						image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
						try:
							height, width = image_reader.read_image_dims(sess, image_data)
						except:
							continue

						class_name = (fileclassnames[i]).replace(os.path.join(dataset_dir, 'images/'), '')
						class_id = class_names_to_ids[class_name]

						example = dataset_utils.image_to_tfexample(
								image_data, b'jpg', height, width, class_id)
						tfrecord_writer.write(example.SerializeToString())

	sys.stdout.write('\n')
	sys.stdout.flush()

def _clean_up_temporary_files(dataset_dir):
	"""Removes temporary files used to create the dataset.

	Args:
		dataset_dir: The directory where the temporary files are stored.
	"""
	tmp_dir = os.path.join(dataset_dir, 'images')
	tf.gfile.DeleteRecursively(tmp_dir)

def _dataset_exists(dataset_dir):
	for split_name in ['train', 'validation', "test_isic"]:
		for shard_id in range(_NUM_SHARDS):
			output_filename = _get_dataset_filename(
					dataset_dir, split_name, shard_id)
			if not tf.gfile.Exists(output_filename):
				return False
	return True

def run(dataset_dir, max_class_size=1000):
	"""Runs the download and conversion operation.

	Args:
		dataset_dir: The dataset directory where the dataset is stored.
	"""
	if not tf.gfile.Exists(dataset_dir + "dataset_test/"):
		tf.gfile.MakeDirs(dataset_dir + "dataset_test/")

	if _dataset_exists(dataset_dir):
		print('Dataset files already exist. Exiting without re-creating them.')
		return

	##########################
	#          ISIC          # 
	##########################
	# Shuffle the data
	filepaths_isic, fileclasses_isic = _get_filenames_and_classes_isic(dataset_dir)
    
	assert(len(filepaths_isic) == len(fileclasses_isic))
	random.seed(_RANDOM_SEED)
	random.shuffle(filepaths_isic)
	random.seed(_RANDOM_SEED)
	random.shuffle(fileclasses_isic)

	##########################
	#        TEST ISIC       #
	##########################
	# Contruct the test set
	test_filepaths_isic = []
	test_fileclasses_isic = []
	index_isic_ = []
	count_isic = 0
	count_isic_ = {}
	count_isic_["pigmented-lesions-benign/melanocytic-nevi/dermoscopy/"] = 0
	count_isic_["pigmented-lesions-malignant/melanoma/dermoscopy/"] = 0

	for j, i in enumerate(filepaths_isic):
		if fileclasses_isic[j] == "pigmented-lesions-benign/melanocytic-nevi/dermoscopy/" and \
			count_isic_["pigmented-lesions-benign/melanocytic-nevi/dermoscopy/"] < _NUM_TEST_ISIC_PER_CLASS:
			test_filepaths_isic.append(i)
			test_fileclasses_isic.append(fileclasses_isic[j])
			count_isic_["pigmented-lesions-benign/melanocytic-nevi/dermoscopy/"] += 1
			index_isic_.append(j)
			count_isic += 1

		elif fileclasses_isic[j] == "pigmented-lesions-malignant/melanoma/dermoscopy/" and \
			count_isic_["pigmented-lesions-malignant/melanoma/dermoscopy/"] < _NUM_TEST_ISIC_PER_CLASS:
			test_filepaths_isic.append(i)
			test_fileclasses_isic.append(fileclasses_isic[j])
			count_isic_["pigmented-lesions-malignant/melanoma/dermoscopy/"] += 1
			index_isic_.append(j)
			count_isic += 1

		if count_isic >= _NUM_TEST_ISIC:
			break
    
	val_dico_isic = [i for i in list(dico_isic.values()) if i is not None]
	count_training_isic = {}
	for i in val_dico_isic:
		count_training_isic[i] = 0
	training_filepaths_isic = []
	training_fileclasses_isic = []
	for i, j in enumerate(filepaths_isic):
		if i not in index_isic_ and count_training_isic[fileclasses_isic[i]] < max_class_size:
			training_filepaths_isic.append(j)
			training_fileclasses_isic.append(fileclasses_isic[i])
			count_training_isic[fileclasses_isic[i]] +=1
            
	##########################
	#          URLS          # 
	##########################
	filepaths, fileclasses = _get_filenames_and_classes(dataset_dir)

	##########################
	#          MERGE         # 
	##########################
	filepaths = filepaths + training_filepaths_isic
	fileclasses = fileclasses + training_fileclasses_isic
    
	# Shuffle the data
	assert(len(filepaths) == len(fileclasses))

	random.seed(_RANDOM_SEED)
	random.shuffle(filepaths)
	random.seed(_RANDOM_SEED)
	random.shuffle(fileclasses)

	##########################
	#          VAL           # 
	##########################

	validation_filepaths = []
	validation_fileclasses = []
	index_ = []
	count = 0
	count_ = {}
	count_["benign-dermal-tumors-cysts-sinuses"] = 0
	count_["cutaneous-lymphoma-and-lymphoid-infiltrates"] = 0
	count_["epidermal-tumors-hamartomas-milia-and-growths-benign"] = 0
	count_["epidermal-tumors-pre-malignant-and-malignant"] = 0
	count_["genodermatoses-and-supernumerary-growths"] = 0
	count_["inflammatory"] = 0
	count_["malignant-dermal-tumor"] = 0
	count_["pigmented-lesions-benign"] = 0
	count_["pigmented-lesions-malignant"] = 0

	for j, i in enumerate(filepaths):
		name_ =  fileclasses[j].split('/')[0]
		if name_ == "benign-dermal-tumors-cysts-sinuses" and \
			count_["benign-dermal-tumors-cysts-sinuses"] < _NUM_VALIDATION_PER_CLASS:
			validation_filepaths.append(i)
			validation_fileclasses.append(fileclasses[j])
			count_["benign-dermal-tumors-cysts-sinuses"] += 1
			index_.append(j)
			count += 1

		elif name_ == "cutaneous-lymphoma-and-lymphoid-infiltrates" and \
			count_["cutaneous-lymphoma-and-lymphoid-infiltrates"] < _NUM_VALIDATION_PER_CLASS:
			validation_filepaths.append(i)
			validation_fileclasses.append(fileclasses[j])
			count_["cutaneous-lymphoma-and-lymphoid-infiltrates"] += 1
			index_.append(j)
			count += 1

		elif name_ == "epidermal-tumors-hamartomas-milia-and-growths-benign" and \
			count_["epidermal-tumors-hamartomas-milia-and-growths-benign"] < _NUM_VALIDATION_PER_CLASS:
			validation_filepaths.append(i)
			validation_fileclasses.append(fileclasses[j])
			count_["epidermal-tumors-hamartomas-milia-and-growths-benign"] += 1
			index_.append(j)
			count += 1

		elif name_ == "epidermal-tumors-pre-malignant-and-malignant" and \
			count_["epidermal-tumors-pre-malignant-and-malignant"] < _NUM_VALIDATION_PER_CLASS:
			validation_filepaths.append(i)
			validation_fileclasses.append(fileclasses[j])
			count_["epidermal-tumors-pre-malignant-and-malignant"] += 1
			index_.append(j)
			count += 1

		elif name_ == "genodermatoses-and-supernumerary-growths" and \
			count_["genodermatoses-and-supernumerary-growths"] < _NUM_VALIDATION_PER_CLASS:
			validation_filepaths.append(i)
			validation_fileclasses.append(fileclasses[j])
			count_["genodermatoses-and-supernumerary-growths"] += 1
			index_.append(j)
			count += 1

		elif name_ == "inflammatory" and \
			count_["inflammatory"] < _NUM_VALIDATION_PER_CLASS:
			validation_filepaths.append(i)
			validation_fileclasses.append(fileclasses[j])
			count_["inflammatory"] += 1
			index_.append(j)
			count += 1

		elif name_ == "malignant-dermal-tumor" and \
			count_["malignant-dermal-tumor"] < _NUM_VALIDATION_PER_CLASS:
			validation_filepaths.append(i)
			validation_fileclasses.append(fileclasses[j])
			count_["malignant-dermal-tumor"] += 1
			index_.append(j)
			count += 1

		elif name_ == "pigmented-lesions-benign" and \
			count_["pigmented-lesions-benign"] < _NUM_VALIDATION_PER_CLASS:
			validation_filepaths.append(i)
			validation_fileclasses.append(fileclasses[j])
			count_["pigmented-lesions-benign"] += 1
			index_.append(j)
			count += 1

		elif name_ == "pigmented-lesions-malignant" and \
			count_["pigmented-lesions-malignant"] < _NUM_VALIDATION_PER_CLASS:
			validation_filepaths.append(i)
			validation_fileclasses.append(fileclasses[j])
			count_["pigmented-lesions-malignant"] += 1
			index_.append(j)
			count += 1

		if count >= _NUM_VALIDATION:
			break
    
	##########################
	#          TRAIN         # 
	##########################
	training_filepaths = [filepaths[i] for i in range(len(filepaths)) if i not in index_]
	training_fileclasses = [fileclasses[i] for i in range(len(fileclasses)) if i not in index_]

	##########################
	#          CLASS         # 
	##########################
	class_names = sorted(list(set(fileclasses)))
	class_names_to_ids = dict(zip(class_names, range(len(class_names))))

	## First, convert the training and validation sets.
	_convert_dataset('test_isic', test_filepaths_isic, test_fileclasses_isic, 
		class_names_to_ids, dataset_dir)
	_convert_dataset('validation', validation_filepaths, validation_fileclasses, 
		class_names_to_ids, dataset_dir)
	_convert_dataset('train', training_filepaths, training_fileclasses, 
		class_names_to_ids, dataset_dir)

	# Finally, write the labels file:

	# We associate a number to finest level classes
	labels_to_class_names = dict(zip(range(len(class_names)), class_names))

	# We associate a number to level nine classes
	class_names_9 = sorted(list(set([name.split('/')[0] for name in class_names])))
	labels_to_class_names_9 = dict(zip(range(len(class_names_9)), class_names_9))

	# Mapping from finest level classes to level nine classes
	labels_to_labels_9 = {}
	for k, v in labels_to_class_names.items():
		name = v.split('/')[0]
		label_9 = list(labels_to_class_names_9.values()).index(name)
		labels_to_labels_9[k] = label_9

	# Write the corresponding files
	dataset_utils.write_label_file(labels_to_class_names, dataset_dir + "dataset_test/",
		filename='labels.txt')
	dataset_utils.write_label_file(labels_to_class_names_9, dataset_dir + "dataset_test/",
		filename='labels_9.txt')
	dataset_utils.write_label_file(labels_to_labels_9, dataset_dir + "dataset_test/",
		filename='labels_to_labels_9.txt')

	#_clean_up_temporary_files(dataset_dir)
	print('\nFinished!')


def main(_):
	if not FLAGS.dataset_dir:
		raise ValueError('You must supply the dataset directory with --dataset_dir')
	run(FLAGS.dataset_dir)


if __name__ == '__main__':
		tf.app.run()

