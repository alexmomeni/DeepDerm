from PIL import Image
import os
import tensorflow as tf

tf.app.flags.DEFINE_string(
	'dataset_dir',
	None,
	'The directory where the images are saved.')

FLAGS = tf.app.flags.FLAGS

def run(dataset_dir):
	rootdir = os.path.join(dataset_dir, 'images')
	for subdir, dirs, files in os.walk(rootdir):
		for file in files:
			extension = file.split('.')[-1]
			if extension == 'jpg':
				fileLoc = subdir+'/'+file
				try:
					img = Image.open(fileLoc)
				except OSError:
					os.remove(fileLoc)
					print("OSError, removed", fileLoc)
					continue
				if img.mode != 'RGB':
					os.remove(fileLoc)
					print("RGBError, removed", fileLoc)

def main(_):
	if not FLAGS.dataset_dir:
		raise ValueError('You must supply the dataset directory with --dataset_dir')
	run(FLAGS.dataset_dir)


if __name__ == '__main__':
	tf.app.run()
