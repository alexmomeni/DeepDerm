"""Download images script.

Usage:
```shell

$ python convert_data.py \
	--dataset_dir=data/
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import requests
import os
import shutil
from functools import reduce
import operator
import itertools
import tensorflow as tf

tf.app.flags.DEFINE_string(
	'dataset_dir',
	None,
	'The directory where the output TFRecords and temporary files are saved.')

FLAGS = tf.app.flags.FLAGS

def get_from_dict(data_dict, map_list):
	return reduce(operator.getitem, map_list, data_dict)

def get_taxonomy_dict():
	taxonomy_file_path = "data/private/taxonomy.txt"
	with open (taxonomy_file_path, "r") as myfile:
		list_taxonomy=myfile.readlines()
	for i, string in enumerate(list_taxonomy):
		list_taxonomy[i] = string.replace('\n', '')
		list_taxonomy[i] = list_taxonomy[i].replace('//', '/')
		list_taxonomy[i] = list_taxonomy[i].lower()
		list_taxonomy[i] = list_taxonomy[i].replace(" ", "")
		list_taxonomy[i] = list_taxonomy[i].split('/')
	list_taxonomy = list(list_taxonomy for list_taxonomy,_ in 
		itertools.groupby(list_taxonomy))
	taxonomy_dict = {}
	for path in list_taxonomy:
		current_level = taxonomy_dict
		for part in path:
			if part not in current_level:
				current_level[part] = {}
			current_level = current_level[part]
	return taxonomy_dict

def main(_):
	count = 0
	if not FLAGS.dataset_dir:
	  raise ValueError('You must supply the dataset directory with --dataset_dir')
	metajson_file_path = "data/private/meta.json"
	dataset_dir = FLAGS.dataset_dir
	taxonomy_dict = get_taxonomy_dict()
	data = json.load(open(metajson_file_path))
	print("Downloading")
	for i, dico in enumerate(data):
		if i!=0 and i % 100 == 0:
			print("..%i.." % count, end="", flush=True)
		try:
			url = dico['link']
			file_name = dico['filename'].replace(".jpg", "")
			taxonomy_path = ["-".join(j).replace(" ", "").replace("/", "").lower() 
				for j in dico['tax_path']]
			if get_from_dict(taxonomy_dict, taxonomy_path) == {}:
				folder_path = dataset_dir + 'images/' + '/'.join(taxonomy_path) + '/'
				if not os.path.exists(folder_path):
					os.makedirs(folder_path)
				response = requests.get(url, stream=True, timeout=20)
				if response.status_code != 200:
					print("status_code != 200")
					pass
				with open(folder_path + file_name + '.jpg', 'wb') as out_file:
					shutil.copyfileobj(response.raw, out_file)
				del response
				count += 1
		except requests.ConnectionError as e:
			print("OOPS!! Connection Error. Make sure you are connected to Internet. Technical Details given below.\n")
			print(str(e))
		except requests.Timeout as e:
			print("OOPS!! Timeout Error")
			print(str(e))
		except requests.RequestException as e:
			print("OOPS!! General Error")
			print(str(e))
		except KeyboardInterrupt:
			print("Someone closed the program")
		except KeyError as e:
			print("No link available")
			print(str(e))
		except:
			print("Unknown error...")

if __name__ == '__main__':
	tf.app.run()