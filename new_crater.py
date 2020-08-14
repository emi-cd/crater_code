import sys, os, glob, shutil
import pandas as pd

import download as dl
import register as rg
import const


def cleaning(path):
	for b in glob.glob(path+'M*'):
		try: 
			os.rmdir(b)
		except:
			continue
	if len(glob.glob(path+'*')) == 1:
		shutil.rmtree(path)


def main(argv):
	if len(argv) != 2:
		print('Please input 2 args. \nex) python new_crater.py 20.7135 335.6698')
		return 0
	point = [argv[0], argv[1]]

	# Prepare the data
	data = dl.get_data_from_point(point)
	pair = dl.make_temporal_pair(data)
	print(point, pair)

	# Register
	point_path = argv[0] + '-' + argv[1] + '/'
	output_path = const.OUTPUT_PATH + point_path
	# os.makedirs(output_path, exist_ok=True)
	# data.to_csv(output_path + 'INFO.csv')

	for key, vals in pair.items():
		dl.download_nac_one(data, key)
		before = rg.NacImage(data.loc[key])
		for val in vals:
			print(key, vals, '->', val)
			dl.download_nac_one(data, val)
			after = rg.NacImage(data.loc[val])
			t_pair = rg.TemporalPair(before, after)
			t_pair.make_dif(output_path, True)

	cleaning(output_path)


if __name__ == '__main__':
	args = sys.argv
	main(args[1:])