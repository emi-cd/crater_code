import collect_nac as cn


CRATER_SIZE = 10.0

def main():
	data = cn.collect_after_img_size(CRATER_SIZE)
	for imgs, output_path in data:
		cn.make_multi_tiff(imgs, output_path)


if __name__ == '__main__':
	main()