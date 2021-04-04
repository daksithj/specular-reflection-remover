import cv2
from sklearn.cluster import KMeans


def read_files():

	for x in range(1500):
		base_image = cv2.imread('Dataset/Specular/' + str(x) + '_' + str(2) + '.png')
		segments = extract_segments(base_image)

		for seg in segments:
			for y in range(1, 4):
				specular_image = cv2.imread('Dataset/Specular/' + str(x) + '_' + str(y) + '.png')
				diffuse_image = cv2.imread('Dataset/Diffuse/' + str(x) + '_' + str(y) + '.png')
				specular_patch = get_patch(specular_image, seg)
				diffuse_patch = get_patch(diffuse_image, seg)
				write_patches(specular_patch, diffuse_patch, x, y)


def write_patches(specular_image, diffuse_image, file_num, ob_view):

	file_name = 'Dataset/Specular_multi/' + str(file_num) + '_' + str(ob_view) + '.png'
	cv2.imwrite(file_name, specular_image)

	file_name = 'Dataset/Diffuse_multi/' + str(file_num) + '_' + str(ob_view) + '.png'
	cv2.imwrite(file_name, diffuse_image)


def extract_segments(image):
	ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
	ss.setBaseImage(image)

	ss.switchToSelectiveSearchFast()

	rects = ss.process()
	mids = []
	for (x, y, w, h) in rects:
		mids.append([(x + w//2), (y + h//2)])

	model = KMeans(n_clusters=6)

	model.fit(mids)

	centres = model.cluster_centers_

	new_cent = []
	for x, y in centres:
		new_cent.append([int(x), int(y)])

	coordinates = []
	for x, y in new_cent:

		x1 = x - 256 // 2
		if (x1 < 0):
			x1 = 0
			x2 = 256
		else:
			x2 = x + 256 // 2
			if x2 >= image.shape[1]:
				x2 = image.shape[1]
				x1 = image.shape[1] - 256

		y1 = y - 256 // 2
		if (y1 < 0):
			y1 = 0
			y2 = 256
		else:
			y2 = y + 256 // 2
			if y2 >= image.shape[1]:
				y2 = image.shape[1]
				y1 = image.shape[1] - 256

		coordinates.append((x1, x2, y1, y2))
	return coordinates


def get_patch(image, coordinates):
	x1, x2, y1, y2 = coordinates
	crop = image[x1:x2, y1:y2, :]
	return crop
