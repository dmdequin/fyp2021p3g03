

def get_boundaries(image):

	mask = np.where(image == 1)
	left = min(mask[1])
	right = max(mask[1])
	upper = min(mask[0])
	lower = max(mask[0])
	return upper, lower, left, right

def get_center(image): # NOT NEEDED ANYMORE ?

	up, dw, lt, rt = get_boundaries(image)
	center = ((up+dw)/2, (lt+rt)/2)
	return center

def zoom(image):

	up, dw, lt, rt = get_boundaries(image)
	rectangle = image[up:dw+1, lt:rt+1]
	return rectangle

def cuts(image):

	center_h = image.shape[0] // 2 # The image shape contains a tuple with height and width (in pixels)
	if image.shape[0] % 2 == 0:
		upside = image[:center_h,:]
		downside = image[center_h:,:]
	else: # If the height is an uneven number of pixels, the cut has to "share" the center, to return 2 equal sides
		upside = image[:center_h+1,:]
		downside = image[center_h:,:]

	center_w = image.shape[1] // 2    
	if image.shape[1] % 2 == 0:
		leftside = image[:,:center_w]
		rightside = image[:,center_w:]
	else:
		leftside = image[:,:center_w+1]
		rightside = image[:,center_w:]

	return upside, downside, leftside, rightside

def test_symmetry(image, rot_deg=30):

	assert (rot_deg <= 90) and (rot_deg >= 0), "Rotation degree should be positive and at most 90 deg"
	optimal = 0

	for deg in range(0,90, rot_deg):
		#rot_image = skimage.transform.rotate(image, deg)
		z = zoom(rot_image)

		upside, downside, leftside, rightside = cuts(z)

		up_dw = np.sum(np.bitwise_and(upside.astype(int), np.flipud(downside).astype(int))) /\
		np.sum(np.bitwise_or(upside.astype(int), np.flipud(downside).astype(int)))

		lt_rt = np.sum(np.bitwise_and(leftside.astype(int), np.fliplr(rightside).astype(int))) /\
		np.sum(np.bitwise_or(leftside.astype(int), np.fliplr(rightside).astype(int)))

		symmetry = (up_dw+lt_rt)/2

		if symmetry > optimal: optimal = symmetry

	return symmetry

def rgb2gray(rgb):
	return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def get_mask(image):
	gray = rgb2gray(image)
	plt.hist(gray)

def crop(image, mask):
    img = image.copy()
    img[mask==0] = 0
    return img

def color_std(image):
	try:
		R = image[np.where(image[:,:,0] != 0) and np.where(image[:,:,1] != 0) and np.where(image[:,:,2] != 0)][:,0]
		G = image[np.where(image[:,:,0] != 0) and np.where(image[:,:,1] != 0) and np.where(image[:,:,2] != 0)][:,1]
		B = image[np.where(image[:,:,0] != 0) and np.where(image[:,:,1] != 0) and np.where(image[:,:,2] != 0)][:,2]
		color_std = (np.std(R) + np.std(G) + np.std(B)) /3
	except:
		color_std = 'NA'
	return color_std