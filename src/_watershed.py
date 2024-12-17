
import scipy.ndimage as nd
from skimage.filters import threshold_triangle
import morphsnakes as ms
from skimage.morphology import binary_erosion, binary_dilation, ball
import numpy as np
from skimage.morphology import h_minima
from skimage.segmentation import watershed as skwatershed




def gaussian_blurr(image, sigma_value, im_res = np.array((5.78, 1, 1))):
	"""applies nd.image gaussian filter to a 3D array while adjusting the resolution difference for z dimension

	Args:
		image (3D array): 3D array of gray values
		sigma (int): sigma for gaussian blurr 
		im_res(array): resolution difference in z, x, y

	
	returns: filtered_image (3D array)
	"""
	filtered_image = nd.gaussian_filter(image, sigma = sigma_value/im_res)

	return filtered_image


def binarize_embryo(image, sigma = 12):
	"""find approximate binary mask of the embryo

	Args:
		image (3D array): 3D array of gray values
		sigma (int): sigma for gaussian blurring of the input image

	Returns: im_outer_membrane (3D array): array approximating the outer membrane of the embryo
	"""
	filtered_image = gaussian_blurr(image, sigma)
	thresh = threshold_triangle(filtered_image)
	binary_embryo = filtered_image > thresh

	return binary_embryo



def create_morphsnakes_outline(image, binary_embryo, morphsnake_iterations = 40, erosion_iterations=3):
	if binary_embryo.shape == image.shape:
		ms_mask = ms.morphological_chan_vese(image, morphsnake_iterations, init_level_set = binary_embryo)
	else:
		ms_mask = ms.morphological_chan_vese(image, morphsnake_iterations)

	im_ms_eroded = binary_erosion(ms_mask)
	for step in range(erosion_iterations):
		im_ms_eroded = binary_erosion(im_ms_eroded)
        
	im_ms_dilated = binary_dilation(im_ms_eroded)
	for step in range(erosion_iterations):
		im_ms_dilated = binary_dilation(im_ms_dilated)
    
	im_outer_membrane = (np.subtract(im_ms_dilated, im_ms_eroded, dtype = int)).astype(bool)
    
	return im_outer_membrane


def normalize_percentile_range(image, perc_distance, bounds = [0, 1]):

	"""normalizes image within indicated percentile range

	Args:
		image (3D array): 3D array of gray values
		perc_distance (int): integer indicated where data should be capped
		bounds(list of two integers): two integers indicating the desired max and min value of the image (eg 0 to 1 or 0 to 255)

	Returns: im_norm_perc (3D array): 3D array of floats, containing gray values normalized within 
									  indicated percentile range of the opriginal image, with specific min and max values
	"""


	im_norm_perc = image.copy()
	upper_perc = np.percentile(im_norm_perc, 100-perc_distance)
	lower_perc = np.percentile(im_norm_perc, perc_distance)
   	
	im_norm_perc[im_norm_perc > upper_perc] = upper_perc
	im_norm_perc[im_norm_perc < lower_perc] = lower_perc
    
	if bounds == [0,1]:
		im_norm_perc = (im_norm_perc - im_norm_perc.min()) / (im_norm_perc.max() - im_norm_perc.min())
	else:
		im_norm_perc = ((im_norm_perc - im_norm_perc.min()) / (im_norm_perc.max() - im_norm_perc.min()))*255
          
	return im_norm_perc


def add_outer_membrane(image, ms_membrane):
	"""normalizes image within indicated percentile range

	Args:
		image (3D array): 3D array of gray values
		ms_membrane (3D array): array approximating the outer membrane of the embryo

	Returns: im_added_membrane (3D array): 3D array of gray values with signal (max gray value) for outer membrane
	"""

	im_added_membrane = image.copy()
	im_added_membrane[ms_membrane == True] = im_added_membrane.max()
    
	return im_added_membrane



def watershed_on_h_min(image, h, binary_embryo, sigma_gaussian = 2):

	"""finds local minima as seed points for watershed and runs watershed to detect cells in the embryo

	Args:
		image (3D array): 3D array of gray values
		h (int): h value as input for the h-minima function
		outside_mask (3D array): boolean array indicating the area around the embryo

	Returns: im_added_membrane (3D array): 3D array of gray values with signal (max gray value) for outer membrane
	"""
	if sigma_gaussian is None:
		mod_image = image
	else:
		mod_image = gaussian_blurr(image, sigma_gaussian)
	
	mod_image[binary_embryo == 0] = 0	
	h_min = h_minima(mod_image, h = h, footprint = ball(3))
	labels, num_features = nd.label(h_min)

	watershed_embryo = skwatershed(mod_image, labels)
	watershed_embryo[watershed_embryo == 1] = 0
    
	return watershed_embryo, num_features