
from tifffile import imwrite
import numpy as np
from itertools import combinations
from statistics import mean
from scipy.ndimage import binary_closing as nd_binary_closing
from skimage.morphology import binary_dilation
import os
from skimage.filters import threshold_minimum
from scipy.ndimage import find_objects
import networkx as nx
import pickle
from skimage.measure import regionprops_table, regionprops
import pandas as pd
from os import path
from labutils import LabUtils

def __increment_slice(_slice, increment, max_value):
    return slice(max(_slice.start - increment, 0), min(_slice.stop + increment, max_value), _slice.step)

def calculate_incremented_bounding_boxes(watershed_labels, increment = 1):

    """ calculate incremented bounding boxes
    Args:
        watershed_labels(array): labelled image of cell positions
        increment (int): factor by how much bounding boxes should be incremented

    Returns: 
        incremented_b_boxes (): A list of tuples, with 3 slices each. Bounding boxes are incremented by one voxel in each dimension. Labels correspond to the indexes of the bounding box.
    """

    #calculate bounding boxes
    b_boxes = find_objects(watershed_labels)
    im_shape = watershed_labels.shape

    #increment bounding boxes
    incremented_b_boxes = {}
    for i, b_box in enumerate(b_boxes):
        if b_box is not None:
            # new_entry = (slice(max(b_box[0].start -increment, 0), min(b_box[0].stop +increment, x_max-1), b_box[0].step), \
            #              slice(max(b_box[1].start -increment, 0), min(b_box[1].stop +increment, y_max-1), b_box[1].step), \
            #              slice(max(b_box[2].start -increment, 0), min(b_box[2].stop +increment, z_max-1), b_box[2].step))
            new_entry = new_entry = (__increment_slice(b_box[0], increment, im_shape[0]-1), \
                                     __increment_slice(b_box[1], increment, im_shape[1]-1), \
                                     __increment_slice(b_box[2], increment, im_shape[2]-1))
            #adapt index to match the label of the object
            incremented_b_boxes[i+1] = new_entry

    return incremented_b_boxes


def extract_touching_surfaces(watershed_labels, output_folder):
    
    """ find membranes between all predicted cells (based on watershed labels)
    Args:
        watershed_labels (3d array): watershed labels as 3D matrix of interegers, where 0 is background

    Returns:
        membrane_labels (3D array): 3D matrix of integers, showing all overlapping membranes
        mapp_mem (dict): dictionary that maps membrane id to tuples of neighboring cells, as labelled in watershed image

    """
    
    #create list of all labels and remove the background label = 0
    unique_labels = list(np.unique(watershed_labels))
    if 0 in unique_labels:
        unique_labels.remove(0)
        
    #calculate bounding boxes for all cells, dilate bounding box by 1 voxel
    incremented_b_boxes = calculate_incremented_bounding_boxes(watershed_labels)
    
    #initiate output_image and mapper dictionary
    membrane_labels = np.zeros_like(watershed_labels)
    mapp_mem = {}
    
    #iterate through labels and find neighbors, calculate overlap
    membrane_id = 1
    for label in unique_labels:
        b_box = incremented_b_boxes[label]
        sub_image = watershed_labels[b_box]
        
        #find neighbours in bounding box
        mask_dilation = binary_dilation(sub_image == label)
        neighbors = np.unique(sub_image[mask_dilation])

        #put all overlaps in one image as labels, map labels to cell combinations in mapper dictionary   
        for neighbor in neighbors:
            if not neighbor in (label, 0):
                n_mask = mask_dilation & (sub_image == neighbor)
                if (n_mask.sum() > 0) and ((label, neighbor) not in mapp_mem.keys()):
                    membrane_labels[b_box][n_mask] = membrane_id
                    mapp_mem[(label, neighbor)] = membrane_id
                    mapp_mem[(neighbor, label)] = membrane_id
                    membrane_id += 1

    #some very small membrane fragment might have gotten covered by bigger ones, so we make sure only existing membranes exist in the mapper
    final_membranes = np.unique(membrane_labels) 
    lost_membranes = set(mapp_mem.values()).difference(set(final_membranes))
    if len(lost_membranes) > 0:
        mapp_mem = {pair: mem_id for pair, mem_id in mapp_mem.items() if mem_id in set(final_membranes)}
   
        
    #output_path = path(output_folder)

    #save interfaces image
    imwrite(output_folder + "all_touching_membrane_labels.tif", membrane_labels)

    #save mapper dictionary
    with open(output_folder + 'mapper_pairs_mem_id.pickle', 'wb') as handle:
        pickle.dump(mapp_mem, handle, protocol=pickle.HIGHEST_PROTOCOL)


                
    return membrane_labels, mapp_mem


def volume_ratio_after_closing(interface_image, mapper, iterations = 2):
    
    """
    calculate ratio of the volumes of each interface before and after using scipy.ndimage.binary_closing 
    --> volume increase points to abrupt shape changes in membrane = not an actual membrane
    Args:
        interface_image
        mapper (dict): dictionary mapping each membrane to the pair of cells it seperates: {(label cell1, label cell2) : membrane_id}
        iterations (int): how many times should the closing algorithm be applied
    
    """
    output = {}
    
    for label in set(mapper.values()):
        
        interface = interface_image == label
        if interface.sum() == 0:
            continue
            
        interface_closed = nd_binary_closing(interface, iterations = iterations)
        
        vol_before = interface.sum()
        vol_after = interface_closed.sum()
        ratio = vol_after/vol_before
        
        output[label] = ratio
        
    return output


def make_training_dataframe(true_membrane_image, false_membrane_image, mapper):
    """ calculate properties of membranes and store in a dataframe to train the LDA classifier
    Args:
        true_membrane_image (array): labelled image (3D array of integers) of membrane positions that were manually scored as true (ground truth)
        false_membrane_image (array): labelled image (3D array of integers) of membrane positions that were manually scored as false (ground truth)
        mapper (dict): dictionary mapping each membrane to the pair of cells it seperates: {(label cell1, label cell2) : membrane_id}

    Returns: membrane_df (Dataframe): Dataframe with columns for label and the labels characteristics, 
                                      as well as membrane_status, indicating the manual annotation of the membrane
    """

    #calculate characteristics of true membranes
    true_ratios = volume_ratio_after_closing(true_membrane_image, mapper)
    props_true = regionprops_table(true_membrane_image, properties=('label',
                                                 'area',
                                                 'axis_major_length', 
                                                'equivalent_diameter_area',
                                                    'extent'))
    props_true_df = pd.DataFrame(props_true)
    props_true_df.set_index('label', inplace = True, drop = True)
    
    true_df = pd.DataFrame(index = true_ratios.keys(), columns = ["membrane_status", "vol_ratio"])
    true_df['vol_ratio'] = true_ratios.values()
    true_df['membrane_status'] = True
    true_df = pd.concat([true_df, props_true_df], axis = 1)
    
    b_boxes = calculate_incremented_bounding_boxes(true_membrane_image, increment = 1)
    true_df["average_distances_between_z_planes"] = None
    
    for label in true_df.index:
        roi = true_membrane_image[b_boxes[label]]
        roi_single_mem = np.zeros_like(roi)
        roi_single_mem[roi == label] = 1
        dist_list = []
        if roi_single_mem.shape[0] > 1:
            for plane in range(roi_single_mem.shape[0]-1): 
                dist_list.append(LabUtils.slice_morphological_distance(roi_single_mem, plane))
        else:
            dist_list.append(0)
        true_df.loc[label, "average_distances_between_z_planes"] = np.mean(dist_list)   
    

    #calculate characteristics of false membranes
    false_ratios = volume_ratio_after_closing(false_membrane_image, mapper)
    props_false = regionprops_table(false_membrane_image, properties=('label',
                                                 'area',
                                                 'axis_major_length', 
                                                'equivalent_diameter_area',
                                                    'extent'))
    props_false_df = pd.DataFrame(props_false)
    props_false_df.set_index('label', inplace = True, drop = True)
    
    false_df = pd.DataFrame(index = false_ratios.keys(), columns = ["membrane_status", "vol_ratio"])
    false_df['vol_ratio'] = false_ratios.values()
    false_df['membrane_status'] = False
    false_df = pd.concat([false_df, props_false_df], axis = 1)

    b_boxes = calculate_incremented_bounding_boxes(false_membrane_image, increment = 1)
    false_df["average_distances_between_z_planes"] = None
    
    for label in false_df.index:
        roi = false_membrane_image[b_boxes[label]]
        roi_single_mem = np.zeros_like(roi)
        roi_single_mem[roi == label] = 1
        dist_list = []
        if roi_single_mem.shape[0] > 1:
            for plane in range(roi_single_mem.shape[0]-1): 
                dist_list.append(LabUtils.slice_morphological_distance(roi_single_mem, plane))
        else:
            dist_list.append(0)
        false_df.loc[label, "average_distances_between_z_planes"] = np.mean(dist_list)   
    
    
    membrane_df = pd.concat([true_df, false_df], axis = 0)
    membrane_df.reset_index(inplace = True)
    membrane_df.rename(columns = {"index" : "label"}, inplace = True)
    membrane_df.to_csv("embryo_membrane_classification_df.txt", sep = "\t", index = None) 
    print("saved merged dataframe")


    return membrane_df


def make_classification_dataframe(interface_image, mapper):
    
    """ calculate properties of membranes and store in a dataframe to use the pretrained the LDA classifier
    Args:
        interface_image (array): labelled image (3D array of integers) of membrane positions
        mapper (dict): dictionary mapping each membrane to the pair of cells it seperates: {(label cell1, label cell2) : membrane_id}
    Returns: membrane_df (Dataframe): Dataframe with columns for label and the labels characteristics, same as in make_training_dataframe
    """

    vol_ratios = volume_ratio_after_closing(interface_image, mapper)
    print("calculated volumes")
    props = regionprops_table(interface_image, properties=('label',
                                                 'area',
                                                 'axis_major_length', 
                                                'equivalent_diameter_area',
                                                    'extent'))
    print("calculated regionprops")
    props_df = pd.DataFrame(props)
    props_df.set_index('label', inplace = True, drop = True)
    
    vol_df = pd.DataFrame(index = vol_ratios.keys(), columns = ["vol_ratio"])
    vol_df['vol_ratio'] = vol_ratios.values()
    membrane_df = pd.concat([vol_df, props_df], axis = 1)

    b_boxes = calculate_incremented_bounding_boxes(interface_image, increment = 1)
    membrane_df["average_distances_between_z_planes"] = None
   

    for label in membrane_df.index:
        roi = interface_image[b_boxes[label]]
        roi_single_mem = np.zeros_like(roi)
        roi_single_mem[roi == label] = 1
        dist_list = []
        if roi_single_mem.shape[0] > 1:
            for plane in range(roi_single_mem.shape[0]-1): 
                dist_list.append(LabUtils.slice_morphological_distance(roi_single_mem, plane))
        else:
            dist_list.append(0)
        membrane_df.loc[label, "average_distances_between_z_planes"] = np.mean(dist_list)   

    membrane_df.reset_index(inplace = True)
    membrane_df.rename(columns = {"index" : "label"}, inplace = True)

    
    membrane_df.to_csv("embryo_membrane_classification_df.txt", sep = "\t") 
    print("saved merged dataframe")


    return membrane_df



def run_classifications(membrane_df, path_to_classifier):
    
    """
    classify membranes as true or false by using a pre-trained LDA classifier
    Args: 
        membrane_df (DataFrame): Dataframe describing metrics of membranes, made using make_classification_dataframe function
        path_to_classifier (str): path to the classifier trained on ground truth data (.pickle)
    Returns: classification (array): 3D boolean arrayindicating whether a membrane with the same index in membrane_df was classified tue or false

    
    """
    #load model
    loaded_model = pickle.load(open(path_to_classifier, 'rb'))
    
    #define parameters for classification
    X = np.array(membrane_df[["vol_ratio", "area", 'axis_major_length', 'equivalent_diameter_area', 'extent', "average_distances_between_z_planes"]])
    
    classification = loaded_model.predict(X)

    return classification


def calculate_class_probabilities(membrane_df, path_to_classifier):
    
    """
    classify membranes as true or false by using a pre-trained LDA classifier
    Args: 
        membrane_df (DataFrame): Dataframe describing metrics of membranes
        path_to_classifier (str): path to the classifier trained on ground truth data (.pickle)
    Returns: class_probs(array(n_samples, n_classes)): Array of floats = tuple of probabilities to belong to each class per sample
    
    """
    #load model
    loaded_model = pickle.load(open(path_to_classifier, 'rb'))
    
    #define parameters for classification
    X = np.array(membrane_df[["vol_ratio", "area", 'axis_major_length', 'equivalent_diameter_area', 'extent', "average_distances_between_z_planes"]])
    
    class_probs = loaded_model.predict_proba(X)

    return class_probs    



def seperate_membranes(membrane_df, mapper, interface_image, path_to_classifier, cut_off_probability = None):
    
    """seperates all interfaces in true and false based on a trained classifier
    Args:
        volume_ratio_dictionary (dict): {membrane_id: volume_ratio}
        mapper (dict): dictionary mapping each membrane to the pair of cells it seperates: {(label cell1, label cell2) : membrane_id}
        interface_image(array): array of integers (labels) showing positions of membranes
        path_to_classifier (str): path to the classifier trained on ground truth data (.pickle)
        cut_off_value (float): probability value for spliting membranes into true and false based on their volume_ratio
    Returns: 
        true_membrane_image, false_membrane_image (3D array): array of integers with positions of true and false membranes, labelled as in the input interface_image
    
    """

    if cut_off_probability == None:
        classification = run_classifications(membrane_df, path_to_classifier)

        #translate classification into labels
        true_indices = list((np.where(classification == True))[0])
        true_labels = list(membrane_df.iloc[true_indices, 0])
        false_indices = list((np.where(classification == False))[0])
        false_labels = list(membrane_df.iloc[false_indices, 0])

        true_pairs = [pair for pair, index in mapper.items() if index in set(true_labels)]
        false_pairs = [pair for pair, index in mapper.items() if index in set(false_labels)]

        #create an image with only true or only false membranes
        true_membrane_image = np.zeros_like(interface_image)
        for label in true_labels:
            true_membrane_image[interface_image == label] = label

        false_membrane_image = np.zeros_like(interface_image)
        for label in false_labels:
            false_membrane_image[interface_image == label] = label

    else:
        #calculate probabilities for membranes to be true or false
        class_probs = calculate_class_probabilities(membrane_df, path_to_classifier)

        striclty_true_indices = list((np.where(class_probs[:,1] > cut_off_probability))[0])
        true_labels = list(membrane_df.iloc[striclty_true_indices, 0])
        striclty_false_indices = list((np.where(class_probs[:,0] > cut_off_probability))[0])
        false_labels = list(membrane_df.iloc[striclty_false_indices, 0])

        true_pairs = [pair for pair, index in mapper.items() if index in set(true_labels)]
        false_pairs = [pair for pair, index in mapper.items() if index in set(false_labels)]

        #create an image with only true or only false membranes
        true_membrane_image = np.zeros_like(interface_image)
        for label in true_labels:
            true_membrane_image[interface_image == label] = label

        false_membrane_image = np.zeros_like(interface_image)
        for label in false_labels:
            false_membrane_image[interface_image == label] = label


            
    return true_membrane_image, false_membrane_image, true_pairs, false_pairs




def find_connected_components(false_pairs_list): 
    
    """find connected components among pairs of cells to be merged, create sets of labels that are to be merged into one cell
    Args:
        false_labels_list (list): list with all pairs of cells that need to be merged
    Returns:
        cc_list (list): list of sets of cells that are to be merged
    """

    G = nx.Graph()
    for edge in false_pairs_list:
        G.add_edge(*edge)
        
    cc_list = list(nx.connected_components(G))
    
    return cc_list


def merge_labels_with_false_membranes(false_pairs_list, original_watershed_labels, output_name):
    
    """combine labels that are seperated by interfaces that are detected as false membranes
    Args:
        false_labels_list (list): list with all pairs of cells that need to be merged
        original_watershed_labels(array): array of integers, labelled image
        output_name(str): name for saving the new watershed image with merged cells
        
    Returns:
        merged_watershed (3D array): new image with merged cells, according to labels in merging_dictionary
    """
    
    #find connected components among cells that need to be merged (group all labels for merging into one big cell)
    cc_list = find_connected_components(false_pairs_list)
    #create image with merged cells based on sets of cells and original watershed image
    merged_watershed = original_watershed_labels.copy()
    for cell_set in cc_list:
        smallest_id = min(cell_set)
        for label in cell_set:
            merged_watershed[merged_watershed == label] = smallest_id
            
    imwrite(output_name, merged_watershed)
    
    return merged_watershed
                
    
def measure_cell_volumes(labels_image, voxel_size = (0.173, 0.173, 1), remove_zero = True):
    
    """ measures the volume in voxels of each labeled component in an input image
    Args:
        labels_image (array): 3D array of integers indicating positions of components
        remove_zero (boolean): True or False, whether label zero should be considered and measured or not 
    Returns: 
        volumes_dictionary (dict): dictionary where keys are labels and values the respective volumes in voxels
    """

    volumes_in_voxels = {}
    volumes_in_um3 = {}
    voxel_volume = voxel_size[0]*voxel_size[1]*voxel_size[2]

    for label in np.unique(labels_image):
        if remove_zero == True and not label == 0:
            volume = (labels_image == label).sum()
            volumes_in_voxels[label] = volume
            volumes_in_um3[label] = volume * voxel_volume
        elif remove_zero == False:
            volume = (labels_image == label).sum()
            volumes_in_voxels[label] = volume
            volumes_in_um3[label] = volume * voxel_volume

    return volumes_in_voxels, volumes_in_um3


def calculate_volume_cutoff_percentage(labels_image, desired_drop_percentage=5):

    """calculates the volumes and the range of volumes in an image with labeled components and a cut-off at a specific percentage within that range
    Args: 
        labels_image (array): 3D array of integers indicating positions of components
        desired_drop_percentag (int): percentage for cut_off calculatipon, set to 5 --> cut-off will indicate where the bottom 5% of the volume range lies
    Returns:
        cut_off (int): value that indicates where the desired percentage of the volume range lies
        volumes (dict): dictionary where keys are labels and values the respective volumes in voxels

    """

    volumes = measure_cell_volumes(labels_image)
    vol_range = max(volumes.values()) - min(volumes.values())
    cut_off = min(volumes.values()) + (vol_range/100*desired_drop_percentage)

    return cut_off, volumes




def drop_cells_below_percentage_cut_off(labels_image, desired_drop_percentage=5):
    
    """ removes small cells from labeled image, "small" is defined by calculate_volume_cutoff_percentage and the desired_drop_percentage
    Args: 
        labels_image (array): 3D array of integers indicating positions of components
        desired_drop_percentag (int): percentage for cut_off calculatipon, set to 5 --> cut-off will indicate where the bottom 5% of the volume range lies
    Returns:
        cut_image (array): 3D array of integers indicating positions of cells
        cut_off (int): value that indicates where the desired percentage of the volume range lies
        volumes (dict): dictionary where keys are labels and values the respective volumes in voxels

    """

    size_cut_off, volumes = calculate_volume_cutoff(labels_image, desired_drop_percentage)
    remaining_labels = [label for label, volume in volumes.items() if volume > cut_off]

    cut_image = np.zeros_like(labels_image)    
    for label in remaining_labels:
        cut_image[labels_image == label] = label

    
    return cut_image, cut_off, volumes




def drop_cells_below_um3_cut_off(labels_image, cut_off = 5000, voxel_size = (0.173, 0.173, 1)):
    
    """ removes small cells from labeled image, "small" is defined by the cut_off parameter. 
    5000 µm3 was chosen as a default, as it is highly unlikely for a cell to be smaller than this in pre-implantation mouse embryos.
    Args: 
        labels_image (array): 3D array of integers indicating positions of components
        cut_off (int): absolute volume threshold value in µm3
    Returns:
        cut_image (array): 3D array of integers indicating positions of cells

    """

    vol, vol_um3 = measure_cell_volumes(labels_image, voxel_size)
    remaining_cells = [label for label, volume in vol_um3.items() if volume > cut_off]
    cut_image = np.zeros_like(labels_image)
    for label in remaining_cells:
        cut_image[labels_image == label] = label
        
    return cut_image
    

def find_most_prominent_neighbour(labels_image, labels_list=None):

    """finds most promionent neighbors (as in other labels that covers most of the cells surface) for labels in a labeled image 
    Args: 
        labels_image (array): 3D array of integers indicating positions of components
        labels_list (list): optionally a list of labels can be provided for which most prominent neighbors will be found. All labels must be in labels_image.

    Returns:
        most_prom_n_dict (dict): dictionary where keys are input labels and values are the labels of their most prominent neighbors

    """


    most_prom_n_dict = {}
    if labels_list == None:
        labels_list = np.unique(labels_image)
    for label in labels_list:
        if label != 0:
            sub_im = labels_image == label
            dilated_mask = (binary_dilation(sub_im)) ^ sub_im
            volumes, vol_um3 = measure_cell_volumes((labels_image[dilated_mask]), remove_zero = False)
            most_prom_n_dict[label] = [label for label, volume in volumes.items() if volume == max(volumes.values())][0]
    
    return most_prom_n_dict



def remove_small_outside_cells_percentage(labels_image, desired_drop_percentage = 5):

    """remove small cells that have background as most prominent neighbor
    Args: 
        labels_image (array): 3D array of integers indicating positions of components
        desired_drop_percentag (int): percentage for cut_off calculatipon, set to 5 --> cut-off will indicate where the bottom 5% of the volume range lies
    Returns:
        cut_image (array): 3D array of integers indicating positions of components, same as input minus dropped cells
        cut_off (int): value that indicates where the desired percentage of the volume range lies --> shows which cells will be considered small
    """

    cut_off, volumes = calculate_volume_cutoff_percentage(labels_image, desired_drop_percentage)
    small_labels = [label for label, volume in volumes.items() if volume < cut_off]
    
    prom_neighbor_dict = find_most_prominent_neighbour(labels_image, small_labels)
    final_remove = [label for label, neighbor in prom_neighbor_dict.items() if neighbor == 0]
    cut_image = labels_image.copy() 
    for label in final_remove:
        cut_image[labels_image == label] = 0
    
    return cut_image, cut_off



def remove_small_outside_cells_um3(labels_image, cut_off = 5000, voxel_size = (0.173, 0.173, 1)):

    """remove small cells that have background as most prominent neighbor
    Args: 
        labels_image (array): 3D array of integers indicating positions of components
        cut_off (int): absolute volume threshold value in µm3
    Returns:
        cut_image (array): 3D array of integers indicating positions of components, same as input minus dropped cells
    """
    vol, vol_um3 = measure_cell_volumes(labels_image, voxel_size)
    small_labels = [label for label, volume in vol_um3.items() if volume < cut_off]
    
    prom_neighbor_dict = find_most_prominent_neighbour(labels_image, small_labels)
    final_remove = [label for label, neighbor in prom_neighbor_dict.items() if neighbor == 0]
    cut_image = labels_image.copy() 
    for label in final_remove:
        cut_image[labels_image == label] = 0
    
    return cut_image

def remove_outside_cells_until_stability(labels_image, cut_off = 5000, voxel_size = (0.173, 0.173, 1)):

    """ iterate through the removal of small cells that have background as most prominent neighbor until the cell number does not change anymore
    Args: 
        labels_image (array): 3D array of integers indicating positions of components
        cut_off (int): absolute volume threshold value in µm3
    Returns:
        cut_image (array): 3D array of integers indicating positions of components, same as input minus dropped cells
    """

    cell_number_before = np.count_nonzero(np.unique(labels_image))
    cut_image = remove_small_outside_cells_um3(labels_image, cut_off = 5000, voxel_size = (0.173, 0.173, 1))
    cell_number_after = np.count_nonzero(np.unique(cut_image))
    
    while cell_number_before > cell_number_after:
        cell_number_before = cell_number_after
        cut_image = remove_small_outside_cells_um3(labels_image, cut_off = 5000, voxel_size = (0.173, 0.173, 1))
        cell_number_after = np.count_nonzero(np.unique(cut_image))

    return cut_image




def find_false_probability_for_membrane(interfaces, predictions_dataframe, path_to_classifier):

    """find probabilities for a membrane to be false according to a pre-trained LDA classifier
    Args: 
        interfaces (array): 3D array of integers indicating positions of membranes
        predictions_dataframe (DataFrame): Dataframe generated with make_classification_dataframe funtion
        path_to_classifier (str): path to the classifier trained on ground truth data (.pickle)
    Returns:
        false_prob_dict (dict): dictionary where keys are membrane labels and values are probabilities for that membrane to be false
    """
    clf = pickle.load(open(path_to_classifier, "rb"))
    X = np.array(predictions_dataframe[["vol_ratio", "area", 'axis_major_length', 'equivalent_diameter_area', 'extent', "average_distances_between_z_planes"]])
    probabilities = clf.predict_proba(X)
    
    false_prob_dict = {}
    for label in np.unique(interfaces):
        if label != 0:
            index = predictions_dataframe[predictions_dataframe['label'] == label].index.values
            prob = probabilities[index]
            false_prob_dict[label] = prob[0][0]
            
    return false_prob_dict




def find_best_merge_for_small_cells(labels_image, desired_drop_percentage, mapper, interfaces, predictions_dataframe, path_to_classifier):

    """find small cells and the neighbour who is most likely to be their correct merging partner
    Args: 
        labels_image (array): 3D array of integers indicating positions of cells
        desired_drop_percentage (int): percentage in volume range that is taken as a cut-off for considering a cell to be too small
        mapper (dict): dictionary mapping each membrane to the pair of cells it seperates: {(label cell1, label cell2) : membrane_id}
        interfaces (array): 3D array of integers indicating positions of membranes
        predictions_dataframe (DataFrame): Dataframe generated with make_classification_dataframe funtion
        path_to_classifier (str): path to the classifier trained on ground truth data (.pickle)
    Returns:
        merging_dict (dict): dictionary where keys are membrane labels and values are the membranes across which they are most likely to need merging
    """

    cut_off, volumes = calculate_volume_cutoff_percentage(labels_image, desired_drop_percentage)
    small_labels = [label for label, volume in volumes.items() if volume < cut_off]
    
    false_prob_dict = find_false_probability_for_membrane(interfaces, predictions_dataframe, path_to_classifier)
    
    #find which membranes are most likely to be false
    merging_dict = {}
    for label in small_labels:
        #find interfaces
        touching_membranes = [membrane for pair, membrane in mapper.items() if label in pair]
        prob_touching = {label: prob for label, prob in false_prob_dict.items() if label in touching_membranes}
        most_likely_false_membrane = [membrane for membrane, prob in prob_touching.items() if prob == max(prob_touching.values())][0]
        merging_dict[label] = most_likely_false_membrane
          
    return merging_dict