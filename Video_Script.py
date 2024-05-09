import argparse
import os
import pygame
from tkinter import Image
from mmseg.datasets import DATASETS
import datasets
import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (
    get_dist_info,
    init_dist,
    load_checkpoint,
    wrap_fp16_model,
)
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.ndimage import gaussian_filter1d
import torch.nn.functional as F
from scipy.signal import savgol_filter
#from mmseg.models.segmentors.encoder_decoder import whole_inference
#from mmseg.datasets import replace_ImageToTensor as rep
from mmcv.parallel import collate, scatter
#from mmseg.models.segmentors.encoder_decoder import inference as inferr
import pandas as pd
from mmcv.utils import DictAction
from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader
from mmseg.apis.inference import inference_segmentor
# from mmseg.registry import DATASETS
# from mmseg.datasets.basesegdataset import BaseSegDataset
from mmseg.datasets import build_dataset as mmseg_build_dataset
from mmseg.datasets.builder import DATASETS as MMSEG_DATASETS

from mmdet.datasets import build_dataset as mmdet_build_dataset
from mmdet.datasets.builder import DATASETS as MMDET_DATASETS
from mmseg.models import build_segmentor
from mmseg.apis import inference

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (
    build_dataloader,
    #replace_ImageToTensor,
)
from mmdet.models import build_detector
import torchvision

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random 

from pathlib import Path
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from scipy.ndimage import convolve1d

from scalabel.label.io import save
from scalabel.label.transforms import bbox_to_box2d
from scalabel.label.typing import Dataset, Frame, Label 
from torch.profiler import profile, record_function, ProfilerActivity
from mmdet.apis import init_detector, inference_detector
from mmseg.apis import init_segmentor
import os.path as osp
from typing import List
from scipy.interpolate import splprep, splev
import time
from sklearn.metrics import mean_squared_error
import math

MODEL_SERVER = "https://dl.cv.ethz.ch/bdd100k/det/models/"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument("--configdet", help="test config file path")
    parser.add_argument("--configseg", help="test config file path")
    parser.add_argument("--sourceimages", help="image name")
    parser.add_argument("--imagefolder", help="image name")
    parser.add_argument(
        "--work-dir",
        help="the directory to save the file containing evaluation metrics",
    )
    parser.add_argument(
        "--aug-test", action="store_true", help="Use Flip and Multi scale aug"
    )
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Whether to fuse conv and bn, this will slightly increase"
        "the inference speed",
    )
    parser.add_argument(
        "--format-only",
        action="store_true",
        help="Format the output results without perform evaluation. It is"
        "useful when you want to format the result to a specific format and "
        "submit it to the test server",
    )
    parser.add_argument(
        "--format-dir", help="directory where the outputs are saved."
    )
    parser.add_argument("--show", action="store_true", help="show results")
    parser.add_argument(
        "--show-dir", help="directory where painted images will be saved"
    )
    parser.add_argument(
        "--show-score-thr",
        type=float,
        default=0.3,
        help="score threshold (default: 0.3)",
    )
    parser.add_argument(
        "--gpu-collect",
        action="store_true",
        help="whether to use gpu to collect results.",
    )
    parser.add_argument(
        "--tmpdir",
        help="tmp directory used for collecting results from multiple "
        "workers, available when gpu-collect is not specified",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument(
        "--opacity",
        type=float,
        default=0.5,
        help="Opacity of painted segmentation map. In (0, 1] range.",
    )
    parser.add_argument(
        "--options", nargs="+", action=DictAction, help="custom options"
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args

def CombineMasks (args,seg, det):
    #rgb_images = {args.sourceimages+'jpg.png': seg}
    #grayscale_images = {args.sourceimages+'_mask.png':det}

    #rgb_images
    # Define the RGB to Grayscale mapping
    rgb_to_grayscale_map = {
        #(86, 211, 219): 30,
        #(219, 94, 86): 50,
        (255, 0, 0) : 50,
        (0, 255, 0) : 30

    }
    grayscale_image = det
    corresponding_rgb_image = seg
    # Convert RGB images to Grayscale as per the mapping
    #for rgb_image_name, rgb_image in rgb_images.items():
    for rgb_color, grayscale_value in rgb_to_grayscale_map.items():
            mask = np.all(corresponding_rgb_image == np.array(rgb_color), axis=-1)
            corresponding_rgb_image[mask] = [grayscale_value]

    # Iterate through the grayscale images and update the corresponding RGB images
    #for grayscale_image_name, grayscale_image in grayscale_images.items():
        #corresponding_rgb_image_name = grayscale_image_name.replace('_mask.png', '.jpg.png')
        #if corresponding_rgb_image_name in rgb_images:
            #corresponding_rgb_image = rgb_images[corresponding_rgb_image_name]
    #this loop is taking the most time
    mask = grayscale_image != 0


    corresponding_rgb_image[mask, 0] = grayscale_image[mask]  # Set R channel
    corresponding_rgb_image[mask, 1] = grayscale_image[mask]  # Set G channel
    corresponding_rgb_image[mask, 2] = grayscale_image[mask]  # Set B channel
    #for i in range(grayscale_image.shape[0]):
        #for j in range(grayscale_image.shape[1]):
            #if grayscale_image[i, j] != 0:
                #corresponding_rgb_image[i, j] = grayscale_image[i, j]

    # Create the "Mask image" containing only masks of pixel values corresponding to 30 and 50 in the "Fused Image"
    mask_image = np.zeros_like(corresponding_rgb_image, dtype=np.uint8)
    mask_image[corresponding_rgb_image == 30] = 1
    mask_image[corresponding_rgb_image == 50] = 2

    colored = cv2.applyColorMap(mask_image, create_custom_colormap(3))

    return colored



def calculate_weighted_sum(prev_paths, current_path):
    weighted_sum = []
    
    for i in range(len(current_path)):
        weighted_value_sum = 0
        weight_sum = 0
        
        for j in range(len(prev_paths)):
            #print(str(i)+" "+str(len(current_path))+" "+str(j)+" "+str(len(prev_paths)))
            x1, y1 = current_path[i]
            x2, y2 = prev_paths[j][i]
            
            # Calculate the distance between the points
            distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            
            # Use inverse distance as the weight (you can adjust this formula)
            weight = 1.0 / (distance + 1e-6)  # Adding a small value to avoid division by zero
            
            weighted_value_sum += prev_paths[j][i][0] * weight
            weight_sum += weight
        
        # Calculate the weighted average for the x-coordinate
        weighted_x = weighted_value_sum / (weight_sum + 1e-6)  # Adding a small value to avoid division by zero
        weighted_sum.append((int(weighted_x), current_path[i][1]))
    
    return weighted_sum

def find_center_path_smooth_new(binary_mask, image, last_5_outs, max_iou, theta=None):

        # Assuming blueMask is given, and is a binary image where blue pixels are set to 1 and other pixels are set to 0.
        binary_image = binary_mask

        rows, cols, _ = binary_mask.shape
        center_locations = []
        
        for i in range(rows):
            row = binary_mask[i, :]
            white_pixels = np.where(row[:, 0] == 255)[0]

            # Example: Weighted average (here weights are the same, but you can change)
            weights = np.ones_like(white_pixels)  # replace with actual weights if needed
            if len(white_pixels) > 0:
                center_x = int(round(np.average(white_pixels, weights=weights)))
                center_locations.append((center_x, i))
                

                
                
        if len(center_locations) == 0:      
            return image, None, None, None, None, last_5_outs


        center_locations = np.array(center_locations)

        # center_locations = np.array(center_locations)[5:-1, :]

        xo = center_locations[:, 0]
        yo = center_locations[:, 1]

        x = xo.copy()
        y = yo.copy()
        
        
        OUTns=(x,y)
        xypoints = center_locations.copy()
        
        # Subsample x and y
        step_size = len(x) // 10

        if step_size == 0:
            step_size = 1
        x = x[::step_size]
        y = y[::step_size]
        
        subsampled_xypoints = xypoints[::step_size]
        # print("points:",subsampled_xypoints)

        # Ensure the last points are the same
        x[-1] = xo[-1]
        y[-1] = yo[-1]
        
        xw=x
        yw=y
        xnn=x.copy()

               
        # Calculate the average 'out' if there are previous 'outs' to average
        if len(last_5_outs) > 1:
            # Determine the number of points to smooth (last 30%)
            num_points_to_smooth = int(len(last_5_outs[0]) * 0.5)
            starting_index = len(last_5_outs[0]) - num_points_to_smooth
            
            # Validate if the starting index is not negative
            starting_index = max(0, starting_index)

            # Extract the points to be smoothed and the remaining points
            points_to_smooth = [xnn[starting_index:] for xnn in last_5_outs if len(xnn) >= num_points_to_smooth]
            remaining_points = [xnn[:starting_index] for xnn in last_5_outs if len(xnn) >= num_points_to_smooth]
            # print("points2smooth:", points_to_smooth)
            
            if points_to_smooth:
                # Calculate the average of the points to be smoothed
                avg_smoothed_points = [sum(col) / len(col) for col in zip(*points_to_smooth)]

                # Get the actual remaining points from the current frame (not averaged)
                actual_remaining_points = xnn[:starting_index]

                # Combine the actual remaining and smoothed points
                smoothed_out = np.concatenate((actual_remaining_points, avg_smoothed_points))
        else:
            smoothed_out = xnn
            
        
        last_5_outs.append(smoothed_out)
        
        new_x = []
        new_y = []

        x_min = np.max(y)
        #Curvature adjustment
        if theta != None:
                        theta = theta / 9.8 * 3.1
                        #print(theta)
                        

                        #print(smoothed_out.tolist())
                        #xs, ys = zip(*smoothed_out.tolist())
                        xs = smoothed_out
                        ys = yw

                        #for (x, y) in trajectory[0]:
                        for i in range(min(len(xs),len(ys))):
                            x = xs[i]
                            y = ys[i]
                            #y_prime = int(y + 30*(x - x_min) * velocity * np.sin(theta))
                            x_prime = int(x - 10*(x - x_min) * np.sin(theta))
                            #print("compare" + str(y) + " "+ str(y_prime))
                            #if y_prime >= 1280: y_prime = 1279
                            #if y_prime >= binary_mask.shape[1]: y_prime = binary_mask.shape[1] - 1
                            if x_prime >= binary_mask.shape[0]: x_prime = binary_mask.shape[0] - 1
                            #center_points[i] = (x, y_prime)
                            #y_prime = y

                            #if(binary_mask[x,y_prime] != (0,0,0)).all():
                            
                            tolerance = 0
                            if 0 <= y < binary_mask.shape[1] and 0 <= x_prime < binary_mask.shape[0]:
                               
                                #if (np.abs(binary_mask[x_prime, y] - (255, 0, 0)) > tolerance).any():
                                if np.array_equal(binary_mask[y, x_prime], [255, 0, 0]):

                                    #new_center_points.append((x_prime, y))
                                    new_x.append(x_prime)
                                    new_y.append(y)
                                    


                            else:

                                print("Indices out of bounds:", x_prime, y)
        
        
        #print(new_center_points[30])
        if len(new_x) >= 1:
            smoothed_out = np.array(new_x)
            y = np.array(new_y)

        # Smooth using Savitzky-Golay filter
        window_length = 5
        if window_length > len(smoothed_out):
            window_length = len(smoothed_out)
        polyorder = 7
        if polyorder >= window_length:
            # Adjust window_length to be greater than polyorder
            polyorder = window_length - 1

        xs = savgol_filter(smoothed_out, window_length, polyorder)

        # Further smooth using moving average
        window_size = len(xs) // 4
        if window_size == 0:
            window_size = len(xs)  # or choose another appropriate default value

        smooth_data = np.convolve(xs, np.ones(window_size)/window_size, mode='valid')
        if len(smooth_data) == 1 and len(xs) > 0:
            new_value = 2 * smooth_data[0] - xs[-1]
            smooth_data = np.array([smooth_data[0], new_value])


        # Interpolation back to the original length
        xq = np.linspace(0, len(smooth_data)-1, len(xo))
        cs = CubicSpline(range(len(smooth_data)), smooth_data)
        interpolated_smooth_data = cs(xq)

        # Round to nearest integer
        interpolated_smooth_data = np.round(interpolated_smooth_data)
        x = interpolated_smooth_data
        y = yo

        # print(out)
        
        
        smoothing_factor= 0.4
        
        smoothing_factor = min(max(smoothing_factor, 0), 1)

        # Calculate the y-values on the straight line
        x_start, x_end = x[0], x[-1]
        x_line = np.linspace(x_start, x_end, len(x))

        # Smoothing the y-coordinates
        x_smoothed = np.round(smoothing_factor * x_line + (1 - smoothing_factor) * x)
            


        n = len(x_smoothed)
        if n >= 15:
            first_30_percent_index = int(n * 0.3)
            last_30_percent_index = int(n * 0.7)
            last_30_percent_index2 = int(n * 0.5)

            # First 30% points
            x_first = np.linspace(0, first_30_percent_index, first_30_percent_index)
            y_first = x_smoothed[:first_30_percent_index]

            spline_first = InterpolatedUnivariateSpline(x_first, y_first, k=3)
            x_new = np.linspace(0, first_30_percent_index, n)
            y_first_extended = spline_first(x_new)

            # Last 30% points
            x_last = np.linspace(last_30_percent_index, n-1, n - last_30_percent_index)
            y_last = x_smoothed[last_30_percent_index:]
            spline_last = InterpolatedUnivariateSpline(x_last, y_last, k=3)
            x_new = np.linspace(last_30_percent_index, n-1, n)
            y_last_extended = spline_last(x_new)

            # Averaging
            x_smoothed1 = np.round(y_first_extended + y_last_extended) / 2

            x_smoothed2 = np.concatenate((x_smoothed[:last_30_percent_index2], x_smoothed1[last_30_percent_index2:]))
        #elif n > 5: 
            #x_smoothed2 = InterpolatedUnivariateSpline(x_smoothed, y, k=3)
        else:
            x_smoothed2 = x_smoothed



        def smooth_array(x):
            sigma = 20  # Standard deviation for Gaussian kernel
            return gaussian_filter1d(x, sigma)

        x_smoothed2 = smooth_array(x_smoothed2)
            










        out = (x_smoothed2, y)
        smoothed_out = out
        #if max_iou <= 0.75:
            #smoothed_out = last_5_CompPaths[-1]
        #else: 
            #smoothed_out = out
            #last_5_CompPaths.append(smoothed_out)


        # # Calculate the average 'out' if there are previous 'outs' to average
        # if len(self.last_5_outs) > 1:
            # all_x = [out[0] for out in self.last_5_outs if len(out[0]) == len(self.last_5_outs[0][0])]
            # all_y = [out[1] for out in self.last_5_outs if len(out[1]) == len(self.last_5_outs[0][1])]

            # if all_x and all_y:
                # avg_x = [sum(col) / len(col) for col in zip(*all_x)]
                # avg_y = [sum(col) / len(col) for col in zip(*all_y)]
                # smoothed_out = (avg_x, avg_y)
            # else:
                # smoothed_out = out
        # else:
            # smoothed_out = out


        # Store the 'out' in the deque
        # self.last_5_outs.append(smoothed_out)
        # print(smoothed_out)
        # print(out[0])
        # print(out[1])

        # color_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
        #if self.reverse_color_channels:
            #color_mask = image[:, :, ::-1]  # Convert BGR to RGB
        #else:
        color_mask = image        

        prev_point = None
        for i in range(len(smoothed_out[0])):
            point = (int(smoothed_out[0][i]), int(smoothed_out[1][i]))
            if np.array_equal(binary_mask[point[1],point[0]], [255, 0, 0]):
                cv2.circle(color_mask, point, 2, (0, 255, 0), -1)

                if prev_point is not None:
                    cv2.line(color_mask, prev_point, point, (0, 255, 0), 1)
                prev_point = point
        
        return color_mask, smoothed_out, xw, yw, OUTns, last_5_outs

def GeneratePath(mask, OrigImg, PrevPaths = [], theta = None):
            #egocolor = np.array([86, 211, 219])
            center_points_list = []
            # Find the rows containing at least one non-zero pixel
            # Create a boolean mask indicating where the desired color is present
            #color_mask = np.all(mask == egocolor, axis=2)

            # Find the rows containing the desired color
            #non_zero_rows = np.any(color_mask, axis=1)
            
            non_zero_rows = np.any(mask, axis=1)

            # Find the row indices of non-zero rows
            non_zero_row_indices = np.where(non_zero_rows)[0]
            non_zero_row_indices = np.unique(non_zero_row_indices)
            # Compute center points for each non-zero row
            #dist = random.randint(20,100) #add eq: x1 + v*tau
            dist = 0
            xstart = 0
            xpos = 0
            if len(non_zero_row_indices)>0:
                xstart = non_zero_row_indices[-1]
                xpos = xstart + dist 
                #non_zero_row_indices = non_zero_row_indices[:len(non_zero_row_indices) - int(2*dist/3)]
                non_zero_row_indices = non_zero_row_indices[:len(non_zero_row_indices) - 20]
            center_points = []
            #print("2")
            for idx, row in enumerate(non_zero_row_indices):

                if idx < 10 or idx >= len(non_zero_row_indices) - 20:
                    continue

                # Find the column indices of non-zero pixels in the current row
                non_zero_cols = np.where(mask[row] > 0)[0]

                # Ignore the row if the number of mask pixels is less than 50
                if len(non_zero_cols) < 50:
                    continue

                # Compute the center point as the average of the column indices
                if idx >= len(non_zero_row_indices) - 50:
                    center_col = np.mean(non_zero_cols)
                else:
                    center_col = non_zero_cols[len(non_zero_cols) // 2]

                center_points.append((row, int(center_col)))
            center_points_list.append(center_points)

            
            #print(center_points_list)
            
            if len(center_points_list[0])>1:
                #print(len(center_points_list[0]))
                x_min = center_points_list[0][0][-1]
                for i, center_points in enumerate(center_points_list):
                    
                    if not center_points:
                        continue
                    #adjust for curvature

                    #theta = joysticks[1].get_axis(0) / 9.8 * 3.1
                    #print(theta)
                    new_center_points = []
                    if theta != None:
                        velocity = 1
                        for (x, y) in center_points:
    
                            y_prime = int( y + 30*(x - x_min) * velocity * np.sin(theta))
                            #print("compare" + str(y) + " "+ str(y_prime))
                            if y_prime >= 1280: y_prime = 1279
                            #center_points[i] = (x, y_prime)
                            #y_prime = y
                            if(mask[x,y_prime] != (0,0,0)).any():
                                new_center_points.append((x, y_prime))
                        #print(i)
                        center_points_list[i] = new_center_points

                            

                    #if new_center_points:
                        #rows, cols = zip(*new_center_points)
                    #else:
                    rows, cols = zip(*center_points)
                    #if len(rows)>=2:
                    cs = CubicSpline(rows, cols)
                    rows_interp = np.linspace(rows[0], rows[-1], num=30)  # You can adjust the number of points as needed
                    cols_interp = cs(rows_interp)
                    
                    interp_center_points = [(int(row), int(col)) for row, col in zip(rows_interp, cols_interp)]

                    center_points_list[i] = interp_center_points
                #print("4")
                # Smooth the interpolated center points using moving average filtering
                window_size = 10  # Adjust this value for desired smoothing
                for i, center_points in enumerate(center_points_list):
                    if not center_points:
                        continue

                    rows, cols = zip(*center_points)
                    smoothed_cols = convolve1d(cols, np.ones(window_size) / window_size, mode='nearest')
                    smoothed_center_points = [(int(row), int(col)) for row, col in zip(rows, smoothed_cols)]
                    center_points_list[i] = smoothed_center_points
                # Default color (e.g., red) for cases other than 0 and 1
                default_color = (0, 0, 255)  # Red color (BGR format)

                #Average the path over previous frames
                #print(PrevPaths)
                if PrevPaths and len(center_points_list[0])>1:
                    #print("1")
                    if len(center_points_list[0]) != len(PrevPaths[0]):
                        center_points_list[0] = resample_path_with_interpolation(center_points_list[0], len(PrevPaths[0]))
                    PrevPaths.append(center_points_list[0])
                    averaged_center_points = []
                    #print(str(len(center_points_list[0]))+" "+str(len(PrevPaths[0])))
                    averaged_center_points = calculate_weighted_sum(PrevPaths, center_points_list[0])
                    #for i in range(len(center_points_list[0])):
                        #print("2")
                        #sum = 0
                        #for j in range (len(PrevPaths)):
                            #print("3")
                            #print(PrevPaths[j])
                            #sum += PrevPaths[j][i][0]
                        #sum = int(float(sum)/float(len(PrevPaths)))
                        #averaged_center_points.append((sum, center_points_list[0][i][1]))
                        #print("4")
                        #print(sum)
                    #print("5")


                    center_points_list[0] = averaged_center_points
                # Draw lines connecting the center points on the image
                #print("5")
                for i, center_points in enumerate(center_points_list):
                    if i == 0:
                        color = (0, 255, 0)  # Green color for "Ego Path"
                    elif i == 1:
                        color = (255, 0, 0)  # Blue color for "Alternate Path"
                    else:
                        color = default_color  # Use the default color for other cases

                    for j in range(len(center_points) - 1):
                        start_row, start_col = center_points[j]
                        end_row, end_col = center_points[j + 1]
                        cv2.line(OrigImg, (start_col, start_row), (end_col, end_row), color, thickness=16)

            rectangle_color = (255, 0, 0)  # BGR color (red in this case)
            y = OrigImg.shape[1]
            startpt = (int(y - 4*y/5), xpos - dist)
            endpt = (int(y - y/5), xstart - dist)
            #startpt = (250,xpos-dist)
            #endpt=(850,xstart-dist)
            shapes = np.zeros_like(OrigImg, np.uint8)
            cv2.rectangle(
                shapes,
                startpt,
                endpt,
                rectangle_color, -1
                #rectangle_thickness,
            )
            out = OrigImg.copy()
            alpha = 0.2
            mask2 = shapes.astype(bool)
            out[mask2] = cv2.addWeighted(OrigImg, alpha, shapes, 1 - alpha, 0)[mask2]
            # Add transparency to the rectangle
            #cv2.addWeighted(image_with_transparency, transparency, image, 1 - transparency, 0, image)
            newmask = np.zeros_like(mask, np.uint8)
            out2 = out.copy()
            mask3 = newmask.astype(bool)
            #out2[mask3] = cv2.addWeighted(OrigImg, alpha, newmask, 1-alpha, 0)[mask3]
            return center_points_list[0], OrigImg

def find_center_path_smooth(binary_mask, image, theta=None):
    
    rows, cols, _ = binary_mask.shape
    trajectory = []

    for i in range(rows):
        row = binary_mask[i, :]
        #white_pixels = np.where(row == 255)[0]
        white_pixels = np.where(row[:, 0] == 255)[0]

        if len(white_pixels) > 0:
            center_x = int(np.mean(white_pixels))
            trajectory.append((center_x, i))

    trajectory = np.array(trajectory)

    if len(trajectory) < 1: return image, trajectory
    #print(trajectory)
    dist = 0 #add eq: x1 + v*tau
    xstart = 0
    xpos = 0

    #print("trajectory length: "+ str(len(trajectory)))
    if len(trajectory)>0:
        x_min = trajectory[-1][1]
        xstart = trajectory[np.argmax(trajectory,axis=0)][0][0]
        #xstart = trajectory[0][0]
        xpos = xstart - dist 


        #color_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
        #theta = joysticks[1].get_axis(0) / 9.8 * 3.1
        
        
        #print(trajectory.shape)
        new_center_points = []
        if theta != None:
                        theta = theta / 9.8 * 3.1
                        #print(theta)
                        
                        velocity = 1
                        xs, ys = zip(*trajectory)

                        #for (x, y) in trajectory[0]:
                        for i in range(len(xs)):
                            x = xs[i]
                            y = ys[i]
                            #y_prime = int(y + 30*(x - x_min) * velocity * np.sin(theta))
                            x_prime = int(x - 30*(y - x_min) * velocity * np.sin(theta))
                            #print("compare" + str(y) + " "+ str(y_prime))
                            #if y_prime >= 1280: y_prime = 1279
                            #if y_prime >= binary_mask.shape[1]: y_prime = binary_mask.shape[1] - 1
                            if x_prime >= binary_mask.shape[0]: x_prime = binary_mask.shape[0] - 1
                            #center_points[i] = (x, y_prime)
                            #y_prime = y

                            #if(binary_mask[x,y_prime] != (0,0,0)).all():
                            
                            tolerance = 0
                            if 0 <= y < binary_mask.shape[1] and 0 <= x_prime < binary_mask.shape[0]:
                               
                                #if (np.abs(binary_mask[x_prime, y] - (255, 0, 0)) > tolerance).any():
                                if np.array_equal(binary_mask[y, x_prime], [255, 0, 0]):

                                    new_center_points.append((x_prime, y))
                                    


                            else:

                                print("Indices out of bounds:", x_prime, y)
        
        
        #print(new_center_points[30])
        if len(new_center_points) >= 1:
            trajectory = np.array(new_center_points)
        #print(trajectory[30])
        #print(trajectory.shape)
        #print(trajectory[0][-1])
        trajectory = trajectory[:len(trajectory) - dist]
        x, y = trajectory[:, 0], trajectory[:, 1]

        tck, u = splprep([x, y], s=500) 
        unew = np.linspace(0, 1.0, 50 *len(trajectory))  

        out = splev(unew, tck)

        prev_point = None
        for i in range(len(out[0])):
         if len(out) == 2 and 0 <= i < min(len(out[0]), len(out[1])):
            point = (int(out[0][i]), int(out[1][i]))
            cv2.circle(image, point, 2, (0, 0, 0), -1)

            if prev_point is not None:
                cv2.line(image, prev_point, point, (0, 0, 0), 1)

            prev_point = point
        '''
        rectangle_color = (255, 0, 0)  # BGR color (red in this case)
        startpt = (250, xpos)
        endpt = (800, xstart)
        shapes = np.zeros_like(image, np.uint8)
        cv2.rectangle(
                shapes,
                startpt,
                endpt,
                rectangle_color, -1
                #rectangle_thickness,
        )
        out = image.copy()
        alpha = 0.2
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(image, alpha, shapes, 1 - alpha, 0)[mask]
        image = out
        '''
    return image, trajectory


def create_custom_colormap(num_classes):
    # Generate a set of evenly spaced colors for the number of classes
    hsv_values = [(i * 180 / num_classes, 255, 255) for i in range(num_classes)]
    colors = [list(map(lambda x: int(x), cv2.cvtColor(np.uint8([[hsv_values[i]]]), cv2.COLOR_HSV2BGR)[0][0])) for i in range(num_classes)]

    # Assign black color (0, 0, 0) to the background class (class 0)
    colors[0] = [0, 0, 0]

    # Create an empty colormap image
    colormap_image = np.zeros((256, 1, 3), dtype=np.uint8)

    # Assign colors to the colormap based on class indices
    for class_idx, color in enumerate(colors):
        colormap_image[class_idx] = color

    return colormap_image


def Draw_Det_Mask(outputs,dim):
    mask_frame = np.zeros(dim, dtype=np.uint8)

    for cat_idx, bbox in enumerate(outputs[0][0]):
 
        bbox_xyxy = bbox[:4].astype(int)
        class_idx = int(bbox[4])
        class_mask = np.zeros(dim, dtype=np.uint8)
        class_mask[bbox_xyxy[1]:bbox_xyxy[3], bbox_xyxy[0]:bbox_xyxy[2]] = 1

        #frame_mask[class_mask > 0] = class_idx
        # Create a binary mask of the bounding box with a unique class ID
        mask_bb = np.zeros_like(mask_frame, dtype=np.uint8)
        mask_bb[bbox_xyxy[1]:bbox_xyxy[3], bbox_xyxy[0]:bbox_xyxy[2]] = cat_idx + 1

        # Merge the bounding box mask into the frame mask
        mask_frame = cv2.bitwise_or(mask_frame, mask_bb)
    colormap = create_custom_colormap(10)
    colored_mask = cv2.applyColorMap(mask_frame, colormap)
        
    return mask_frame

def Draw_Seg_Mask(outputs):
    #PALETTE = [[219, 94, 86], [86, 211, 219], [0, 0, 0]]
    PALETTE = [[255,0,0], [0,255,0], [0, 0, 0]]
    # Create an empty mask image
    outputs = outputs[0]
    height, width = outputs.shape[:2]
    mask_image = np.zeros((height, width, 3), dtype=np.uint8)

    for pid in np.unique(outputs):
        mask = (outputs == pid).astype(np.uint8)
        # Set the corresponding pixel values in the mask image
        mask_image[mask != 0] = PALETTE[pid]

    return (mask_image)

def draw(outd, outs, height, width):
    mask_frame = np.zeros((height,width), dtype=np.uint8)
    rgb_to_grayscale_map = {
        (86, 211, 219): 30,
        (219, 94, 86): 50
    }
    for cat_idx, bbox in enumerate(outd[0][0]):
 
        bbox_xyxy = bbox[:4].astype(int)
        class_idx = int(bbox[4])
        class_mask = np.zeros((height,width), dtype=np.uint8)
        class_mask[bbox_xyxy[1]:bbox_xyxy[3], bbox_xyxy[0]:bbox_xyxy[2]] = 1

        #frame_mask[class_mask > 0] = class_idx
        # Create a binary mask of the bounding box with a unique class ID
        mask_bb = np.zeros_like(mask_frame, dtype=np.uint8)
        mask_bb[bbox_xyxy[1]:bbox_xyxy[3], bbox_xyxy[0]:bbox_xyxy[2]] = cat_idx + 1

        # Merge the bounding box mask into the frame mask
        mask_frame = cv2.bitwise_or(mask_frame, mask_bb)
    PALETTE = [[219, 94, 86], [86, 211, 219], [0, 0, 0]]
    mask_image = np.zeros((height,width, 3), dtype=np.uint8)
    outs = outs[0]
    for pid in np.unique(outs):
        mask = (outs == pid).astype(np.uint8)
        # Set the corresponding pixel values in the mask image
        mask_image[mask != 0] = PALETTE[pid]

    for rgb_color, grayscale_value in rgb_to_grayscale_map.items():
            mask = np.all(mask_image == np.array(rgb_color), axis=-1)
            mask_image[mask] = [grayscale_value]    
    mask = mask_frame != 0


    mask_image[mask, 0] = mask_frame[mask]  # Set R channel
    mask_image[mask, 1] = mask_frame[mask]  # Set G channel
    mask_image[mask, 2] = mask_frame[mask]  # Set B channel
    #for i in range(grayscale_image.shape[0]):
        #for j in range(grayscale_image.shape[1]):
            #if grayscale_image[i, j] != 0:
                #corresponding_rgb_image[i, j] = grayscale_image[i, j]

    # Create the "Mask image" containing only masks of pixel values corresponding to 30 and 50 in the "Fused Image"
    mask_image = np.zeros_like(mask_image, dtype=np.uint8)
    mask_image[mask_image == 30] = 1
    mask_image[mask_image == 50] = 2

    colored = cv2.applyColorMap(mask_image, create_custom_colormap(3))
    return (colored)


def init_segmentor(config, checkpoint=None, device='cuda:0'):
    """Initialize a segmentor from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str, optional) CPU/CUDA device option. Default 'cuda:0'.
            Use 'cpu' for loading model on CPU.
    Returns:
        nn.Module: The constructed segmentor.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    config.model.train_cfg = None
    model = build_segmentor(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        #model.CLASSES = checkpoint['meta']['CLASSES']
        #model.PALETTE = checkpoint['meta']['PALETTE']
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model

def predict_img(args, ds="kitti"):

    torch.cuda.set_device(0)
    """Detection Model"""
    #Loading model config
    cfg_det = mmcv.Config.fromfile(args.configdet)
    if cfg_det.load_from is None:
        cfg_name = os.path.split(args.config)[-1].replace(".py", ".pth")
        cfg_det.load_from = MODEL_SERVER + cfg_name
    if args.cfg_options is not None:
        cfg_det.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg_det.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    cfg_det.model.pretrained = None
    if cfg_det.model.get("neck"):
        if isinstance(cfg_det.model.neck, list):
            for neck_cfg in cfg_det.model.neck:
                if neck_cfg.get("rfp_backbone"):
                    if neck_cfg.rfp_backbone.get("pretrained"):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg_det.model.neck.get("rfp_backbone"):
            if cfg_det.model.neck.rfp_backbone.get("pretrained"):
                cfg_det.model.neck.rfp_backbone.pretrained = None
    



    #Build detector
    cfg_det.model.train_cfg = None
    modeldet = build_detector(cfg_det.model, test_cfg=cfg_det.get("test_cfg"))
    modeldet.to(0)
    fp16_cfg = cfg_det.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(modeldet)
    checkpoint = load_checkpoint(modeldet, cfg_det.load_from, map_location="cpu")
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    if "CLASSES" in checkpoint.get("meta", {}):
        modeldet.CLASSES = checkpoint["meta"]["CLASSES"]

    

    """Segmentation Model"""
    #Loading model config
    cfg_seg = mmcv.Config.fromfile(args.configseg)
    if cfg_seg.load_from is None:
        cfg_name = os.path.split(args.config)[-1].replace(".py", ".pth")
        cfg_seg.load_from = MODEL_SERVER + cfg_name
    if args.options is not None:
        cfg_seg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg_seg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        # hard code index
        cfg_seg.data.test.pipeline[1].img_ratios = [
            0.5,
            0.75,
            1.0,
            1.25,
            1.5,
            1.75,
        ]
        cfg_seg.data.test.pipeline[1].flip = True
    cfg_seg.model.pretrained = None
    cfg_seg.data.test.test_mode = True
    

    #Build segmentor
    cfg_seg.model.train_cfg = None
    modelseg = build_segmentor(cfg_seg.model, test_cfg=cfg_seg.get("test_cfg"))

    constructed_filename = str("C:/Users/fatim/Desktop/DTNet/bdd100k-models/drivable/"+cfg_seg.load_from.split()[-1].split("/")[-1])
    #print("Constructed Filename:", constructed_filename)

    modelseg = init_segmentor(cfg_seg, constructed_filename, device=0)
    modelseg.CLASSES = ("direct", "alternative", "background")
    modelseg.PALETTE = [[219, 94, 86], [86, 211, 219], [0, 0, 0]]
    #modelseg.PALETTE = [[0, 0, 255], [0, 255, 0], [0, 0, 0]]
    modelseg.to(0)
    #output_path_p = 'C:/Users/Rashid/Desktop/bdd100k/data_road/training/res_p'
    #output_path_g = 'C:/Users/Rashid/Desktop/bdd100k/data_road/training/res_g'
    #output_path_ = 'C:/Users/Rashid/Desktop/bdd100k/data_road/training/res_c'
    #output_m = 'C:/Users/Rashid/Desktop/bdd100k/data_road/training/kitti_predicted'
    #output_ogm = 'C:/Users/Rashid/Desktop/bdd100k/data_road/training/og_masks_bdd'
    #output_s = 'C:/Users/Rashid/Desktop/bdd100k/data_road/training/Steering'
    if ds == "kitti":
        images = os.listdir(os.path.join(args.imagefolder, "image_2"))
        labels = os.listdir(os.path.join(args.imagefolder, "gt_image_2"))
        #images = os.listdir('C:/Users/Rashid/Desktop/bdd100k/DataSearch/bdd100k/images/100k/val')
        #labels = os.listdir('C:/Users/Rashid/Desktop/bdd100k/DataSearch/bdd100k/labels/drivable/masks/val')

        columns = ["Image Name", "RMSE", "ED", "HD"]  # Add more metrics as needed
        df = pd.DataFrame(columns=columns)

        for i in range(289):
            print("image "+ str(i)+ " of "+str(len(images)))
            img = cv2.imread(os.path.join(args.imagefolder + "/image_2", images[i]))
            if images[i][:3] == "umm":
                label = cv2.imread(os.path.join(args.imagefolder + "/gt_image_2", "umm_road_"+images[i][4:]))
            elif images[i][:3] == "um_":
                label = cv2.imread(os.path.join(args.imagefolder + "/gt_image_2", "um_lane_"+images[i][3:]))
            elif images[i][:3] == "uu_":
                label = cv2.imread(os.path.join(args.imagefolder + "/gt_image_2", "uu_road_"+images[i][3:]))

            if images[i][:3] == "umm":
                labelname = images[i]
            elif images[i][:3] == "um_":
                labelname = images[i]
            elif images[i][:3] == "uu_":
                labelname = images[i]

            #img = cv2.imread(os.path.join('C:/Users/Rashid/Desktop/bdd100k/DataSearch/bdd100k/images/100k/val', images[i]))
            #label = cv2.imread(os.path.join('C:/Users/Rashid/Desktop/bdd100k/DataSearch/bdd100k/labels/drivable/masks/val', labels[i]))
            labell = label.copy()
            mask1 = (labell[:,:,2]==0)
            mask2 = (labell[:,:,2]==1)

            labell[mask1] = [255,0,0]
            labell[mask2] = [0,0,0]
            #cv2.imwrite(os.path.join(output_ogm, images[i]), labell)
            img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0  # Change format to channel-first
            img_tensor = img_tensor.unsqueeze(0).to(0)
            img_meta = [dict(
                    filename=images[i],  # Set the filename as needed
                    ori_shape=img.shape,
                    img_shape=img.shape,
                    pad_shape=img.shape,
                    scale_factor=1.0,
                    flip=False,
                    batch_input_shape=(1, 3, img.shape[0], img.shape[1]),  # Adjust as needed
            )]
            with torch.no_grad():
                outputsdet = modeldet.forward_test([img_tensor], [img_meta], rescale=True)
            with torch.no_grad():
                outputsseg = inference_segmentor(modelseg, img)
                
            det = Draw_Det_Mask(outputsdet, (img.shape[0], img.shape[1]))

            seg = Draw_Seg_Mask(outputsseg)


            # Create masks based on the element-wise comparisons
            #masky = (labell[:,:,1]!=0)  
            #maskp = (labell[:,:,0]!=0)
            #seg[maskp] = [0,255,0]
            #seg[masky] = [255,0,0]



            comb = CombineMasks(args, seg, det)

            mask = (comb[:,:,1]==255)
            if label[:3] =="umm": comb[mask] = [255, 0, 0]
            else: comb[mask] = [0, 0, 0]

            imgcopy = img.copy()
            imgcopy1 = img.copy()
            imgcopy2 = img.copy()
            #pathp, predicted= GeneratePath(comb, img, theta =-3.1)
            #cv2.imwrite(os.path.join(output_s, str(images[i][:-4])+"_left.jpg"), predicted)
            #pathp, predicted= GeneratePath(comb, imgcopy1)
            #cv2.imwrite(os.path.join(output_s, str(images[i][:-4])+"_center.jpg"), predicted)
            #pathp, predicted= GeneratePath(comb, imgcopy2, theta =3.1)
            #cv2.imwrite(os.path.join(output_s, str(images[i][:-4])+"_right.jpg"), predicted)
            #cv2.imwrite(os.path.join(output_m, labelname), seg)
            predicted, pathp = find_center_path_smooth(comb, imgcopy1)

            #cv2.imwrite(os.path.join(output_path_p, labels[i]), predicted)

            #comb[mask] = [255, 0, 255]
            #comb[mask==0] = [0, 0, 255]

            mask = (labell[:, :, 0] != 255)
            labell[mask] = [0, 0, 0]
            #label[mask == 0] = [255, 0, 0]
            #mask = (labell[:,:,0]!=0)
            #labell[mask] = [255, 0, 0]
            #labell[mask == 0] = [0, 0, 0]
            pathg, groundtruth = GeneratePath(labell, imgcopy)
            #cv2.imwrite(os.path.join(output_path_g, labels[i]), groundtruth)

            values = compare2(pathg, pathp, images[i])
            df = pd.concat([df, values], ignore_index=True)

            path_color = np.array([0, 255, 0])  

            # Create masks to isolate path pixels in each image
            mask1 = cv2.inRange(predicted, path_color, path_color)
            groundtruth[mask1==255] = [255, 0, 0]

            label_rect_position = (groundtruth.shape[1] - 320, 10)
            label_rect_size = (310, 80)
            cv2.rectangle(groundtruth, label_rect_position, (label_rect_position[0] + label_rect_size[0], label_rect_position[1] + label_rect_size[1]), (255, 255, 255), -1)

            #Add labels inside the rectangle
            label_text_position = (groundtruth.shape[1] - 310, 45)
            cv2.putText(groundtruth, "Ground Truth Path", label_text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            label_text_position = (groundtruth.shape[1] - 310, 80)
            cv2.putText(groundtruth, "Predicted Path", label_text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        


            #cv2.imwrite(os.path.join(output_path_, images[i]), groundtruth)

            #cv2.imshow("Result", groundtruth)
            #cv2.waitKey(0)  # Wait until a key is pressed
            #cv2.destroyAllWindows()  # Close all windows
        numeric_columns = df.select_dtypes(include=np.number)


        averages = numeric_columns.mean()
        print(len(numeric_columns))
        print(averages)
        df_with_averages = pd.concat([averages], ignore_index=True)
        df_with_averages.to_csv('path_values_kitti.csv', index=False)


    elif ds == "bdd":
        
        images = os.listdir('C:/Users/fatim/Desktop/DTNet/bdd100k-models/DataSearch/bdd100k/images/100k/val')
        labels = os.listdir('C:/Users/fatim/Desktop/DTNet/bdd100k-models/DataSearch/bdd100k/labels/drivable/masks/val')
        
        columns = ["Image Name", "RMSE", "ED", "HD"]  # Add more metrics as needed
        df = pd.DataFrame(columns=columns)
        for i in range(len(images)):
            print("image "+ str(i)+ " of "+str(len(images)))
            img = cv2.imread(os.path.join('C:/Users/fatim/Desktop/DTNet/bdd100k-models/DataSearch/bdd100k/images/100k/val', images[i]))
            label = cv2.imread(os.path.join('C:/Users/fatim/Desktop/DTNet/bdd100k-models/DataSearch/bdd100k/labels/drivable/masks/val', labels[i]))
            labell = label.copy()
            mask1 = (labell[:,:,2]==0)
            mask2 = (labell[:,:,2]==1)

            labell[mask1] = [255,0,0]
            labell[mask2] = [0,0,0]
            #cv2.imwrite(os.path.join(output_ogm, images[i]), labell)
            img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0  # Change format to channel-first
            img_tensor = img_tensor.unsqueeze(0).to(0)
            img_meta = [dict(
                    filename=images[i],  # Set the filename as needed
                    ori_shape=img.shape,
                    img_shape=img.shape,
                    pad_shape=img.shape,
                    scale_factor=1.0,
                    flip=False,
                    batch_input_shape=(1, 3, img.shape[0], img.shape[1]),  # Adjust as needed
            )]
            with torch.no_grad():
                outputsdet = modeldet.forward_test([img_tensor], [img_meta], rescale=True)
            with torch.no_grad():
                outputsseg = inference_segmentor(modelseg, img)
                
            det = Draw_Det_Mask(outputsdet, (img.shape[0], img.shape[1]))

            seg = Draw_Seg_Mask(outputsseg)


            #Create masks based on the element-wise comparisons
            masky = (labell[:,:,1]!=0)  
            maskp = (labell[:,:,0]!=0)
            seg[maskp] = [0,255,0]
            seg[masky] = [255,0,0]



            comb = CombineMasks(args, seg, det)

            mask = (comb[:,:,1]==255)

            imgcopy = img.copy()
            imgcopy1 = img.copy()
            imgcopy2 = img.copy()
            #pathp, predicted= GeneratePath(comb, img, theta =-3.1)
            #cv2.imwrite(os.path.join(output_s, str(images[i][:-4])+"_left.jpg"), predicted)
            #pathp, predicted= GeneratePath(comb, imgcopy1)
            #cv2.imwrite(os.path.join(output_s, str(images[i][:-4])+"_center.jpg"), predicted)
            #pathp, predicted= GeneratePath(comb, imgcopy2, theta =3.1)
            #cv2.imwrite(os.path.join(output_s, str(images[i][:-4])+"_right.jpg"), predicted)
            #cv2.imwrite(os.path.join(output_m, labelname), seg)
            predicted, pathp = find_center_path_smooth(comb, imgcopy1)

            #cv2.imwrite(os.path.join(output_path_p, labels[i]), predicted)

            comb[mask] = [255, 0, 255]
            comb[mask==0] = [0, 0, 255]

            mask = (labell[:, :, 0] != 255)
            labell[mask] = [0, 0, 0]
            #label[mask == 0] = [255, 0, 0]
            #mask = (labell[:,:,0]!=0)
            #labell[mask] = [255, 0, 0]
            #labell[mask == 0] = [0, 0, 0]
            pathg, groundtruth = GeneratePath(labell, imgcopy)
            #cv2.imwrite(os.path.join(output_path_g, labels[i]), groundtruth)

            values = compare2(pathg, pathp, images[i])
            df = pd.concat([df, values], ignore_index=True)

            path_color = np.array([0, 255, 0])  

            # Create masks to isolate path pixels in each image
            mask1 = cv2.inRange(predicted, path_color, path_color)
            groundtruth[mask1==255] = [255, 0, 0]

            label_rect_position = (groundtruth.shape[1] - 320, 10)
            label_rect_size = (310, 80)
            cv2.rectangle(groundtruth, label_rect_position, (label_rect_position[0] + label_rect_size[0], label_rect_position[1] + label_rect_size[1]), (255, 255, 255), -1)

            #Add labels inside the rectangle
            label_text_position = (groundtruth.shape[1] - 310, 45)
            cv2.putText(groundtruth, "Ground Truth Path", label_text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            label_text_position = (groundtruth.shape[1] - 310, 80)
            cv2.putText(groundtruth, "Predicted Path", label_text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        


            #cv2.imwrite(os.path.join(output_path_, images[i]), groundtruth)

            #cv2.imshow("Result", groundtruth)
            #cv2.waitKey(0)  # Wait until a key is pressed
            #cv2.destroyAllWindows()  # Close all windows
        numeric_columns = df.select_dtypes(include=np.number)


        averages = numeric_columns.mean()
        print(len(numeric_columns))
        print(averages)
        df_with_averages = pd.concat([averages], ignore_index=True)
        df_with_averages.to_csv('path_values_bdd.csv', index=False)


        
def compare2(gt, pred, name):
        new_x = []
        
        #ygt = prdpath[1]
        #ygt = [int(item[0]) if isinstance(item, np.ndarray) else item for item in ygt]

        #xgt = prdpath[0]
        #xgt = [int(item[0]) if isinstance(item, np.ndarray) else item for item in xgt]

        # Convert sequence2 to integers
        #ypr = y.astype(int).tolist()
        #xpr = x.astype(int).tolist()
        if gt:
            xgt, ygt = zip(*gt)
        else:
            xgt = [0 for x in range(len(pred))]
            ygt = []
        if pred.size > 0:
            xpr, ypr = zip(*pred)
            if not ygt: ygt = ypr
        else:
            xpr = [0 for x in range(len(gt))]
            ypr = ygt

        if (not ygt) and (not ypr):
            current_data = pd.DataFrame({
            "Image Name": [name],  # Note: these are lists now
            "RMSE": [0],
            #"MAD": [0],
            "ED": [0],
            "HD": [0],
            "FD": [0],
            "DTW": [0],
            })
            return current_data
        

        for y_value in ygt:
            if y_value in ypr:
                index = ypr.index(y_value)
                new_x.append(xpr[index])
            else:
                new_x.append(0)

        
        RMSEPath =rmse(np.array(new_x), np.array(xgt))
        # print(RMSEPath)

        MADPath=mad(xgt, ygt, new_x, ygt)
        MAD=MADPath[0]
        MED = mean_euclidean_distance(xgt, ygt, new_x, ygt)
        HD = hausdorff(xgt, ygt, new_x, ygt)
        FD = frechet(xgt, ygt, new_x, ygt)
        DTW = Dynamic_Time_Warping(xgt, ygt, new_x, ygt)
       
        # print("MAD:", MADPath[0])
        # print("ED:", MED)
        # print("HD:", HD)


        current_data = pd.DataFrame({
            "Image Name": [name],  # Note: these are lists now
            "RMSE": [RMSEPath],
            #"MAD": [MAD],
            "ED": [MED],
            "HD": [HD],
            "FD": [FD],
            "DTW": [DTW],

        })

        # Append the current data to the main DataFrame
        #df = pd.concat([df, current_data], ignore_index=True)

        return current_data

def mad(x1, y1, x2, y2):
    """Compute the Mean Absolute Difference (MAD) between two sets of points."""
    assert len(x1) == len(x2) == len(y1) == len(y2), "All arrays should have the same length."
    differences_x = np.abs(np.array(x1) - np.array(x2))
    differences_y = np.abs(np.array(y1) - np.array(y2))
    return np.mean(differences_x), np.mean(differences_y)

def mean_euclidean_distance(x1, y1, x2, y2):
    """Compute the Euclidean distance for all points and return the mean."""
    assert len(x1) == len(x2) == len(y1) == len(y2), "All arrays should have the same length."
    distances = [np.linalg.norm(np.array([x1[i], y1[i]]) - np.array([x2[i], y2[i]])) for i in range(len(x1))]
    return np.mean(distances)

def hausdorff(x1, y1, x2, y2):
    """Compute the Hausdorff distance between two sets of points."""
    from scipy.spatial import distance
    points1 = np.column_stack((x1, y1))
    points2 = np.column_stack((x2, y2))
    return distance.directed_hausdorff(points1, points2)[0]

def euclidean_distance(p1, p2):
    """
    Calculate the Euclidean distance between two points.
    """
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def frechet(x1, y1, x2, y2):
    """
    Calculate the Frechet distance between two curves defined by sequences of points.
    """
    curve1 = [(x1[i], y1[i]) for i in range(len(x1))]
    curve2 = [(x2[i], y2[i]) for i in range(len(x2))]

    n = len(curve1)
    m = len(curve2)

    # Initialize memoization table with -1
    memo = np.ones((n, m)) * -1

    # Initialize the first row and column of the memoization table
    memo[0, 0] = euclidean_distance(curve1[0], curve2[0])
    for i in range(1, n):
        memo[i, 0] = max(memo[i - 1, 0], euclidean_distance(curve1[i], curve2[0]))
    for j in range(1, m):
        memo[0, j] = max(memo[0, j - 1], euclidean_distance(curve1[0], curve2[j]))

    # Fill the memoization table iteratively
    for i in range(1, n):
        for j in range(1, m):
            memo[i, j] = max(
                min(
                    memo[i - 1, j],
                    memo[i, j - 1],
                    memo[i - 1, j - 1]
                ),
                euclidean_distance(curve1[i], curve2[j])
            )

    return memo[n - 1, m - 1]


def Dynamic_Time_Warping(x1, y1, x2, y2):
    """
    Calculate the Dynamic Time Warping (DTW) distance between two sequences of x and y coordinates.
    """
    # Compute the Euclidean distance matrix
    n = len(x1)
    m = len(x2)
    euclidean_matrix = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            euclidean_matrix[i, j] = euclidean_distance((x1[i], y1[i]), (x2[j], y2[j]))

    # Initialize the DTW matrix with zeros
    dtw_matrix = np.zeros((n, m))

    # Initialize the first row and column of the DTW matrix
    dtw_matrix[0, 0] = euclidean_matrix[0, 0]
    for i in range(1, n):
        dtw_matrix[i, 0] = dtw_matrix[i - 1, 0] + euclidean_matrix[i, 0]
    for j in range(1, m):
        dtw_matrix[0, j] = dtw_matrix[0, j - 1] + euclidean_matrix[0, j]

    # Fill in the rest of the DTW matrix
    for i in range(1, n):
        for j in range(1, m):
            cost = euclidean_matrix[i, j]
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1])

    # Return the DTW distance (bottom-right element of the matrix)
    return dtw_matrix[n - 1, m - 1]


def findcolors(img):
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Use color segmentation to identify unique colors in the image
    unique_colors = np.unique(hsv_image.reshape(-1, hsv_image.shape[2]), axis=0)

    # Print the unique colors in HSV format
    for color in unique_colors:
        print(f"HSV Color: {color}")

    # Convert the unique colors back to RGB for visualization
    unique_colors_rgb = cv2.cvtColor(np.uint8([[unique_colors]]), cv2.COLOR_HSV2BGR)

    # Display the unique colors as RGB
    for rgb_color in unique_colors_rgb[0]:
        print(f"RGB Color: {tuple(rgb_color)}")


def compare():
    predicted = 'C:/Users/Rashid/Desktop/bdd100k/data_road/training/res_p'
    groundtruth = 'C:/Users/Rashid/Desktop/bdd100k/data_road/training/res_g'
    pred_imgs = os.listdir(predicted)
    gt_images = os.listdir(groundtruth)
    mse_values = []
    euc_values = []
    empty = 0
    for i in range(len(pred_imgs)):
        print("image "+ str(i)+ " of "+str(len(pred_imgs)))
        image1 = cv2.imread(os.path.join(predicted, pred_imgs[i]))
        image2 = cv2.imread(os.path.join(groundtruth, gt_images[i]))
        #print(pred_imgs[i])
        #print(gt_images[i])

        path_color = np.array([0, 255, 0])  

        # Create masks to isolate path pixels in each image
        mask1 = cv2.inRange(image1, path_color, path_color)
        mask2 = cv2.inRange(image2, path_color, path_color)

        num_resampled_points = 100 # Adjust as needed


        if np.any(mask1 != 0) and np.any(mask2 != 0):
            resampled_path1 = resample_path_with_interpolation(mask1, num_resampled_points)
            resampled_path2 = resample_path_with_interpolation(mask2, num_resampled_points)
            height, width = image1.shape[:2]
            resampled_path1 = np.array(resampled_path1)
            resampled_path2 = np.array(resampled_path2)
            index_max_y_path1 = np.argmax(resampled_path1[:, 1])  # Assuming 'y' values are in the second column (index 1)
            index_max_y_path2 = np.argmax(resampled_path2[:, 1])  # Assuming 'y' values are in the second column (index 1)

            # Use the indices to access the elements
            ymax_path1 = resampled_path1[index_max_y_path1][1]  # This is the (x, y) pair with the maximum 'y' in path1
            ymax_path2 = resampled_path2[index_max_y_path2][1]
            #print(str(resampled_path1[np.argmax(resampled_path1[:, 1])])+" "+str(resampled_path2[np.argmax(resampled_path2[:, 1])])+" "+str(ymax))
            #print(str(ymax_path1)+" "+str(ymax_path2))

            index_min_y_path1 = np.argmin(resampled_path1[:, 1])  # Assuming 'y' values are in the second column (index 1)
            index_min_y_path2 = np.argmin(resampled_path2[:, 1])  # Assuming 'y' values are in the second column (index 1)

            # Use the indices to access the elements
            ymin_path1 = resampled_path1[index_min_y_path1][1]  # This is the (x, y) pair with the maximum 'y' in path1
            ymin_path2 = resampled_path2[index_min_y_path2][1]
            #print(str(ymin_path1)+" "+str(ymin_path2))

            ymax = np.min([ymax_path1, ymax_path2])
            ymin = np.max([ymin_path1, ymin_path2])
            #print(str(ymax)+" "+str(ymin))
            #mask_image = np.zeros((height, width, 3), dtype=np.uint8)
            #for j in range(len(resampled_path1) - 1):
                        #start_row, start_col = resampled_path1[j]
                        #end_row, end_col = resampled_path1[j + 1]
                        #cv2.line(mask_image, (start_col, start_row), (end_col, end_row), (255,255,255), thickness=8)
            #for j in range(len(resampled_path2) - 1):
                        #start_row, start_col = resampled_path2[j]
                        #end_row, end_col = resampled_path2[j + 1]
                        #cv2.line(mask_image, (start_col, start_row), (end_col, end_row), (255,0,255), thickness=8)
            #print(len(resampled_path1))
            #print(len(resampled_path2))
            # Calculate Root Mean Square Error (RMSE)
            #mse = mean_squared_error(resampled_path1, resampled_path2)
            sum = 0
            sume = 0
            yvalues = []
            for x1, y1 in resampled_path1:
                found = False
                for x2, y2 in resampled_path2:
                    if y1 == y2:
                        sum += np.square((x1 - x2))
                        sume += np.sqrt(np.square(x2-x1) + np.square(y2-y1))
                        found = True
                        yvalues.append(y2)
                if found == False:
                    if y1 in yvalues:
                        continue
                    elif y1 > ymax or y1 < ymin:
                        sum += np.square((x1 - 0))
                        sume += np.sqrt(np.square(0-x1) + np.square(0-y1))
                        yvalues.append(y2)
                    else:
                        #print("here" + str(np.sqrt(np.square(x2-x1) + np.square(y2-y1)))+" "+str(np.sqrt(np.square(0-x1) + np.square(0-y1))))
                        x2, y2 = find_nearest_pt(y1, resampled_path2)
                        sum += np.square((x1 - x2))
                        sume += np.sqrt(np.square(x2-x1) + np.square(y2-y1))
                        yvalues.append(y2)

            for x,y in resampled_path2:
                if y not in yvalues:
                    if y > ymax or y < ymin:
                        sum += np.square((x - 0))
                        sume += np.sqrt(np.square(0-x) + np.square(0-y))
                        yvalues.append(y2)
                        
                    else:
                        x2, y2 = find_nearest_pt(y, resampled_path1)
                        sum += np.square((x - x2))
                        sume += np.sqrt(np.square(x2-x) + np.square(y2-y))
                        yvalues.append(y2)


            sume = sume/len(resampled_path1)
            mse = np.sqrt(sum/len(resampled_path1))

            for x1, y1 in resampled_path1:
                for x2, y2 in resampled_path2:
                    pass
                    #sume += np.sqrt(np.square(x2-x1) + np.square(y2-y1))
            #sume = sume/len(resampled_path1)
        elif np.all(mask1 == 0) and np.all(mask2 == 0):
            mse = 0
            sume = 0
        else:
            mse = 100
            sume = 100
            empty += 1
        
        mse_values.append(mse)
        euc_values.append(sume)
        print("MSE:", mse)
        print("Euclidean Distance:", sume)
    
    average_mse = np.mean(mse_values)
    average_euc = np.mean(euc_values)

    print("Average MSE:", average_mse)
    print("Average Euclidean:", average_euc)
    #print(empty)

def find_nearest_pt(y1, path):

    minval = 1000
    point = (0,0)
    for x, y in path:
        temp = abs(y1 - y)
        if temp < minval:
            point = (x,y)
        minval = min(minval, temp)

    return point


    
def overlay():
    predicted = 'C:/Users/Rashid/Desktop/bdd100k/data_road/training/res_p'
    groundtruth = 'C:/Users/Rashid/Desktop/bdd100k/data_road/training/res_g'
    out = 'C:/Users/Rashid/Desktop/bdd100k/data_road/training/res_combined'
    pred_imgs = os.listdir(predicted)
    gt_images = os.listdir(groundtruth)
    image = cv2.imread(os.path.join(predicted, pred_imgs[0]))
    height = image.shape[0]
    width = image.shape[1]

    fps = 7
    outvid = cv2.VideoWriter("outkitti.mp4",
                          cv2.VideoWriter_fourcc(*"mp4v"),
                          fps,
                          (width, height))  # Specify frameSize as (width, height)

    for i in range(len(pred_imgs)):
        print("image "+ str(i)+ " of "+str(len(pred_imgs)))
        image1 = cv2.imread(os.path.join(predicted, pred_imgs[i]))
        image2 = cv2.imread(os.path.join(groundtruth, gt_images[i]))
        #print(pred_imgs[i])
        #print(gt_images[i])

        path_color = np.array([0, 255, 0])  

        # Create masks to isolate path pixels in each image
        mask1 = cv2.inRange(image1, path_color, path_color)
        image2[mask1!=0] = [255, 0, 0]
        

        label_rect_position = (image2.shape[1] - 320, 10)
        label_rect_size = (310, 80)
        cv2.rectangle(image2, label_rect_position, (label_rect_position[0] + label_rect_size[0], label_rect_position[1] + label_rect_size[1]), (255, 255, 255), -1)

        #Add labels inside the rectangle
        label_text_position = (image2.shape[1] - 310, 45)
        cv2.putText(image2, "Ground Truth Path", label_text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        label_text_position = (image2.shape[1] - 310, 80)
        cv2.putText(image2, "Predicted Path", label_text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imwrite(os.path.join(out, pred_imgs[i]), image2)
        outvid.write(image2)
    outvid.release


def resample_path_with_interpolation(path, num_points):
    #rows, cols = zip(path)
    rows, cols = np.where(path > 0)
    #cs = CubicSpline(rows, cols)
    #rows_interp = np.linspace(rows[0], rows[-1], num=num_points)  # You can adjust the number of points as needed
    #cols_interp = cs(rows_interp)
    # Check for and handle repeated x values
    unique_rows, unique_indices = np.unique(rows, return_index=True)
    unique_cols = cols[unique_indices]

    # Add a small amount of noise to repeated x values to make them unique
    noise = np.random.normal(0, 1e-6, len(unique_rows))
    unique_rows_with_noise = unique_rows + noise
    #print(len(unique_rows_with_noise))
    #print(len(unique_cols))
    # Perform Cubic Spline interpolation
    cs = CubicSpline(unique_rows_with_noise, unique_cols)

    resampled_rows = np.linspace(min(unique_rows_with_noise), max(unique_rows_with_noise), num_points)
    resampled_cols = cs(resampled_rows)
                    
    interp_center_points = [(int(row), int(col)) for row, col in zip(resampled_rows, resampled_cols)]

    return interp_center_points

def predict (args, video = None, output_name = "out.mp4"):

    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    """Detection Model"""
    #Loading model config
    cfg_det = mmcv.Config.fromfile(args.configdet)
    if cfg_det.load_from is None:
        cfg_name = os.path.split(args.config)[-1].replace(".py", ".pth")
        cfg_det.load_from = MODEL_SERVER + cfg_name
    if args.cfg_options is not None:
        cfg_det.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg_det.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    cfg_det.model.pretrained = None
    if cfg_det.model.get("neck"):
        if isinstance(cfg_det.model.neck, list):
            for neck_cfg in cfg_det.model.neck:
                if neck_cfg.get("rfp_backbone"):
                    if neck_cfg.rfp_backbone.get("pretrained"):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg_det.model.neck.get("rfp_backbone"):
            if cfg_det.model.neck.rfp_backbone.get("pretrained"):
                cfg_det.model.neck.rfp_backbone.pretrained = None
    



    #Build detector
    cfg_det.model.train_cfg = None
    modeldet = build_detector(cfg_det.model, test_cfg=cfg_det.get("test_cfg"))
    modeldet.to(0)
    fp16_cfg = cfg_det.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(modeldet)
    checkpoint = load_checkpoint(modeldet, cfg_det.load_from, map_location="cpu")
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    if "CLASSES" in checkpoint.get("meta", {}):
        modeldet.CLASSES = checkpoint["meta"]["CLASSES"]

    

    """Segmentation Model"""
    #Loading model config
    cfg_seg = mmcv.Config.fromfile(args.configseg)
    if cfg_seg.load_from is None:
        cfg_name = os.path.split(args.config)[-1].replace(".py", ".pth")
        cfg_seg.load_from = MODEL_SERVER + cfg_name
    if args.options is not None:
        cfg_seg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg_seg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        # hard code index
        cfg_seg.data.test.pipeline[1].img_ratios = [
            0.5,
            0.75,
            1.0,
            1.25,
            1.5,
            1.75,
        ]
        cfg_seg.data.test.pipeline[1].flip = True
    cfg_seg.model.pretrained = None
    cfg_seg.data.test.test_mode = True
    

    #Build segmentor
    cfg_seg.model.train_cfg = None
    modelseg = build_segmentor(cfg_seg.model, test_cfg=cfg_seg.get("test_cfg"))

    constructed_filename = str("C:/Users/Rashid/Desktop/bdd100k/Combined/"+cfg_seg.load_from.split()[-1].split("/")[-1])
    
    #print("Constructed Filename:", constructed_filename)

    modelseg = init_segmentor(cfg_seg, constructed_filename, device=0)
    modelseg.CLASSES = ("direct", "alternative", "background")
    modelseg.PALETTE = [[219, 94, 86], [86, 211, 219], [0, 0, 0]]
    #modelseg.PALETTE = [[0, 0, 255], [0, 255, 0], [0, 0, 0]]
    modelseg.to(0)

    #Activate joystick

    #pygame.init()
    #joysticks = []
    #velocity = 0.1
    #for j in range(0, pygame.joystick.get_count()):
                    
        #joysticks.append(pygame.joystick.Joystick(j))
        #joysticks[-1].init()
    #print(joysticks[1].get_axis(0))

    #Split video
    #vid = video if video else args.sourceimages
    vid = str("C:/Users/Rashid/Desktop/bdd100k/videos/day.mp4")
    vid = str("C:/Users/Rashid/Desktop/bdd100k/videos/original/0000f77c-62c2a288.mov")
    cap = cv2.VideoCapture(vid)
    
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_name,
                          cv2.VideoWriter_fourcc(*"mp4v"),
                          fps,
                          (width, height))  # Specify frameSize as (width, height)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    outseg = cv2.VideoWriter(output_name[:-4]+"_seg.mp4",
                          cv2.VideoWriter_fourcc(*"mp4v"),
                          fps,
                          (width, height))  # Specify frameSize as (width, height)
    outcomb = cv2.VideoWriter(output_name[:-4]+"_overlay.mp4",
                          cv2.VideoWriter_fourcc(*"mp4v"),
                          fps,
                          (width, height))  # Specify frameSize as (width, height)
    #with profile(activities=[
        #ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        #with record_function("model_inference"):
    
    paths = []
    timer = 0
    for frame in range(n_frames):
              #if frame <=100:
                if frame == 1: timer = time.perf_counter()
                print("Frame" + str(frame) + "/" + str(n_frames))
                ret, img = cap.read()  # Read the frame from the video capture
                if not ret:
                    break

                img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0  # Change format to channel-first
                img_tensor = img_tensor.unsqueeze(0).to(0)
                img_meta = [dict(
                    filename=vid,  # Set the filename as needed
                    ori_shape=img.shape,
                    img_shape=img.shape,
                    pad_shape=img.shape,
                    scale_factor=1.0,
                    flip=False,
                    batch_input_shape=(1, 3, img.shape[0], img.shape[1]),  # Adjust as needed
                )]

                with torch.no_grad():

                    outputsdet = modeldet.forward_test([img_tensor], [img_meta], rescale=True)
                    det = Draw_Det_Mask(outputsdet, (img.shape[0], img.shape[1]))
                    del outputsdet
                x = modelseg.backbone(img_tensor)
                if modelseg.with_neck:
                    x = modelseg.neck(x)
                with torch.no_grad():

                    #outputsseg = inference_segmentor(modelseg, img)
                    
                    outputsseg = modelseg.decode_head.forward_test(img_tensor, img_meta, cfg_seg)#, rescale=True)


                print(outputsseg)
                seg = Draw_Seg_Mask(outputsseg)

                outseg.write(seg)
                segtemp = seg.copy()
                path_color = np.array([255, 0, 0])
                mask1 = cv2.inRange(seg, path_color, path_color)
                seg[mask1==0] = [0, 0, 0]

                
                comb = CombineMasks(args, seg, det)

                path, img= GeneratePath(comb, img, paths)
                result = cv2.addWeighted(img, 1, segtemp, 0.5, 0)
                if path: paths.append(path)
                if len(paths)>100: paths.pop(0)

                outcomb.write(result)


                out.write(img)

    
                
            
    #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    out.release()
    outseg.release()
    outcomb.release()
    cap.release()
    timer -= -time.perf_counter()
    print(str(timer)+" seconds")
    print(str(n_frames/timer)+" fps")
        
def predict_videos(args):
    path = "C:/Users/Rashid/Desktop/bdd100k/videos"
    in_path = os.path.join(path, "original")
    out_path = os.path.join(path, "predicted")
    vids = os.listdir(os.path.join(path, "original"))
    i=0
    for video in vids:
        i+=1
        if i > 17:
            print("video " + str(i) + " of "+str(len(vids)))
            predict(args, os.path.join(in_path, video), os.path.join(out_path, video))

def predict_feed(args):
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    
    obs_virtual_camera = cv2.VideoCapture(0)
    torch.cuda.set_device(0)
    """Detection Model"""
    #Loading model config
    cfg_det = mmcv.Config.fromfile(args.configdet)
    if cfg_det.load_from is None:
        cfg_name = os.path.split(args.config)[-1].replace(".py", ".pth")
        cfg_det.load_from = MODEL_SERVER + cfg_name
    if args.cfg_options is not None:
        cfg_det.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg_det.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    cfg_det.model.pretrained = None
    if cfg_det.model.get("neck"):
        if isinstance(cfg_det.model.neck, list):
            for neck_cfg in cfg_det.model.neck:
                if neck_cfg.get("rfp_backbone"):
                    if neck_cfg.rfp_backbone.get("pretrained"):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg_det.model.neck.get("rfp_backbone"):
            if cfg_det.model.neck.rfp_backbone.get("pretrained"):
                cfg_det.model.neck.rfp_backbone.pretrained = None
    



    #Build detector
    cfg_det.model.train_cfg = None
    modeldet = build_detector(cfg_det.model, test_cfg=cfg_det.get("test_cfg"))
    modeldet.to(0)
    fp16_cfg = cfg_det.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(modeldet)
    checkpoint = load_checkpoint(modeldet, cfg_det.load_from, map_location="cpu")
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    if "CLASSES" in checkpoint.get("meta", {}):
        modeldet.CLASSES = checkpoint["meta"]["CLASSES"]

    

    """Segmentation Model"""
    #Loading model config
    cfg_seg = mmcv.Config.fromfile(args.configseg)
    if cfg_seg.load_from is None:
        cfg_name = os.path.split(args.config)[-1].replace(".py", ".pth")
        cfg_seg.load_from = MODEL_SERVER + cfg_name
    if args.options is not None:
        cfg_seg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg_seg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        # hard code index
        cfg_seg.data.test.pipeline[1].img_ratios = [
            0.5,
            0.75,
            1.0,
            1.25,
            1.5,
            1.75,
        ]
        cfg_seg.data.test.pipeline[1].flip = True
    cfg_seg.model.pretrained = None
    cfg_seg.data.test.test_mode = True
    

    #Build segmentor
    cfg_seg.model.train_cfg = None
    modelseg = build_segmentor(cfg_seg.model, test_cfg=cfg_seg.get("test_cfg"))

    constructed_filename = str("C:/Users/fatim/Desktop/DTNet/bdd100k-models/drivable/"+cfg_seg.load_from.split()[-1].split("/")[-1])
    #print("Constructed Filename:", constructed_filename)

    modelseg = init_segmentor(cfg_seg, constructed_filename, device=0)
    modelseg.CLASSES = ("direct", "alternative", "background")
    modelseg.PALETTE = [[219, 94, 86], [86, 211, 219], [0, 0, 0]]
    #modelseg.PALETTE = [[0, 0, 255], [0, 255, 0], [0, 0, 0]]
    modelseg.to(0)
    #paths = []
    i = 0
    pygame.init()
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    velocity = 0.1
    # Prints the joystick's name
    JoyName = pygame.joystick.Joystick(0).get_name()
    #print ("Name of the joystick:")
    #print (JoyName)
    # Gets the number of axes
    JoyAx = pygame.joystick.Joystick(0).get_numaxes()
    #print ("Number of axis:")
    #print (JoyAx)

    prev = []
    while True:
     if i%12 == 0:
        ret, img = obs_virtual_camera.read()
    
        if not ret:
            break
    
        # Process the frame here (e.g., apply image processing)
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0  # Change format to channel-first
        img_tensor = img_tensor.unsqueeze(0).to(0)
        img_meta = [dict(
                    filename="file",  # Set the filename as needed
                    ori_shape=img.shape,
                    img_shape=img.shape,
                    pad_shape=img.shape,
                    scale_factor=1.0,
                    flip=False,
                    batch_input_shape=(1, 3, img.shape[0], img.shape[1]),  # Adjust as needed
        )]

        with torch.no_grad():

                    outputsdet = modeldet.forward_test([img_tensor], [img_meta], rescale=True)
                    #outputsdet = [output.cpu() for output in outputsdet]
        det = Draw_Det_Mask(outputsdet, (img.shape[0], img.shape[1]))
        del outputsdet
        #print(torch.cuda.memory_summary(device=None, abbreviated=False))
        with torch.no_grad():

                    #outputsseg = inference_segmentor(modelseg, img)#.cpu()
                    outputsseg = modelseg.forward_test([img_tensor], [img_meta], rescale=True)
                    #outputsdet = [output.cpu() for output in outputsseg]

        seg = Draw_Seg_Mask(outputsseg)
        del outputsseg
        #det = Draw_Det_Mask(outputsdet, (img.shape[0], img.shape[1]))
        comb = CombineMasks(args, seg, det)
        #path, img= GeneratePath(comb, img, paths)

        pygame.event.pump()
        #print (pygame.joystick.Joystick(0).get_axis(0))
        #img, path= find_center_path_smooth(comb, img, joystick.get_axis(0))
        img, _, _, _, _, path = find_center_path_smooth_new(comb, img, prev, 1, joystick.get_axis(0))
        #Path Averaging if needed
        #if path: paths.append(path)
        #if len(paths)>100: paths.pop(0)
        cv2.namedWindow("Processed Frame", cv2.WINDOW_NORMAL)

        cv2.resizeWindow("Processed Frame", int(1280*1.25), int(720*1.25))  

        cv2.imshow("Processed Frame", img)
        del img
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cv2.namedWindow("Segmentation Mask", cv2.WINDOW_NORMAL)

        cv2.resizeWindow("Segmentation Mask", int(1280*1.25), int(720*1.25)) 

        cv2.imshow("Segmentation Mask", comb)
        del comb
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
     i += 1

    obs_virtual_camera.release()
    cv2.destroyAllWindows()
def check_color():
        labels = os.listdir('C:/Users/Rashid/Desktop/bdd100k/DataSearch/bdd100k/labels/drivable/masks/val')
        label = cv2.imread(os.path.join('C:/Users/Rashid/Desktop/bdd100k/DataSearch/bdd100k/labels/drivable/masks/val',labels[3]))

        mask = (label[:,:,2]==0)
        mask2 = (label[:,:,2]==1)
        label[mask] = [255,0,0]
        label[mask2] = [0,255,0]

        cv2.imshow("Result", label)
        cv2.waitKey(0)  # Wait until a key is pressed
        cv2.destroyAllWindows()  # Close all windows

def test(args):
    img = cv2.imread('C:/Users/Rashid/Desktop/bdd100k/videos/test.png')
    img_height, img_width, _ = img.shape  # Get the height and width of the image
    img_ = img.copy()
    # Create a blank mask with the same dimensions as the image
    blank_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    img = CombineMasks(args, img, blank_mask)
    _, img = GeneratePath(img, img_)
    cv2.imshow("Result", img)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()  # Close all windows








def main() -> None:
    """Main function for model inference."""
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    args = parse_args() 
    #check_color()
    #predict_videos(args)
    #predict_img(args, ds="bdd")
    #predict(args)
    #compare()
    #overlay()
    predict_feed(args)
    #test(args)

    


if __name__ == "__main__":
    main()