#
# This script creates a number of traffic sign prototype images by perspectively transforming a number of icons and using a pre-trained CycleGAN to 
# transfer them into life-like traffic sign images.
#
# If you want to run this script, you need to adapt the paths at the beginning accordingly.
#
# Sebastian Houben, sebastian.houben@ini.rub.de
# Real-time Computer Vision
# Institute for Neural Computation
# Ruhr-University of Bochum
#

import sys
import os
from tools.add_background import add_backgrounds
from tools.diversify_images import diversify_images
from tools.perspectify_images import perspectify_images
from tools.clear_directory import clear_directory
from cyclegan.apps.run import main as cyclegan_main
from generate_labels import main as label_main

# CHANGE THESE LINES TO YOUR NEEDS 

# configuration file that describes the used CycleGAN and how it was trained (if you don't know what you are doing yet, leave it as it is)
config_file = os.path.split(os.path.realpath(__file__))[0] + r"\..\..\configs\gtsb2icon128_get_started_config.json" 
# you need to change "checkpoint_path" entry in the config file to the path where you stored the pre-trained model (you can obtain it at http://benchmark.ini.rub.de/extern_downloads/cyclegan_traffic_sign_generation.zip)

# path with traffic sign icons (available at: http://benchmark.ini.rub.de/extern_downloads/traffic_sign_icons_original.zip, you can also compile your own set of traffic sign images, e.g., from https://commons.wikimedia.org/wiki/Road_signs_of_Germany)
original_icon_dataset_path = r"ENTER THE PATH WHERE YOU EXTRACTED THE PACKAGE FROM http://benchmark.ini.rub.de/extern_downloads/traffic_sign_icons_original.zip"
# folder where prototype images (i.e. images generated from icons that undergo a slight change in perspective and are enhanced with background colors that the CycleGAN will later turn into life-like background )
dest_icon_path = r""
# folder where the life-like generated images are stored
dest_generated_path = r""
# number of life-like images to generate (approximately)
no_generated_imgs = 100

# CHANGE THESE LINES TO YOUR NEEDS --- END

if not os.path.exists(original_icon_dataset_path):
    print( "The following path does not exist: %s" % original_icon_dataset_path)
    return

if not os.path.exists(config_file):
    print( "The following file does not exist: %s" % config_file)
    return

# number of signs per icon image
diversify_count = round(int(no_generated_imgs) / len([file_name for file_name in os.listdir(original_icon_dataset_path) if os.path.isfile(os.path.join(original_icon_dataset_path, file_name)) and file_name.endswith(".png")]))

# path for temporary results that is deleted afterwards
tmp_icon_path = dest_icon_path + '_trans'

if not os.path.exists(tmp_icon_path):
    os.mkdir(tmp_icon_path)

if os.path.exists(dest_icon_path):
    clear_directory(dest_icon_path)
if os.path.exists(tmp_icon_path):
    clear_directory(tmp_icon_path)

# transform icons to generated prototype image set
target_size = (128, 128)
perspectify_images(original_icon_dataset_path, tmp_icon_path, target_size, (0.0, 0.24), (0.4, 0.6), (-0.1, 0.1), diversify_count, extension = ".png") # perspective distortion 
add_backgrounds(tmp_icon_path, dest_icon_path, "homogeneous", extension = ".png") # change background color
label_main([None, dest_icon_path]) # add a label file, in case you want to train on these

clear_directory(tmp_icon_path)
os.rmdir(tmp_icon_path)

# these are commandline options that are used to call the run script from the trained CycleGAN
cyclegan_options = list()
cyclegan_options.append("run.py")
cyclegan_options.append( config_file ) 
cyclegan_options.append("yx") # direction (icon -> real == "xy", real -> icon == "yx")
cyclegan_options.append(dest_icon_path)
cyclegan_options.append(dest_generated_path)
cyclegan_options.append("-e")
cyclegan_options.append(".png") # png image file extension
cyclegan_options.append("-i")
cyclegan_options.append(dest_icon_path + "/labels.json") # use label file to generate labels for randomly generated image files
cyclegan_options.append("-o")
cyclegan_options.append(dest_generated_path + "/labels.json") # put label file in destination folder alongside the generated image files

cyclegan_main(cyclegan_options) # load CycleGAN and execute it on every prototype image


