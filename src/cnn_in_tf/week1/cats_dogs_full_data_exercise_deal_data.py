# In this exercise you will train a CNN on the FULL Cats-v-dogs dataset
# This will require you doing a lot of data preprocessing because
# the dataset isn't split into training and validation for you
# This code block has all the required inputs
import os
import zipfile
import random
import tensorflow as tf
# from keras.optimizers import RMSprop
# from keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile

# This code block downloads the full Cats-v-Dogs dataset and stores it as
# cats-and-dogs.zip. It then unzips it to /tmp
# which will create a tmp/PetImages directory containing subdirectories
# called 'Cat' and 'Dog' (that's how the original researchers structured it)
# If the URL doesn't work,
# .   visit https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765
# And right click on the 'Download Manually' link to get a new URL

# !wget --no-check-certificate \
#     "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip" \
#     -O "/tmp/cats-and-dogs.zip"


# local_zip = '/tmp/cats-and-dogs.zip'
# local_zip = 'E:/dataset/cat_dogs/kagglecatsanddogs_3367a.zip'
# zip_ref = zipfile.ZipFile(local_zip, 'r')
# zip_ref.extractall('E:/dataset/cat_dogs')
# zip_ref.close()

print(len(os.listdir('E:/dataset/cat_dogs/PetImages/Cat/')))
print(len(os.listdir('E:/dataset/cat_dogs/PetImages/Dog/')))


# Use os.mkdir to create your directories
# You will need a directory for cats-v-dogs, and subdirectories for training
# and testing. These in turn will need subdirectories for 'cats' and 'dogs'
dir_base = 'E:/dataset/cat_dogs'

try:
    #YOUR CODE GOES HERE
    os.mkdir(dir_base+ '/cats-v-dogs')
    os.mkdir(dir_base+'/cats-v-dogs/training')
    os.mkdir(dir_base+'/cats-v-dogs/testing')
    os.mkdir(dir_base+'/cats-v-dogs/training/cats')
    os.mkdir(dir_base+'/cats-v-dogs/training/dogs')
    os.mkdir(dir_base+'/cats-v-dogs/testing/cats')
    os.mkdir(dir_base+'/cats-v-dogs/testing/dogs')
except OSError:
    pass



# Write a python function called split_data which takes
# a SOURCE directory containing the files
# a TRAINING directory that a portion of the files will be copied to
# a TESTING directory that a portion of the files will be copie to
# a SPLIT SIZE to determine the portion
# The files should also be randomized, so that the training set is a random
# X% of the files, and the test set is the remaining files
# SO, for example, if SOURCE is PetImages/Cat, and SPLIT SIZE is .9
# Then 90% of the images in PetImages/Cat will be copied to the TRAINING dir
# and 10% of the images will be copied to the TESTING dir
# Also -- All images should be checked, and if they have a zero file length,
# they will not be copied over
#
# os.listdir(DIRECTORY) gives you a listing of the contents of that directory
# os.path.getsize(PATH) gives you the size of the file
# copyfile(source, destination) copies a file from source to destination
# random.sample(list, len(list)) shuffles a list


def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    # YOUR CODE STARTS HERE
    files = []
    for filename in os.listdir(SOURCE):
        file = SOURCE + filename
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")

    training_length = int(len(files) * SPLIT_SIZE)
    testing_length = int(len(files) - training_length)
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[0:training_length]
    testing_set = shuffled_set[-testing_length:]

    for filename in training_set:
        this_file = SOURCE + filename
        destination = TRAINING + filename
        copyfile(this_file, destination)

    for filename in testing_set:
        this_file = SOURCE + filename
        destination = TESTING + filename
        copyfile(this_file, destination)
    # YOUR CODE ENDS HERE


CAT_SOURCE_DIR = dir_base+ "/PetImages/Cat/"
TRAINING_CATS_DIR = dir_base+ "/cats-v-dogs/training/cats/"
TESTING_CATS_DIR = dir_base+ "/cats-v-dogs/testing/cats/"
DOG_SOURCE_DIR = dir_base+ "/PetImages/Dog/"
TRAINING_DOGS_DIR = dir_base+ "/cats-v-dogs/training/dogs/"
TESTING_DOGS_DIR = dir_base+ "/cats-v-dogs/testing/dogs/"

split_size = .9
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)

# Expected output
# 666.jpg is zero length, so ignoring
# 11702.jpg is zero length, so ignoring
print(len(os.listdir('/tmp/cats-v-dogs/training/cats/')))
print(len(os.listdir('/tmp/cats-v-dogs/training/dogs/')))
print(len(os.listdir('/tmp/cats-v-dogs/testing/cats/')))
print(len(os.listdir('/tmp/cats-v-dogs/testing/dogs/')))
# Expected output:
# 11250
# 11250
# 1250
# 1250

