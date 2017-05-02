The code (attached in the zip file) is written for Tensorflow version 0.12. You will need to install Tensorflow on your lab server account before running the code. You may follow the steps mentioned in the "Installing Tensorflow v0.12".txt file.

The code assumes that there is a "data" folder in the directory where you run it, with 3 subfolders... "train", "valid" and "test"... put your images and their corresponding transcriptions in the right folders...

Run,
$ python normalize_padd.py

Just to tell you what's happening in this code. It will rescale all your images to a height of 48. It also white-padds the smaller images horizontally so that all the images are of the same dimension, 
i.e. 48 x (max width of all your images)...
It creates a pickle file, "train_widths.pickle", which contains a dictionary with filenames as keys and their rescaled_widths as values (the widths mentioned here are after rescalling and before white-padding).

There's a file named "label_generate.py" that reads all your labels and stores them into a string named DIGITS. If you want to do it some other way, just make sure to override the string variable named DIGITS in the same file.


Before running the training, you may need to change a couple of parameters in the common.py file according to your needs...
Whatever values of batch sizes you use there, just make sure that 
BATCH_SIZE * BATCHES = no. of training images
and, 
TEST_BATCH_SIZE * TEST_BATCHES = no. of test images

In order to run the training, type:
$ python trainer2.py
