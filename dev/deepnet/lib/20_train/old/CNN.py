# Trains CNN model with arbirtrary number and type of branches
# INPUT:  encoded branch input file(s)
# INPUT:  hyperparameters
# OUTPUT: trained model

import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, Activation, LSTM, Bidirectional, Concatenate, Input, BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
from keras import metrics
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU
from keras import regularizers
from keras import initializers
from optparse import OptionParser, OptionGroup

usage = "usage: %prog [options]"
opt = OptionParser(usage=usage)

opt.add_option(
    "-i", "--ifile", 
    action="append",
    dest="ifiles",
    help="Input branch files, use multiple times for more branches"
)
opt.add_option(
    "-o", "--ofile", 
    action="store",
    dest="ofile",
    help="Output file location"
)
# Group: Optional Parameters
group = OptionGroup(
    opt, "Optional parameters"
)
group.add_option(
    "--logfile", 
    action="store",
    dest="logfile",
    help="Output logfile location"
)
group.add_option(
    "-v", "--verbose",
    action="store_true", 
    dest="verbose", 
    default=True,
    help="print progress [default]"
)
group.add_option(
    "-q", "--quiet",
    action="store_false", dest="verbose",
    help="run quietly"
)
group.add_option(
    "--rseed", 
    action="store",
    dest="rseed",
    help="Randomization seed",
    default=8
)
opt.add_option_group(group)
# Group: Hyperparameters
hypergroup = OptionGroup(
    opt, "Training Hyperparameters"
)
hypergroup.add_option(
    "-h", "--hyperfile", 
    action="store",
    dest="hyperfile",
    help="Hyperparameters file. Values in file override manual entries" #TODO check if this needs to be done the other way instead
)
hypergroup.add_option(
    "--batch_size",
    action="store",
    default=256,
    help="Batch Size. Default=256"
)
hypergroup.add_option(
    "--dropout",
    action="store",
    default=0.2,
    help="Dropout. Default=0.2"
)
hypergroup.add_option(
    "--lr",
    action="store",
    default=0.0001,
    help="Learning Rate. Default=0.0001"
)
hypergroup.add_option(
    "--filter_num",
    action="store",
    default=40,
    help="Filter Number. Default=40"
)
opt.add_option_group(hypergroup)
(options, args) = opt.parse_args()

# TODO: discuss : should we add info about network architecture (e.g. number of convolutions, neurons etc) as input?

### Setting Random Seed
tf.set_random_seed(options.rseed)
# TODO print to logfile

### TODO
# - open input files and store as "branch" objects
# - make array of branches
# - get branch dimensions
branches = ["seq", "cons"]
#TODO print array info to logfile

### Storing Hyperparameters
batch_size = options.batch_size
dropout = options.dropout
lr = options.lr
filter_num = options.filter_num
# TODO : open hyperparameter file and update values as needed
# TODO: print hyperparameters to logfile


sess = tf.Session(config=tf.ConfigProto(log_device_placement=True)) #check configuration TODO

number_of_convolutions = 3 # possibly from parameters input?

train_x ###TODO is this where the input data gets imported?
train_y

output = []
inputData = []

for branch in branches: ### maybe I'm missing it but 'branch' is only ever called to get its branch.shape. Where does the data get imported?
    x = Input(shape = (branch.shape))
    inputData.append(x)
    
    for convolution in range(0,number_of_convolutions-1):
        
        x = Conv1D(filters = filter_num * 2, kernel_size = kernelsize[convolution], strides = 1, padding = "same")(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size = 2, padding = "same")(x)
        x = Dropout(rate = dropout, noise_shape = None, seed = None)(x)

	#x = Bidirectional(LSTM(16, return_sequences=True, activation='tanh', recurrent_activation='hard_sigmoid', dropout=dropout, recurrent_dropout=dropout))(x)
	#x = Bidirectional(LSTM(8, return_sequences=True, activation='tanh', recurrent_activation='hard_sigmoid', dropout=dropout, recurrent_dropout=dropout))(x)

	output.append(Flatten()(x))

x = []
if len(branches) == 1:
    x = outputs[0]
else:
    x = keras.layers.concatenate(outputs)    
    
for dense in range(0,number_of_dense-1):
    x = Dense(units = number_of_units[dense])(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(rate = dropout, noise_shape = None, seed = None)(x)
x = Dense(units = number_of_branches, activation = "softmax")(x)


classification_model = Model(inputData, x)
if len(modes) == 1:
		classification_model = Model(inputData[0], x)
    
print "\t\t\tCompiling network components."
sgd = SGD(
        lr = tmp_lr,
        decay = 1e-6,
        momentum = 0.9,
        nesterov = True)
classification_model.compile(
        optimizer = sgd,
        loss = "categorical_crossentropy",
        metrics = ["accuracy"])
mcp = ModelCheckpoint(filepath = tmp_output_directory + "/CNNonRaw.hdf5",
				verbose = 0,
				save_best_only = True)
earlystopper = EarlyStopping(monitor = 'val_loss', 
                patience = 40,
                min_delta = 0,
                verbose = 1,
                mode = 'auto')
csv_logger = CSVLogger(tmp_output_directory + "/CNNonRaw.log.csv", 
            append=True, 
            separator='\t')

print "\t\t\tTraining network."
history = classification_model.fit(
    train_x[0], 
    train_y,
    batch_size = tmp_batch_size,
    epochs = 600,
    verbose = 1,
    validation_data = (valid_x[0], valid_y),
    callbacks = [mcp, earlystopper, csv_logger])

print "\t\t\tTesting network."
tresults = classification_model.evaluate(
    test_x[0], 
    test_y,
    batch_size = tmp_batch_size,
    verbose = 1,
    sample_weight = None)
print "\t\t\t[loss, acc]"
print tresults

# summarize history for accuracy  
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.ylim(0.0, 1.0)
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.savefig(tmp_output_directory + "/CNNonRaw.acc.png", dpi=300)
plt.clf()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylim(0.0, max(max(history.history['loss']), max(history.history['val_loss'])))
plt.title('Model Loss')
plt.ylabel('Categorical Crossentropy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig(tmp_output_directory + "/CNNonRaw.loss.png", dpi=300)
plt.clf()
