import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import random as rn

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = '0'

# Necessary for starting Numpy generated random numbers in a well-defined initial state.
np.random.seed(20)

# Necessary for starting core Python generated random numbers in a well-defined state.
rn.seed(1984)

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

# Will make random number generation in the TensorFlow backend have a well-defined initial state.
tf.set_random_seed(8)

# Import MuStARD specific library
pathname = os.path.abspath(os.path.dirname(sys.argv[0])) + "/../../lib/python"
sys.path.insert(0, pathname)
from Files import Format

def main():

	input_directory = sys.argv[1]
	output_directory = sys.argv[2]
	input_mode = sys.argv[3]

	# Report the GPU device information, if present
	sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

	print "\n\t\t\tConverting data to one-hot encoding."

	the_one = dict()
	the_one["train"] = dict()
	the_one["test"] = dict()
	the_one["valid"] = dict()
	modes = input_mode.split(",")
	for mode in modes:

		train_file = input_directory + "/train." + mode + ".tab.gz"
		test_file = input_directory + "/test." + mode + ".tab.gz"
		valid_file = input_directory + "/validation." + mode + ".tab.gz"

		print "\t\t\tLoading train data - " + mode
		the_one["train"][mode], the_one["train"]["labels"], the_one["train"]["l_dim"], the_one["train"]["seq_dim"] = Format.to_one_hot(train_file, mode, "train", "unweighted")
		#the_one["train"][mode], the_one["train"]["labels"], the_one["train"]["l_dim"], the_one["train"]["seq_dim"], the_one["train"]["wghts"] = Format.to_one_hot(train_file, mode, "train", "weighted")

		print "\t\t\tLoading test data - " + mode
		the_one["test"][mode], the_one["test"]["labels"], the_one["test"]["l_dim"], the_one["test"]["seq_dim"] = Format.to_one_hot(test_file, mode, "train", "unweighted")

		print "\t\t\tLoading validation data - " + mode
		the_one["valid"][mode], the_one["valid"]["labels"], the_one["valid"]["l_dim"], the_one["valid"]["seq_dim"] = Format.to_one_hot(valid_file, mode, "train", "unweighted")

	output_directory = output_directory + "/" + "_".join(modes)
	if not os.path.exists(output_directory):
		os.mkdir(output_directory, 0755)

	# Lists with multiple values of each hyperparameter to be used in a grid-search type of approach of optimization
	batch_size_list = [256]
	dropout_list = [0.2]
	lr_list = [0.0001]
	filter_num_list = [40]

	# Loops that will apply the same architecture on different combinations of hyperparameters and save the results in different directories
	for tmp_batch_size in batch_size_list:

		for tmp_dropout in dropout_list:

			for tmp_lr in lr_list:

				for tmp_filter_num in filter_num_list:

					tmp_output_directory = output_directory + "/batch" + str(tmp_batch_size) + "_dropout" + str(tmp_dropout) + "_lr" + str(tmp_lr) + "_filters" + str(tmp_filter_num)
					if not os.path.exists(tmp_output_directory):
						os.mkdir(tmp_output_directory, 0755)
					else:

						print "\n\t\t\t############################################################################################"
						print "\t\t\tSKIPPING Training with params - batch: " + str(tmp_batch_size) + " dropout: " + str(tmp_dropout) + " lr: " + str(tmp_lr) + " filters: " + str(tmp_filter_num)
						print "\t\t\t############################################################################################"
						continue

					print "\n\t\t\t############################################################################################"
					print "\t\t\tTraining with params - batch: " + str(tmp_batch_size) + " dropout: " + str(tmp_dropout) + " lr: " + str(tmp_lr) + " filters: " + str(tmp_filter_num)
					print "\t\t\t############################################################################################"

					call_model(the_one, modes, tmp_batch_size, tmp_dropout, tmp_lr, tmp_filter_num, tmp_output_directory)

					K.clear_session()


def encoder_sequence_branch(sequence_input, tmp_filter_num, tmp_dropout):

	x = Conv1D(filters = tmp_filter_num * 2, kernel_size = 16, strides = 1, padding = "same")(sequence_input)
	x = LeakyReLU()(x)
	x = BatchNormalization()(x)
	x = MaxPooling1D(pool_size = 2, padding = "same")(x)
	x = Dropout(rate = tmp_dropout, noise_shape = None, seed = None)(x)

	x = Conv1D(filters = tmp_filter_num, kernel_size = 12, strides = 1, padding = "same")(x)
	x = LeakyReLU()(x)
	x = BatchNormalization()(x)
	x = MaxPooling1D(pool_size = 2, padding = "same")(x)
	x = Dropout(rate = tmp_dropout, noise_shape = None, seed = None)(x)

	x = Conv1D(filters = int(tmp_filter_num / 2), kernel_size = 8, strides = 1, padding = "same")(x)
	x = LeakyReLU()(x)
	x = BatchNormalization()(x)
	x = MaxPooling1D(pool_size = 2, padding = "same")(x)
	x = Dropout(rate = tmp_dropout, noise_shape = None, seed = None)(x)

	#x = Bidirectional(LSTM(16, return_sequences=True, activation='tanh', recurrent_activation='hard_sigmoid', dropout=tmp_dropout, recurrent_dropout=tmp_dropout))(x)
	#x = Bidirectional(LSTM(8, return_sequences=True, activation='tanh', recurrent_activation='hard_sigmoid', dropout=tmp_dropout, recurrent_dropout=tmp_dropout))(x)
	sequence_output = Flatten()(x)

	return sequence_output


def encoder_fold_branch(fold_input, tmp_filter_num, tmp_dropout):

	x = Conv1D(filters = tmp_filter_num * 2, kernel_size = 30, strides = 1, padding = "same")(fold_input)
	x = LeakyReLU()(x)
	x = BatchNormalization()(x)
	x = MaxPooling1D(pool_size = 2, padding = "same")(x)
	x = Dropout(rate = tmp_dropout, noise_shape = None, seed = None)(x)

	x = Conv1D(filters = tmp_filter_num, kernel_size = 20, strides = 1, padding = "same")(x)
	x = LeakyReLU()(x)
	x = BatchNormalization()(x)
	x = MaxPooling1D(pool_size = 2, padding = "same")(x)
	x = Dropout(rate = tmp_dropout, noise_shape = None, seed = None)(x)

	x = Conv1D(filters = int(tmp_filter_num / 2), kernel_size = 10, strides = 1, padding = "same")(x)
	x = LeakyReLU()(x)
	x = BatchNormalization()(x)
	x = MaxPooling1D(pool_size = 2, padding = "same")(x)
	x = Dropout(rate = tmp_dropout, noise_shape = None, seed = None)(x)

	#x = Bidirectional(LSTM(16, return_sequences=True, activation='tanh', recurrent_activation='hard_sigmoid', dropout=tmp_dropout, recurrent_dropout=tmp_dropout))(x)
	#x = Bidirectional(LSTM(8, return_sequences=True, activation='tanh', recurrent_activation='hard_sigmoid', dropout=tmp_dropout, recurrent_dropout=tmp_dropout))(x)
	fold_output = Flatten()(x)

	return fold_output


def encoder_conservation_branch(conservation_input, tmp_filter_num, tmp_dropout):

	x = Conv1D(filters = tmp_filter_num * 2, kernel_size = 20, strides = 1, padding = "same")(conservation_input)
	x = LeakyReLU()(x)
	x = BatchNormalization()(x)
	x = MaxPooling1D(pool_size = 2, padding = "same")(x)
	x = Dropout(rate = tmp_dropout, noise_shape = None, seed = None)(x)

	x = Conv1D(filters = tmp_filter_num, kernel_size = 15, strides = 1, padding = "same")(x)
	x = LeakyReLU()(x)
	x = BatchNormalization()(x)
	x = MaxPooling1D(pool_size = 2, padding = "same")(x)
	x = Dropout(rate = tmp_dropout, noise_shape = None, seed = None)(x)

	x = Conv1D(filters = int(tmp_filter_num / 2), kernel_size = 10, strides = 1, padding = "same")(x)
	x = LeakyReLU()(x)
	x = BatchNormalization()(x)
	x = MaxPooling1D(pool_size = 2, padding = "same")(x)
	x = Dropout(rate = tmp_dropout, noise_shape = None, seed = None)(x)

	#x = Bidirectional(LSTM(16, return_sequences=True, activation='tanh', recurrent_activation='hard_sigmoid', dropout=tmp_dropout, recurrent_dropout=tmp_dropout))(x)
	#x = Bidirectional(LSTM(8, return_sequences=True, activation='tanh', recurrent_activation='hard_sigmoid', dropout=tmp_dropout, recurrent_dropout=tmp_dropout))(x)
	conservation_output = Flatten()(x)

	return conservation_output


def call_model(the_one, modes, tmp_batch_size, tmp_dropout, tmp_lr, tmp_filter_num, tmp_output_directory):
	"""This function assembles the model and performs the training.
	The architecture example in this function uses two different convolutional layers with shared parameters.
	One layer is applied on the forward strand and the second one on the reverse strand.
	The output of these layers is concatenated before forwarded to the biderectional LSTM layer.
	"""

	sequence_inputs = []
	sequence_outputs = []

	label_dim = the_one["train"]["l_dim"]
	sequence_dim = the_one["train"]["seq_dim"]

	train_x = []
	train_y = the_one["train"]["labels"]
	test_x = []
	test_y = the_one["test"]["labels"]
	valid_x = []
	valid_y = the_one["valid"]["labels"]

	#print label_dim
	#print sequence_dim
	#print train_y
	#print test_y
	#print valid_y
	#sys.exit()

	for mode in modes:

		if mode == "sequence":

			sequence_input = Input(shape = (sequence_dim, 4))

			sequence_output = encoder_sequence_branch(sequence_input, tmp_filter_num, tmp_dropout)

			sequence_inputs.append(sequence_input)
			sequence_outputs.append(sequence_output)

		elif mode == "RNAfold":

			fold_input = Input(shape = (sequence_dim, 3))

			fold_output = encoder_fold_branch(fold_input, tmp_filter_num, tmp_dropout)

			sequence_inputs.append(fold_input)
			sequence_outputs.append(fold_output)

		elif mode == "conservation":

			conservation_input = Input(shape = (sequence_dim, 1))

			conservation_output = encoder_conservation_branch(conservation_input, tmp_filter_num, tmp_dropout)

			sequence_inputs.append(conservation_input)
			sequence_outputs.append(conservation_output)

		train_x.append(the_one["train"][mode])
		test_x.append(the_one["test"][mode])
		valid_x.append(the_one["valid"][mode])

	print "\t\t\tConnecting network components."

	#concatenated = keras.layers.concatenate(sequence_outputs, axis = -1)
	concatenated = []
	if len(modes) == 1:
		concatenated = sequence_outputs[0]
	else:
		concatenated = keras.layers.concatenate(sequence_outputs)

	#out = Dense(units = 150, kernel_initializer = "he_normal", kernel_regularizer = regularizers.l2(0.00001))(concatenated)
	out = Dense(units = 100)(concatenated)
	out = LeakyReLU()(out)
	out = BatchNormalization()(out)
	out = Dropout(rate = tmp_dropout, noise_shape = None, seed = None)(out)
	#out = Dense(units = 100, kernel_initializer = "he_normal", kernel_regularizer = regularizers.l2(0.00001))(out)
	out = Dense(units = 75)(out)
	out = LeakyReLU()(out)
	out = BatchNormalization()(out)
	out = Dropout(rate = tmp_dropout, noise_shape = None, seed = None)(out)
	#out = Dense(units = 50, kernel_initializer = "he_normal", kernel_regularizer = regularizers.l2(0.00001))(out)
	out = Dense(units = 50)(out)
	out = LeakyReLU()(out)
	out = BatchNormalization()(out)
	out = Dropout(rate = tmp_dropout, noise_shape = None, seed = None)(out)
	out = Dense(units = label_dim, activation = "softmax")(out)

	classification_model = Model(sequence_inputs, out)
	if len(modes) == 1:
		classification_model = Model(sequence_inputs[0], out)

	print "\t\t\tCompiling network components."
	sgd = SGD(lr = tmp_lr,
			decay = 1e-6,
			momentum = 0.9,
			#clipnorm = 1.,
			#clipvalue = 0.5,
			nesterov = True)
	#rmsprop = RMSprop(lr=0.01)
	#adam = Adam(lr = 0.01)
	classification_model.compile(optimizer = sgd,
			loss = "categorical_crossentropy",
			metrics = ["accuracy"])

	#model.summary()
	#sys.exit()

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

	if len(modes) == 1:
		print "\t\t\tTraining network."
		history = classification_model.fit(train_x[0], train_y,
			batch_size = tmp_batch_size,
			epochs = 600,
			verbose = 1,
			validation_data = (valid_x[0], valid_y),
			callbacks = [mcp, earlystopper, csv_logger])
			#class_weight = the_one["train"]["wghts"])

		print "\t\t\tTesting network."
		tresults = classification_model.evaluate(test_x[0], test_y,
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
	else:
		print "\t\t\tTraining network."
		history = classification_model.fit(train_x, train_y,
			batch_size = tmp_batch_size,
			epochs = 600,
			verbose = 1,
			validation_data = (valid_x, valid_y),
			callbacks = [mcp, earlystopper, csv_logger])
			#class_weight = the_one["train"]["wghts"])

		print "\t\t\tTesting network."
		tresults = classification_model.evaluate(test_x, test_y,
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


if __name__ == '__main__':
	main()
