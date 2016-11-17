from data_processing import *

sample_log_file = "DelftX_AE1110x_2T2015-events.log"
sample_course_axis = "axis_DelftX_AE1110x_2T2015.csv"


#This is some sample code for using the work-in-progress MOOC data processing code.
#Fundamentally, the code is just to help the user make sense of the different levels of abstraction between raw MOOC data and running a model to answer a research question.


#Step 1: Parsing from Log data to a Pandas DataFrame
#Filter log events as desired for research question


print("Reading in sample log file:", sample_log_file)
my_data = MOOC_Data(sample_log_file, sample_course_axis)

print("len of data:", len(my_data.sorted_data))

my_data.sorted_data = my_data.filter_data_problem_check()
my_data.sorted_data = my_data.filter_data_navigation_only()
my_data.sorted_data = my_data.filter_data_by_time()
my_data.output_to_disk("delft2T2015dataframefiltered.csv")

#Could also load dataframe from disk
"""
dataframefromdisk = "delft2T2015dataframefiltered.csv"
my_data = MOOC_Data_From_Disk(dataframefromdisk, sample_course_axis)
"""

#Step 2: Bridge between log events and tokenized indices. 
#Continue to filter data as needed for research question, but now at the numpy array level for input to Keras.

my_verticals = Vertical_Output(my_data)
print(len(my_verticals.mooc_data.sorted_data))

my_verticals.populate_mappings_based_on_verticals_in_course_axis()
my_verticals.populate_pre_index_data()
print('length of pre index data', len(my_verticals.pre_index_data))
#print(my_verticals.pre_index_data.columns)
#print(my_verticals.pre_index_data.loc[0])
#print(my_verticals.pre_index_data.loc[0].vertical_url)
my_verticals.pre_index_data = my_verticals.remove_contiguous_repeats_from_pre_index_data()
my_verticals.pre_index_data.to_csv("pre_index_data.csv")

"""
#Can read pre index data from disk if saved previously
my_verticals.pre_index_data = pd.read_csv("pre_index_data.csv")
"""

my_verticals.current_full_indices, my_verticals.current_full_indices_userids = my_verticals.create_full_indices_based_on_pre_index_data_ignoring_time_spent()
my_verticals.prepend_1_to_current_full_indices()

print("Len of full indices:", len(my_verticals.current_full_indices))
print("Example sequence:", my_verticals.current_full_indices[5])

sequence_max_len = 1500
X, y = my_verticals.expose_x_y(max_len = sequence_max_len)
print("Length of exposed X:", len(X))
print("Length of a sample sequence:", len(X[20]))


#Step 3: Build a Keras LSTM Model and train on data from the Step 2 Bridge.

print("Building keras model and attempting to train...")
my_keras_model = MOOC_Keras_Model()
my_keras_model.embedding_vocab_size = len(my_verticals.mappings)
my_keras_model.keras_model = my_keras_model.create_basic_lstm_model(2, 0.01, 128, Adagrad, 100, my_keras_model.embedding_vocab_size, sequence_max_len, my_keras_model.embedding_vocab_size)
my_keras_model.set_model_name('Baseline_Input_Output')

hill_climbing_proportion = 0.1
hill_climbing_index = int(len(my_keras_model.X) * (1 - hill_climbing_proportion))
train_x = X[:hill_climbing_index]
train_y = y[:hill_climbing_index]
validation_data = (X[hill_climbing_index:], y[hill_climbing_index:])

my_keras_model.early_stopping_model_fit(train_x, train_y, validation_data, loss_nonimprove_limit = 3)

#Step 4: Build a recommendation oracle or other downstream task that utilizes the keras model.
