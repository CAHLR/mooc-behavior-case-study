import datetime
import pandas as pd
import json
import numpy as np
import keras
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers import Input, Merge, merge
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import TimeDistributed
import csv

def process_time(time):
    return datetime.datetime.strptime(time[:-6], '%Y-%m-%dT%H:%M:%S.%f' if '.' in time[:-6] else '%Y-%m-%dT%H:%M:%S')

class MOOC_Data(object):
    """
    Accepts raw edx log of json objects
    Converts to pandas dataframe
    Has several filtering methods
    Ultimate output is sorted_data to be fed to an Abstract_Mapped_To_Indices_Data 
    """
    def __init__(self, log_file, course_axis_file):
        """
        Most important attribute is sorted_data
        sorted_data is a pandas dataframe with columns from the original 
        Attributes:
        sorted_data: DataFrame of all rows of data sorted by time.
                     Columns are: 
        course_axis: DataFrame of the course axis
        problem_check_data: DataFrame of all rows of data that are done by users with a problem check within time constraints        
        Reads in json event log and converts to pandas DataFrame
        """
        working_data = []
        fail_count = 0
        with open(log_file) as data_file:
            for line in data_file.readlines():
                try:
                    data = json.loads(line)
                    working_data.append(line)
                except:
                    fail_count+=1
                    print("Failed to read line:")
                    print(line)
                    continue
        print("Number of failed lines: ", fail_count)
        print("Length of successfuly read data: ", len(working_data))
        print("Converting to pandas dataframe now...")
        data_json_str = "[" + ','.join(working_data) + "]" #Converts json to one large string, since that is what pandas needs
        data_df = pd.read_json(data_json_str)
#        self.raw_data = data_df
        self.sorted_data = data_df.sort_values('time')
        self.course_axis = pd.read_csv(course_axis_file)
        print("Completed loading json file and converted to Pandas DataFrame processing")

    def output_to_disk(self, output_name):
        print("Outputting sorted dataframe to disk.")
        self.sorted_data.to_csv(output_name)

    def filter_data_problem_check(self, minimum_problem_checks = 1):
        """
        Returns a COPY of sorted_data that is filtered to only include users with a problem check.
        If you want to overwrite .sorted_data, then reassign it to the returned dataframe of this method
        """
        data_df = self.sorted_data
        print("Number of rows before filtering for problem check:", len(data_df))
        only_problem_check = data_df[data_df['event_type'] == 'problem_check']
        users_with_problem_check = set(only_problem_check.username)
        df_only_users_with_problem_check = data_df[data_df['username'].isin(users_with_problem_check)].sort_values('time')
        print("Number of rows from users with a problem check:", len(df_only_users_with_problem_check))
        return df_only_users_with_problem_check

    def filter_data_navigation_only(self):
        """
        Returns a COPY of sorted_data that is filtered to only include rows from navigation events.
        """
        data_df = self.sorted_data
        print("Number of rows before filtering by only navigation:", len(data_df))
        seq_rows = data_df[data_df['event_type'].isin(['seq_goto', 'seq_next', 'seq_prev'])]
        print('seq rows', len(seq_rows))
        slash_rows = data_df[data_df['event_type'].str.startswith('/')]
        slash_rows = slash_rows[slash_rows['event_type'].str.contains('courseware')]
        print('slash rows with courseware', len(slash_rows))
        navigation_rows = pd.concat([seq_rows, slash_rows]).sort_values('time')
        print("Length of navigation rows:", len(navigation_rows))
        return navigation_rows
    
    def filter_data_by_time(self, earliest_time = datetime.datetime(1900, 1, 1, 0, 0, 0, 000000), latest_time = datetime.datetime(2100, 12, 31, 23, 59, 59, 999999)):
        """
        Returns a COPY of sorted_data that is filtered to only include rows that are between earliest_time and latest_time
        """
        data_df = self.sorted_data
        print("Length of data before filtering by time:", len(data_df))
        if 'datetime_time' not in data_df.columns:
            data_df['datetime_time'] = data_df['time'].apply(process_time)
        data_df = data_df[data_df['datetime_time'] <= latest_time]
        data_df = data_df[data_df['datetime_time'] >= earliest_time]
        print("Length of data after filtering by time:", len(data_df))
        return data_df

class MOOC_Data_From_Disk(MOOC_Data):
    """
    From Disk means the sorted, processed dataframe is saved to a csv. Reads in a dataframe csv, not an event log.
    """
    def __init__(self, dataframe_file_name, course_axis_file):
        self.sorted_data = pd.read_csv(dataframe_file_name)
        self.course_axis = pd.read_csv(course_axis_file)
        print("Successfully read Dataframe from disk.", dataframe_file_name)

class Abstract_Bridge_Between_MOOC_Data_and_Embedding_Indices(object):
    """
    Accepts MOOC_Data object, which contains a sorted_data attribute as well as a course axis
    Creates a new internal DataFrame named pre_index_data that contains columns for userid, timestamp, and unique representation of action. Can contain additional columns.
    The unique representation of action is what should be converted to indices (for output into embedding layer).
    Outputs X y used to train model, along with vocab_size for keras model.
    """
    def __init__(self, MOOC_Data):
        """
        """
        self.mooc_data = MOOC_Data
        self.pre_index_data = pd.DataFrame(columns=['user', 'timestamp', 'unique_representation_of_event'])
        self.mappings = None
        self.current_full_indices = []
        self.current_full_indices_userids = []

    @property
    def r_mappings(self):
        r_mapping = {v: k for k, v in self.mappings.items()}
        return r_mapping

    def populate_pre_index(self):
        """
        Abstract Method
        """
        print("Warning: Need to implement populate_pre_index")
        return -1

    def expose_x_y(self):
        """
        Abstract Method
        Returns X and y numpy arrays that will be fed into Keras Model
        """
        print("WARNING: NEED TO IMPLEMENT EXPOSE_X_Y")
        return -1

def construct_vertical(sequential, chapter, vertical=-1):
    return '/' + chapter + '/' + sequential + '/' + str(vertical)

class Vertical_Output(Abstract_Bridge_Between_MOOC_Data_and_Embedding_Indices):
    def __init__(self, MOOC_Data):
        Abstract_Bridge_Between_MOOC_Data_and_Embedding_Indices.__init__(self, MOOC_Data)

    def populate_mappings_based_on_verticals_in_course_axis(self):
        course_axis = self.mooc_data.course_axis
        mapping = {1: 'pre_start'}
        current_index = 2
        for action in list(course_axis[course_axis.category=='vertical'].path):
            mapping[current_index] = action
            current_index += 1
        r_mapping = {v: k for k, v in mapping.items()}
        self.mappings = mapping

    def populate_time_spent_in_pre_index_data(self):
        if not self.pre_index_data:
            print("Warning: pre_index_data is empty.")
            return
        groupedbyuser = self.pre_index_data.groupby('user')
        for user, data in groupedbyuser:
            for row in data.iterrows():
                123412341234
                print("not yet implemented")
    
    def create_full_indices_based_on_pre_index_data_ignoring_time_spent(self):
        """
        Returns pre_index_data mapped to indices (list of lists), as well as corresponding list of user ids
        """
        if not self.mappings:
            print("Error: Mappings not yet populates")
            return -1
        list_of_indices = []
        list_of_indices_userids = []
        grouped_by_user = self.pre_index_data.groupby('user')
        for user_id, data in grouped_by_user:
            list_of_indices_userids.append(user_id)
            current_user_indices = []
            full_indices = [self.r_mappings[url] for url in list(data.unique_representation_of_event)]
            list_of_indices.append(full_indices)
        return list_of_indices, list_of_indices_userids
        
    def prepend_1_to_current_full_indices(self):
        """
        MUTATES self.current_full_indices such that a 1 is prepended to all lists
        Will not prepend a 1 if there is already a 1 at the start
        """
        for seq in self.current_full_indices:
            if seq[0] == 1:
                continue
            else:
                seq.reverse()
                seq.append(1)
                seq.reverse()
                continue

        
    def remove_contiguous_repeats_from_pre_index_data(self, keep_highest_time_spent = True):
        """
        Returns a copy of pre_index_data where contiguous repeats are removed
        """
        grouped_by_user = self.pre_index_data.groupby('user')
        data_to_dataframe = [] #will be 4 columns, matching pre_index_data
        for user_id, data in grouped_by_user:
            previous_element = None
            for row in data.iterrows():
                index = row[0]
                values = row[1]
                u = values.user
                t = values.timestamp
                url = values.unique_representation_of_event
                t_spent = values.time_spent
                if not previous_element:
                    previous_element = url
                    data_to_dataframe.append([u, t, url, t_spent])
                    continue
                else:
                    if url == previous_element:
                        if keep_highest_time_spent:
                            currently_recorded_time_spent = data_to_dataframe[-1][-1]
                            if isinstance(t_spent, str):
                                #t_spent is therefore 'endtime'
                                data_to_dataframe[-1][-1] = t_spent
                            elif t_spent > currently_recorded_time_spent:
                                data_to_dataframe[-1][-1] = t_spent
                            else:
                                continue
                        continue
                    else:
                        previous_element = url
                        data_to_dataframe.append([u, t, url, t_spent])
        temp_df = pd.DataFrame(data_to_dataframe, columns = self.pre_index_data.columns)
        return temp_df

    def populate_time_spent(self):
        """
        Returns a copy of self.pre_index_data with the time_spent column populated with integer between 0 and 3
        """
        pre_index_data = self.pre_index_data
        grouped_by_user = pre_index_data.groupby('user')
        data_to_append = []
        for user_id, data in grouped_by_user:
            new_time_sequence = []
            timestamps = [process_time(elem) for elem in list(data.timestamp)]
            for i in range(0, len(timestamps) - 1):
                current_time = timestamps[i]
                next_time = timestamps[i+1]
                second_difference = (next_time - current_time).total_seconds()
                new_time_sequence.append(second_difference)
            new_time_sequence.append('endtime')

            i = 0
            for row in data.iterrows():
                index = row[0]
                values = row[1]
                user = values.user
                timestamp = timestamps[i]
                rep = values.unique_representation_of_event
                time_spent = new_time_sequence[i]
                data_to_append.append([user, timestamp, rep, time_spent])
                i+=1

        temp_df = pd.DataFrame(data_to_append, columns = ['user', 'timestamp', 'unique_representation_of_event', 'time_spent'])
        return temp_df


    def populate_pre_index_data(self):
        """
        Populates self.pre_index_data with a dataframe that includes a vertical_url column,
        such that navigational data is resolved to a specific course URL
        """
        course_axis = self.mooc_data.course_axis
        ordered_vertical_paths = list(course_axis[course_axis.category == 'vertical'].path)
        sequential_paths = list(course_axis[course_axis.category == 'sequential'].path)
        chapter_paths = list(course_axis[course_axis.category == 'chapter'].path)
        chapter_set =  set([elem[1:] for elem in list(course_axis[course_axis.category == 'chapter'].path)])
        all_paths = list(course_axis.path)
        seq_counts = [0, 0, 0]
        every_category = [0, 0, 0, 0]
        prev_next_conversions = [0, 0]
        data_to_append = [] #should be 4 element per row, with columns user timestamp vertical_url time_spent
        #will eventually append data_to_append to the pre_index_data dataframe, to eventually convert to x, y in expose_x_y
        sequential_to_chap = {}
        prev_next_conversions = [0, 0]
        for path in sequential_paths:
            seq = path.split('/')[-1]
            chap = path.split('/')[-2]
            sequential_to_chap[seq] = chap
        
        grouped_by_user = self.mooc_data.sorted_data.groupby('username')
        for user_id, data in grouped_by_user:
            chapter_location = {} #key is chapter, value is [sequential, vertical]
            sequential_location = {}
            for chapter in chapter_set:
                for p in sequential_paths:
                    if chapter in p:
                        chapter_location[chapter] = [p.split('/')[-1], 1]
                        break            
            for sequential in sequential_to_chap:
                sequential_location[sequential] = 1
            for row in data.iterrows():
                index = row[0]
                values = row[1]
                et = values.event_type
                is_seq = False
                seq_events = ['seq_goto', 'seq_next', 'seq_prev']
                for event in seq_events:
                    if event in et:
                        is_seq = True
                        break
                if is_seq:
                    e = json.loads(values.event)
                    action = str(e['new']) + '_' + str(et) + '_' + e['id'].split('/')[-1]
                    split = action.split('_')
                    new_vertical = int(split[0])
                    split = action.split('@')
                    seq_id = split[-1]
                    if seq_id not in sequential_to_chap:
                        continue
                    chap_id = sequential_to_chap[seq_id]
                    new_action = construct_vertical(seq_id, chap_id, new_vertical)
                    test_string = new_action #basically testing if the new potential vertical is an actual possible vertical according to course axis
                    if test_string not in all_paths:
                        if event == 'seq_prev':
                            corresponding_vertical_index = ordered_vertical_paths.index(construct_vertical(seq_id,chap_id,new_vertical+1))
                            new_action = ordered_vertical_paths[corresponding_vertical_index-1]
                            split = new_action.split('/')
                            new_vertical = split[-1]
                            seq_id = split[2]
                            chap_id = split[1]
                            prev_next_conversions[0]+=1
                        elif event == 'seq_next':
                            corresponding_vertical_index = ordered_vertical_paths.index(construct_vertical(seq_id,chap_id,new_vertical-1))
                            new_action = ordered_vertical_paths[corresponding_vertical_index+1]
                            split = new_action.split('/')
                            new_vertical = split[-1]
                            seq_id = split[2]
                            chap_id = split[1]
                            prev_next_conversions[1]+=1
                        else:
                            raise Exception()

                    sequential_location[seq_id] = new_vertical
                    if event == 'seq_prev':
                        seq_counts[0] += 1
                    elif event == 'seq_next':
                        seq_counts[1] += 1
                    elif event == 'seq_goto':
                        seq_counts[2] += 1
                    else:
                        raise Exception()
                    every_category[0] += 1
                else:
                    action = et
                    split = action.split('/')
                    last_elem = split[-1]
                    if len(last_elem) == 1 or len(last_elem) == 2:
                        seq_id = split[-2]
                        if seq_id not in sequential_to_chap:
                            continue
                        chap_id = split[-3]
                        if chap_id not in chapter_location:
                            continue
                        try:
                            new_vertical = int(last_elem)
                        except:
                            new_vertical = last_elem
                        new_action = construct_vertical(seq_id, chap_id, new_vertical)
                        test_string = new_action
                        if test_string not in all_paths:
                            print("Nonsense vertical: ", test_string, "Debug code: 1")
                            new_vertical = 1 #resolve nonsense direct vertical to vertical 1
                            new_action = construct_vertical(seq_id, chap_id, new_vertical)
                        sequential_location[seq_id] = new_vertical
                        every_category[1] += 1
                    elif split[-3] == 'courseware':
                        chap_id = split[-2]
                        if chap_id not in chapter_location:
                            print("couldn't find this chapter in course axis:", action)
                            continue
                        seq_id = chapter_location[chap_id][0]
                        new_vertical = int(chapter_location[chap_id][1])
                        new_action = construct_vertical(seq_id, chap_id, int(new_vertical))
                        test_string = new_action
                        if test_string not in all_paths:
                            print("Nonsense vertical: ", test_string, "Debug code: 2")
                            new_vertical = 1 #resolve nonsense direct vertical to vertical 1
                            new_action = construct_vertical(seq_id, chap_id, new_vertical)
                        every_category[2] += 1
                    #is a chapter event with no related sequential
                    else:
                    #is a sequential event with no vertical
                        seq_id = split[-2]
                        chap_id = split[-3]
                        if seq_id not in sequential_location:
                            continue
                        if chap_id not in chapter_location:
                            continue
                        new_vertical = sequential_location[seq_id]
                        new_action = construct_vertical(seq_id, chap_id, int(new_vertical))
                        test_string = new_action
                        if test_string not in all_paths:
                            print("Nonsense vertical: ", test_string, "Debug code: 3")
                            new_vertical = 1 #resolve nonsense direct vertical to vertical 1
                            new_action = construct_vertical(seq_id, chap_id, new_vertical)
                        every_category[3] += 1
                chapter_location[chap_id] = [seq_id, new_vertical]
                time = values.time
                data_to_append.append([user_id, time, new_action, 'not yet calculated'])
        temp_df = pd.DataFrame(data_to_append, columns = ['user', 'timestamp', 'unique_representation_of_event', 'time_spent'])
        self.pre_index_data = temp_df
        print('seq_prev, seq_next, seq_goto counts:', seq_counts)
        print('seq events, direct vertical in url, only chapter in url, chapter and sequential in url counts:', every_category)
        print('times seq_prev was used on first vertical in sequence, ditto for seq_next counts:',prev_next_conversions)
    
    def expose_x_y(self, max_len = 5000, min_len = 3):
        """
        Returns X, y numpy arrays based on current_full_indices
        """
        x_windows = [seq[:-1] for seq in self.current_full_indices if len(seq) >= min_len]
        y_windows = [seq[1:] for seq in self.current_full_indices if len(seq) >= min_len]
        X = sequence.pad_sequences(x_windows, maxlen = max_len, padding = 'post', truncating = 'post')
        padded_y_windows = sequence.pad_sequences(y_windows, maxlen=max_len, padding = 'post', truncating = 'post')
        self.padded_y_windows = padded_y_windows
        y = np.zeros((len(padded_y_windows), max_len, len(self.mappings)), dtype = np.bool)
        for i, output in enumerate(padded_y_windows):
            for t, resource_index in enumerate(output):
                if resource_index == 0:
                    continue
                else:
                    y[int(i), int(t), int(resource_index)-1] = 1
        return X, y

class MOOC_Keras_Model(object):
    """
    
    """
    def __init__(self):
        """
        """
        self.keras_model = None
        self.X = None
        self.padded_y_windows = None
        self.y = None
        self.model_params = None
        self.model_histories = []
        self.embedding_vocab_size = None
        self.best_epoch = None
        self.previous_val_loss = []

    def import_data(self, X, y, additional_params = []):
        """
        """
    
    def set_model_name(self, name):
        if not self.model_params:
            print("WARNING: Create LSTM model before setting model name.")
            return -1
        self.model_name = name + self.model_params_to_string

    @property
    def model_params_to_string(self):
        mp = self.model_params
        return '_' + str(mp['layers']) + '_' + str(mp['lrate']) + '_' + str(mp['hidden_size']) + '_' + str(mp['opt']) + '_' + str(mp['e_size']) + '_' + str( mp['output_dim']) + '_' + str(mp['input_len']) + '_' + str(mp['embedding_vocab_size'])

    def create_basic_lstm_model(self, layers, lrate, hidden_size, opt, e_size, output_dim, input_len, embedding_vocab_size):
        """
        Returns a LSTM model
        """
        print('building a functional API model')

        self.model_params = {'layers': layers, 'lrate': lrate, 'hidden_size': hidden_size, 'opt': opt, 'e_size': e_size, 'output_dim': output_dim, 'input_len': input_len, 'embedding_vocab_size': embedding_vocab_size}

        main_input = Input(shape=(input_len,), name = 'main_input', dtype='int32')
        x = Embedding(output_dim = e_size, input_dim = embedding_vocab_size+1, input_length = input_len, mask_zero = True)(main_input)
        for i in range(layers):
            print("adding layer " + str(i))
            x = LSTM(hidden_size, dropout_W = 0.2, return_sequences = True)(x)
        main_loss = TimeDistributed(Dense(output_dim, activation='softmax'))(x)
        model = Model(input=[main_input], output = [main_loss])
        opt = opt(lr = lrate)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

    def load_keras_weights_from_disk(self):
        """
        """

    def early_stopping_model_fit(self, train_x, train_y, validation_data, epoch_limit = 200, loss_nonimprove_limit = 3, batch_size = 64, save_models_to_folder = None):
        """
        """
        early_stopping_met = False
        for i in range(epoch_limit):
            print("epoch:", i)
            current_history = self.keras_model.fit(train_x, train_y, batch_size = batch_size, nb_epoch = 1, validation_data = validation_data)
            current_history = current_history.history
            validation_loss = current_history['val_loss'][0]
            validation_accuracy_dictionary = self.compute_validation_accuracy(validation_data, b_s = batch_size)
            average_of_average_accuracy = np.mean(validation_accuracy_dictionary['averages'])
            accuracy = validation_accuracy_dictionary['accuracy']
            self.previous_val_loss.append(validation_loss)
            if len(self.previous_val_loss) > loss_nonimprove_limit:
                min_val_loss = min(self.previous_val_loss)
                recent_losses = self.previous_val_loss[-loss_nonimprove_limit-1:]
                print(recent_losses)
                if min(recent_losses) > min_val_loss:
                    early_stopping_met = True
                if validation_loss == min_val_loss:
                    self.best_epoch = i
                    self.best_average_of_average_accuracy = average_of_average_accuracy
                    self.best_accuracy = accuracy                    
            if early_stopping_met:
                print("Early stopping reached.")
                print("Best epoch according to validation loss:", self.best_epoch)
                print("Best epoch's accuracy:", self.best_accuracy)
                print("Best epoch's average accuracy:", self.best_average_of_average_accuracy)
                return

    def compute_validation_accuracy(self, validation_data, b_s = 64):
        """
        """
        validation_x = validation_data[0]
        validation_y = validation_data[1]
        just_x_indices = validation_x
        if isinstance(validation_x, list):
            just_x_indices = validation_x[0]
        predictions = self.keras_model.predict(validation_x, batch_size = b_s)

        per_student_accuracies = []
        total_correct_predictions = 0
        total_incorrect_predictions = 0

        for student_sequence_index, current_x in enumerate(just_x_indices):
            corresponding_predictions = list(predictions[student_sequence_index])
            corresponding_answers = list(validation_y[student_sequence_index])
            current_student_correct = 0
            current_student_incorrect = 0
            for prediction_index in range(len(current_x)):
                current_value = current_x[prediction_index]
                if current_value == 0:
                    continue
                else:
                    current_softmax = list(corresponding_predictions[prediction_index]) #softmax probability distribution
                    best_prediction = current_softmax.index(max(current_softmax)) + 1
                    correct_answer = list(corresponding_answers[prediction_index]).index(1) + 1
                    is_correct = best_prediction == correct_answer
                    
                    if is_correct:
                        current_student_correct += 1
                        total_correct_predictions += 1
                    else:
                        current_student_incorrect +=1
                        total_incorrect_predictions += 1
            acc = float(current_student_correct) / (current_student_incorrect + current_student_correct)
            per_student_accuracies.append(acc)
        total_val_acc = float(total_correct_predictions) / (total_correct_predictions + total_incorrect_predictions)
        print("Total validation accuracy:", total_val_acc)
        print("Average accuracy:", np.mean(per_student_accuracies))
        return_dict = {}
        return_dict['accuracy'] = total_val_acc
        return_dict['averages'] = per_student_accuracies
        return return_dict

"""
    def simple_model_fit(self, epochs = 10, batch_size = 64, validation_proportion = 0.1):
        for i in range(epochs):
            print("epoch:", i)
            validation_i = int(len(self.X) * .9)
            hist = self.keras_model.fit(self.X[:validation_i], self.y[:validation_i], batch_size = batch_size, nb_epoch = 1, validation_data = (self.X[validation_i:], self.y[validation_i:]))
            self.model_histories.append(hist)
"""
