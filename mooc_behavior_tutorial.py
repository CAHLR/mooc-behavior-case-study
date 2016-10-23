#This file trains an RNN on student actions starting with a student action log
import csv
import json
import pandas
#from datetime import datetime
import numpy as np
import datetime
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import TimeDistributed
import csv
#import mooc_constants
import pandas as pd

#Step 1: Specify the mooc log file.
log_file_name = 'DelftX_AE1110x_2T2015-events.log'
sorted_file = 'ORDERED_DelftX_AE1110x_2T2015-events.log'
#Step 2: Sort by time, in case it is not already sorted.
def generate_ordered_event_copy(event_log_file_name):
    """
    Takes in an event log file, with one action per row, orders the actions by time, and then writes a new file.

    ../data/BerkeleyX_Stat_2.1x_1T2014-events.log

    """
    output_name = "ORDERED_" + event_log_file_name.split('/')[-1]

    all_data_paired_with_time = []
    with open(event_log_file_name) as data_file:
        for line in data_file.readlines():
            try:
                data = json.loads(line)
            except:
                print(line)
                continue
            time_element = data['time']
            if '.' in time_element:
                date_object = datetime.strptime(time_element[:-6], '%Y-%m-%dT%H:%M:%S.%f')
            else:
                date_object = datetime.strptime(time_element[:-6], '%Y-%m-%dT%H:%M:%S')
            all_data_paired_with_time.append((line, date_object))
    print('sorting by time ...')
    s = sorted(all_data_paired_with_time, key=lambda p: p[1])
    to_output = [pair[0] for pair in s]
#    return to_output

    print("dumping json to",output_name)
    with open(output_name, mode='w') as f:
        for line in to_output:
            f.write(line)
    return output_name

#Step 3: Preprocess to only grab rows we are interested in. For the purpose of this example, we only want actions related to which page the student is at. Thus, we exclude events such as quiz taking, video viewing, etc.
def generate_courseware_and_seq_events(log_file, earliest_time = datetime.datetime.min, latest_time = datetime.datetime.max, require_problem_check = False, bug_test = False):
    """
    log_file is the name of a sorted log of student actions where each row is a json object
    9/13 update: produces (pageview index, time) pairs in list
    """
    user_to_pageview_and_time_pairs = {}
    user_to_all_json = {}
    with open(log_file) as data_file:
        seq_events = ['seq_next', 'seq_prev', 'seq_goto']
        users_that_have_a_problem_check = set()
        for line in data_file.readlines():
            data = json.loads(line)
            user = data['username']
            if user not in user_to_pageview_and_time_pairs:
                user_to_pageview_and_time_pairs[user] = []
                if bug_test:
                    user_to_all_json[user] = []
            if bug_test:
                user_to_all_json[user].append(data)
            t = datetime.datetime.strptime(data['time'][:-6], '%Y-%m-%dT%H:%M:%S.%f' if '.' in data['time'][:-6] else '%Y-%m-%dT%H:%M:%S')
            if not earliest_time <= t <= latest_time:
                continue
            et = data['event_type']
            if et == 'problem_check':
                users_that_have_a_problem_check.add(user)
                continue
            if et in seq_events:
                event = json.loads(data['event'])
                action = str(event['new']) + '_' + str(et) + '_' + event['id'].split('/')[-1]
            elif et[0] == '/' and 'courseware' in et and 'data:image' not in et and '.css' not in et:
                action = et
            else:
                continue
            user_to_pageview_and_time_pairs[user].append((action, t))
    filtered_user_to_pageviews = {user: pairs for user, pairs in user_to_pageview_and_time_pairs.items() if user in users_that_have_a_problem_check}
    if bug_test:
        return filtered_user_to_pageviews, user_to_all_json
    return filtered_user_to_pageviews

#Step 4: Convert output from step 3 into a URL-esque representation of where the student is at.
course_axis = pandas.read_csv('axis_DelftX_AE1110x_2T2015.csv')
print('finished axis')

def convert_to_verticals(user_and_action_dict, course_axis, drop_chapter_events = False):
    print("drop_chapter_events is set to", drop_chapter_events)
    chap_drops = 0
    seq_events = ['seq_next', 'seq_prev', 'seq_goto']
    seq_counts = [0, 0, 0]
    every_category = [0, 0, 0, 0]
    prev_next_conversions = [0, 0]
    only_sequentials = course_axis[course_axis.category == 'sequential']
    all_paths = list(course_axis.path)
    sequential_paths = list(only_sequentials.path)
    ordered_vertical_paths = list(course_axis[course_axis.category == 'vertical'].path)
    sequential_to_chap = {}
    special_gotos = set()
    for path in sequential_paths:
        seq = path.split('/')[-1]
        chap = path.split('/')[-2]
        sequential_to_chap[seq] = chap
#    chapter_set = set()
    only_chapters = course_axis[course_axis.category == 'chapter']
    chapter_set = set([elem[1:] for elem in list(only_chapters.path)])
    print(chapter_set)
    def construct_vertical(sequential, chapter, vertical=-1):
        """
        .
        """
        return '/' + chapter + '/' + sequential + '/' + str(vertical)
    user_to_pageviews = {}
    for userid, pairlist in user_and_action_dict.items():
        chapter_location = {} #key is chapter, value is [sequential, vertical]
        sequential_location = {}
        for chapter in chapter_set:
            for p in sequential_paths:
                if chapter in p:
                    chapter_location[chapter] = [p.split('/')[-1], 1]
                    break
        for sequential in sequential_to_chap:
            sequential_location[sequential] = 1
        new_actions = []
        new_times = []
        for pair in pairlist:
#            print(pair)
#            print(userid)
            action = pair[0]
            time = pair[1]
#            if action == '/courses/BerkeleyX/Stat_2.1x/1T2014/courseware/':
                #print('skipping', action)
#                continue
            is_seq = False
            for event in seq_events:
                if event in action:
                    is_seq = True
#                    print(event)
                    break
            if is_seq:
#                print(action)
                split = action.split('_')
                new_vertical = int(split[0])
                split = action.split('@')
                seq_id = split[-1]
#                print("LOOKING AT VERTICAL:", new_vertical, "WITH SEQUENTIAL ID:", seq_id)
                if seq_id not in sequential_to_chap:
#                    print("skipping", seq_id)
                    continue
                chap_id = sequential_to_chap[seq_id]
                new_action = construct_vertical(seq_id, chap_id, new_vertical)
                test_string = new_action
                if test_string not in all_paths:
#                    print(test_string)
                    if event == 'seq_prev':
                        corresponding_vertical_index = ordered_vertical_paths.index(construct_vertical(seq_id,chap_id,new_vertical+1))
                        new_action = ordered_vertical_paths[corresponding_vertical_index-1]
                        split = new_action.split('/')
                        new_vertical = split[-1]
                        seq_id = split[2]
                        chap_id = split[1]
                        prev_next_conversions[0]+=1
                    elif event == 'seq_next':
#                        if new_vertical == 7 and '4555126bb263441a99fa8eea3771801c' in action:
#                            print("skipping new element...")
#                            print(action)
#                            continue
                        corresponding_vertical_index = ordered_vertical_paths.index(construct_vertical(seq_id,chap_id,new_vertical-1))
                        new_action = ordered_vertical_paths[corresponding_vertical_index+1]
                        split = new_action.split('/')
                        new_vertical = split[-1]
                        seq_id = split[2]
                        chap_id = split[1]
                        prev_next_conversions[1]+=1
                    else:
                        special_gotos.add(action)
                        continue
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
                split = action.split('/')
#                if 'courseware' in action:
#                    print(split)
        #        print(action)
                last_elem = split[-1]
                if len(last_elem) == 1 or len(last_elem) == 2:
         #           print(action)
                    #already has a direct vertical
                    seq_id = split[-2]
                    if seq_id not in sequential_to_chap:
                        continue
                    chap_id = split[-3]
                    try:
                        new_vertical = int(last_elem)
                    except:
                        new_vertical = 999999
#                        print("skipping...", action)
#                        continue
                    new_action = construct_vertical(seq_id, chap_id, new_vertical)
                    test_string = new_action
                    if test_string not in all_paths:
                        print("Nonsense vertical: ", test_string, 1)
                        new_vertical = 1 #resolve nonsense direct vertical to vertical 1
                        new_action = construct_vertical(seq_id, chap_id, new_vertical)
                    sequential_location[seq_id] = new_vertical
                    every_category[1] += 1
                elif split[-3] == 'courseware':
                    #is a chapter event with no related sequential
                    if drop_chapter_events:
                        print("skipping chapter event...", action)
                        chap_drops += 1
                        continue
                    chap_id = split[-2]
                    if chap_id not in chapter_location:
#                        print("couldn't find this chapter in course axis:", action)
                        continue
                    seq_id = chapter_location[chap_id][0]
                    new_vertical = int(chapter_location[chap_id][1])
                    new_action = construct_vertical(seq_id, chap_id, int(new_vertical))
                    test_string = new_action
                    if test_string not in all_paths:
                        print("Nonsense vertical: ", test_string, 2)
                        new_vertical = 1 #resolve nonsense direct vertical to vertical 1
                        new_action = construct_vertical(seq_id, chap_id, new_vertical)

                    every_category[2] += 1
                    #is a chapter event with no related sequential
                else:
                    #is a sequential event with no vertical
                    seq_id = split[-2]
                    chap_id = split[-3]
                    if seq_id not in sequential_location:
                        #print("skipping", seq_id)
                        #print(action)
#                        print("skipping", seq_id)
                        continue
                    new_vertical = sequential_location[seq_id]
                    new_action = construct_vertical(seq_id, chap_id, int(new_vertical))
                    test_string = new_action
                    if test_string not in all_paths:
                        print("Nonsense vertical: ", test_string, 3)
                        new_vertical = 1 #resolve nonsense direct vertical to vertical 1
                        new_action = construct_vertical(seq_id, chap_id, new_vertical)

                    every_category[3] += 1
            chapter_location[chap_id] = [seq_id, new_vertical]
            new_actions.append(new_action)
            new_times.append(time)
        user_to_pageviews[userid] = list(zip(new_actions, new_times))
    print("number of chap drops:", chap_drops)
    print(special_gotos)
    print(seq_counts)
    print(every_category)
    print(prev_next_conversions)
    return {k:v for k, v in user_to_pageviews.items() if v}

print("generating first actions")
u_p = generate_courseware_and_seq_events(sorted_file)
print("resolving to verticals")
u_to_page = convert_to_verticals(u_p, course_axis, drop_chapter_events=False)

unique_set = set()
for u in u_to_page:
    val = [p[0] for p in u_to_page[u]]
    for v in val:
        unique_set.add(v)


mapping = {1: 'pre_start'}
current_index = 2
for action in list(course_axis[course_axis.category=='vertical'].path):
    mapping[current_index] = action
    current_index += 1
r_mapping = {v: k for k, v in mapping.items()}


mapped_actions = {}
for u in u_to_page:
    pageviews = [p[0] for p in u_to_page[u]]
    times = [p[1] for p in u_to_page[u]]
    converted_pageviews = [r_mapping[elem] for elem in pageviews]
    converted_pageviews.reverse()
    converted_pageviews.append(r_mapping['pre_start'])
    converted_pageviews.reverse()
    converted_times = times
    converted_times.reverse()
    converted_times.append("PRESTARTTIME")
    converted_times.reverse()
    mapped_actions[u] = list(zip(converted_pageviews,converted_times))

no_repeat_mapped_actions = {}
old_lens = []
new_lens = []
def remove_continguous_repeats(pairlist):
    """
    """
    previous_elem = False
    result_list = []
    for i in range(len(pairlist)):
        current_elem = pairlist[i]
        if previous_elem:
            if current_elem[0] == previous_elem:
                continue
            else:
                previous_elem = current_elem[0]
                result_list.append(current_elem)
        else:
            previous_elem = current_elem[0]
            result_list.append(current_elem)
    return result_list

for u in mapped_actions:
    mapped = mapped_actions[u]
    old_lens.append(len(mapped))
    no_repeat_mapped_actions[u] = remove_continguous_repeats(mapped)
    new_lens.append(len(no_repeat_mapped_actions[u]))

mapped_actions = no_repeat_mapped_actions
actions = [[p[0] for p in lst] for lst in list(mapped_actions.values())]
times = [[p[1] for p in lst] for lst in list(mapped_actions.values())]
userids = mapped_actions.keys()
mappings = mapping

window_len = 1000
x_windows = []
y_windows = []

for a in actions:
    if len(a) < 2:
        continue
    x_windows.append(a[:-1])
    y_windows.append(a[1:])

vocab_size = len(mappings)
X = sequence.pad_sequences(x_windows, maxlen=window_len, padding='post', truncating='post')
padded_y_windows = sequence.pad_sequences(y_windows, maxlen=window_len, padding = 'post', truncating ='post')
y = np.zeros((len(padded_y_windows), window_len, vocab_size), dtype = np.bool)
for i, output in enumerate(padded_y_windows):
    for t, resource_index in enumerate(output):
        if resource_index == 0:
            continue
        else:
            y[int(i), int(t), int(resource_index)-1] = 1

def construct_model(e_size, hidden_size, layers, lrate, opt):
    e_size = e_size
    HIDDEN_SIZE = hidden_size
    LAYERS = layers
    lrate = lrate
    print('building a model')
    model = Sequential()
    model.add(Embedding(vocab_size+1, e_size, input_length = window_len, mask_zero = True))
    for i in range(LAYERS):
        print("adding layer " + str(i))
        model.add(LSTM(HIDDEN_SIZE, return_sequences = True,dropout_W=0.2))
    model.add(TimeDistributed(Dense(vocab_size)))
    model.add(Activation('softmax'))
    opt = opt(lr = lrate)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    if isinstance(opt, Adagrad):
        o = 'adagrad'
    elif isinstance(opt, RMSprop):
        o = 'rmsprop'
    modelname = "oracle_directvertical_modelweights_"+str(LAYERS)+'_'+str(HIDDEN_SIZE)+'_'+str(lrate)+'_'+str(e_size)+'_'+o+'_'
    return model, modelname

embedding_size = [80]
hsize = [128]
layers = [2]
lrate = [0.01]
opt = [RMSprop]
#opt = [Adagrad]


for esize in embedding_size:
    for hidden_size in hsize:
        for l in layers:
            for lr in lrate:
                for o in opt:
                    model, modelname = construct_model(esize, hidden_size, l, lr, o)
                    hist = model.fit(X, y, batch_size = BATCHSIZE, nb_epoch = 5)
