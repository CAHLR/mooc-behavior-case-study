import json
import datetime
import pandas as pd


def process_data(file_name, earliest_time = datetime.datetime.min, latest_time = datetime.datetime.max):
    # read the entire file into a python array
    with open(file_name, 'r') as f:
        data = f.readlines()
    # remove the trailing "\n" from each line
    data = map(lambda x: x.rstrip(), data)
    data_json_str = "[" + ','.join(data) + "]"
    # now, load it into pandas
    data_df = pd.read_json(data_json_str)
    #only consider users that have event_type as problem_check
    users_with_problem_check = data_df[data_df['event_type'] == 'problem_check']

    #processes event_type. et here refers to event type
    def process_et(et):
        seq_events = ['seq_next', 'seq_prev', 'seq_goto']
        if et in seq_events:
            event = json.loads(data['event'])
            return str(event['new']) + '_' + str(et) + '_' + event['id'].split('/')[-1]
        elif et[0] == '/' and 'courseware' in et and 'data:image' not in et and '.css' not in et:
            return et

    #processes the time
    def process_time(time):
        return datetime.datetime.strptime(time[:-6], '%Y-%m-%dT%H:%M:%S.%f' if '.' in time[:-6] else '%Y-%m-%dT%H:%M:%S')

    users_with_problem_check['processed_event_type'] = users_with_problem_check['event_type'].apply(process_et)
    users_with_problem_check['processed_time'] = users_with_problem_check['time'].apply(process_time)
    users_with_problem_check = users_with_problem_check[users_with_problem_check['processed_time'] <= latest_time]
    users_with_problem_check = users_with_problem_check[users_with_problem_check['processed_time'] >= earliest_time]
    print(users_with_problem_check.head())
    return users_with_problem_check


#implement what he does in this earliest_time <= t <= latest_time, and put above code in a function