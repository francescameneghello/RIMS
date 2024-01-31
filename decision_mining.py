import pm4py
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from lxml import etree
from collections import defaultdict
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import os
import json
from pm4py.objects.log.importer.xes import importer as xes_importer
import pickle
from sklearn.metrics import f1_score
from sklearn import tree
from matplotlib import pyplot as plt
from hyperopt import tpe
import numpy as np
from hyperopt import Trials, hp, fmin
from hyperopt.pyll import scope


def retrieve_aligned_trace(log, net, im, fm):
    parameters = {}
    parameters[
        alignments.Variants.VERSION_STATE_EQUATION_A_STAR.value.Parameters.PARAM_ALIGNMENT_RESULT_IS_SYNC_PROD_AWARE] = True
    aligned_traces = alignments.apply_log(log, net, im, fm, parameters=parameters)

    aligned_log = []
    for trace in aligned_traces:
        new_trace = []
        for event in trace['alignment']:
            if event[0][1] != '>>':
                new_trace.append(event[0][1])
        aligned_log.append(new_trace)

    return aligned_log

def retrive_XOR_petrinet(path_petrinet):
    tree = etree.parse(path_petrinet)
    root = tree.getroot()
    net = root.find('net')
    page = net.find('page')

    xor_petri = defaultdict(list)

    for place in page.findall('place'):
        xor_petri[place.get('id')] = []

    for arc in page.findall('arc'):
        if arc.get('source') in xor_petri:
            xor_petri[arc.get('source')].append(arc.get('target'))

    remove = [key for key in xor_petri if len(xor_petri[key]) < 2]
    for r in remove:
        del xor_petri[r]
    return xor_petri


def extract_event_from_trace(trace, activity):
    i = len(trace)-1
    while i >= 0:
        if trace[i]['concept:name'] == activity:
            return trace[i]
        else:
            i -= 1
    return None


def extract_enriched_alignment_last(original_trace, aligned_log, attrib):
    new_trace = []
    for trace in aligned_log:
        for e in trace['alignment']:
            if e[0][1] != '>>':
                if e[1][0] != '>>':
                    event_original = extract_event_from_trace(original_trace, e[1][0])
                    event = [e[0][1], event_original['role'], event_original['start_timestamp'],
                             event_original['end_timestamp'], event_original['st_wip'], event_original['st_task_wip'],
                             event_original['queue'], event_original['rc_op'], event_original['amount']]
                else:
                    event = [e[0][1], 'invisible']
                for a in attrib:
                    event.append(original_trace.attributes[a])
                new_trace.append(event)
    return new_trace


def extract_enriched_alignment(original_trace, aligned_log, attrib):
    new_trace = []
    for trace in aligned_log:
        for e in trace['alignment']:
            if e[0][1] != '>>':
                if e[1][0] != '>>':
                    event_original = extract_event_from_trace(original_trace, e[1][0])
                    if 'org:resource' in event_original:
                        event = [e[0][1], event_original['org:resource']]
                    else:
                        event = [e[0][1], 'Role 0']
                else:
                    event = [e[0][1], 'invisible']
                for a in attrib:
                    event.append(original_trace.attributes[a])
                new_trace.append(event)
    return new_trace


def create_prefix(traces, transitions):
    prefix_traces = {key: [] for key in transitions}
    for trace in traces:
        prefix = []
        for e in trace:
            if e[0] in transitions:
                prefix_traces[e[0]].append(prefix.copy())
            prefix.append(e)
    return prefix_traces


def find_max_length(prefix_traces):
    maximum = []
    for key in prefix_traces:
        list_len = [len(i) for i in prefix_traces[key]]
        maximum.append(max(list_len) if len(list_len) > 0 else 0)
    return max(maximum)


def trace_filter_decision_point(aligned_log, list_transitions):
    trace_filter = []
    for trace in aligned_log:
        for element in list_transitions:
            for event in trace:
                if element == event[0]:
                    trace_filter.append(trace.copy())
    return trace_filter


def retrieve_resource_to_role(path_role):
    resource_to_role = {}
    if os.path.exists(path_role):
        with open(path_role) as file:
            data = json.load(file)
            for idx, key in enumerate(data["roles"]):
                for resource in data["roles"][key]:
                    resource_to_role[resource] = int(key[len(key) - 1])
    return resource_to_role


def simple_encoding(prefix_traces, resource_to_role, attrib, activity2n):
    max_lenght = find_max_length(prefix_traces)
    columns = ['e_' + str(i) for i in range(1, max_lenght + 1)] + ['last_resource']
    columns += [a for a in attrib] + ['label']
    traces = []
    for key in prefix_traces:
        for trace in prefix_traces[key]:
            encoding_trace = [activity2n[e[0]] for e in trace]
            if len(trace) < max_lenght:
                encoding_trace = encoding_trace + ([activity2n['PAD']] * (max_lenght - len(trace)))
            length = len(trace) - 1
            flag = True
            while length > 0 and flag:
                if trace[length][1] != 'invisible':
                    res = trace[length][1]
                    if res in resource_to_role:
                        encoding_trace.append(resource_to_role[res])
                    else:
                        encoding_trace.append(1)
                    flag = False
                length -= 1
            if len(encoding_trace) == max_lenght:
                encoding_trace.append(0)
            for i in range(0, len(attrib)):
                encoding_trace += [trace[0][i + 2]]
            encoding_trace = encoding_trace + [key]
            traces.append(encoding_trace)

    df = pd.DataFrame(traces, columns=columns)
    return max_lenght, df


def frequency_encoding(net, prefix_traces):
    transitions_petri = [trans.name for trans in net.transitions]
    columns = transitions_petri + ['label']

    traces = []
    for key in prefix_traces:
        for trace in prefix_traces[key]:
            encoding_trace = []
            for t in transitions_petri:
                encoding_trace.append(trace.count(t))
            traces.append(encoding_trace + [key])
    df = pd.DataFrame(traces, columns=columns)

    return df

def del_other_traces(trace_log, caseid):
    index_to_delete = []
    for trace in trace_log:
        if trace.attributes['concept:name'] != caseid:
            index_to_delete.append(trace.attributes['concept:name'])
    del trace_log[index_to_delete[0]]


#NAME = 'BPI_Challenge_2012_W_Two_TS'
#path_log = "/Users/francescameneghello/Desktop/BPI_Challenge_2012_W_Two_TS_AMOUNT.xes"
#path_petrinet = "/Users/francescameneghello/Documents/GitHub/RIMS/BPI_Challenge_2012_W_Two_TS/rims/BPI_Challenge_2012_W_Two_TS.pnml"
#path_role = "/Users/francescameneghello/Documents/GitHub/RIMS/BPI_Challenge_2012_W_Two_TS/rims/BPI_Challenge_2012_W_Two_TS_dispr_meta.json"
#ATTRIB = ['AMOUNT_REQ']

NAME = 'confidential_1000'
path_log = "confidential_1000/confidential_1000.xes"
path_petrinet = NAME + "/rims/" + NAME + ".pnml"
path_role = NAME + "/rims/" + NAME + "_diapr_meta.json"
ATTRIB = []

log = xes_importer.apply(path_log)
log = pm4py.filter_event_attribute_values(log, "lifecycle:transition", ["complete"], level="event", retain=True)
net, im, fm = pm4py.read_pnml(path_petrinet)
decision_points = retrive_XOR_petrinet(path_petrinet)
aligned_traces = retrieve_aligned_trace(log, net, im, fm)
parameters = {alignments.Variants.VERSION_STATE_EQUATION_A_STAR.value.Parameters.PARAM_ALIGNMENT_RESULT_IS_SYNC_PROD_AWARE: True}
aligned_log = []
for trace in log:
    new_trace = []
    caseid = trace.attributes['concept:name']
    trace_log = pm4py.filter_trace_attribute_values(log, 'concept:name', [caseid], retain=True)
    aligned_traces = alignments.apply_log(trace_log, net, im, fm, parameters=parameters)
    new_trace = extract_enriched_alignment(trace_log[0], aligned_traces, ATTRIB)
    aligned_log.append(new_trace)
print('######################## LOG aligned ########################')
data_to_save = {}
data_to_save['ATTRIB'] = ATTRIB
transitions_petri = [trans.name for trans in net.transitions] + ['PAD']
activity2n = {a: n for n, a in enumerate(transitions_petri)}
n2activity = {n: a for n, a in enumerate(transitions_petri)}
data_to_save['encoding_transitions'] = activity2n
print('decision_points', decision_points)
for decision in decision_points:
    print("############## Compute training for", decision, "decision point ##############")
    list_transitions = decision_points[decision]
    trace_filter = trace_filter_decision_point(aligned_log, list_transitions)
    prefix_traces = create_prefix(trace_filter, list_transitions)
    resource_to_role = retrieve_resource_to_role(path_role)
    padding, df = simple_encoding(prefix_traces, resource_to_role, ATTRIB, activity2n)
    data_to_save[decision] = {}
    data_to_save[decision]['transitions'] = list_transitions
    data_to_save[decision]['padding'] = padding
    y = df.label  # Target variable
    X = df.drop(['label'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    space = {'max_depth': scope.int(hp.quniform('max_depth', 1, 21, 1)),
             'min_samples_split': scope.int(hp.uniform('min_samples_split', 2, 11)),
             'min_samples_leaf': scope.int(hp.uniform('min_samples_leaf', 3, 26)),
             'criterion': hp.choice('criterion', ['gini', 'entropy', 'log_loss'])}

    criterion_dict = {0: 'gini', 1: 'entropy', 2: 'log_loss'}

    def objective(params):
        clf = DecisionTreeClassifier(**params)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return -f1_score(y_test, y_pred, average='macro')


    trials = Trials()
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=100,
                trials=trials)

    print("Best Hyperparameters:", best)
    clf = DecisionTreeClassifier(criterion=criterion_dict[best['criterion']],
                                 max_depth=int(best['max_depth']),
                                 min_samples_leaf=int(best['min_samples_leaf']),
                                 min_samples_split=int(best['min_samples_split']))
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    data_to_save[decision]['best_parameters'] = str(best)
    data_to_save[decision]['Accuracy'] = metrics.accuracy_score(y_test, y_pred)
    data_to_save[decision]['f1'] = f1_score(y_test, y_pred, average=None).tolist()
    data_to_save[decision]['f1_score_macro'] = f1_score(y_test, y_pred, average='macro')
    data_to_save[decision]['f1_score_micro'] = f1_score(y_test, y_pred, average='micro')
    data_to_save[decision]['f1_score_weighted'] = f1_score(y_test, y_pred, average='weighted')
    data_to_save[decision]['confusion_matrix'] = confusion_matrix(y_test, np.array(y_pred)).tolist()
    data_to_save[decision]['features_importances'] = {list(df.columns)[i]: v for i, v in
                                                      enumerate(clf.feature_importances_)}
    data_to_save[decision]['features_importances'] = tree.export_text(clf)
    #fig = plt.figure(figsize=(25, 20))
    #_ = tree.plot_tree(clf,
    #                   feature_names=list(df.columns)[:-1],
    #                   class_names=list(df.columns)[-1],
    #                   filled=True)

    if data_to_save[decision]['f1_score_macro'] > 0.60:
        pickle.dump(clf, open(decision +'.pkl', 'wb'))
        data_to_save[decision]['prediction'] = True
    else:
        data_to_save[decision]['prediction'] = False
        ### compute probability
        data_to_save[decision]['probability'] = {}
        tot = sum([len(prefix_traces[key]) for key in prefix_traces])
        for key in prefix_traces:
            data_to_save[decision]['probability'][key] = len(prefix_traces[key])/tot

with open(NAME + "_decision_points.json", "w") as outfile:
    json.dump(data_to_save, outfile, indent=len(data_to_save))
    

#########
'''path_log = "PurchasingExample/PurchasingExample.xes"

log = xes_importer.apply(path_log)
for trace in log:
    print(trace.attributes['concept:name'])
    caseid = trace.attributes['concept:name']
    trace_log = pm4py.filter_trace_attribute_values(log, 'concept:name', [caseid], retain=True)
    print(caseid, len(trace_log))'''