from datetime import timedelta
import simpy
import pm4py
import random
import numpy as np
from checking_process import SimulationProcess
from pm4py.objects.petri_net import semantics
from MAINparameters import*
import pickle
from numpy.random import choice

class Token(object):

    def __init__(self, id, params, process: SimulationProcess, attrib, NAME_EXPERIMENT, rp_feature='all_role'):
        self.id = id
        self.net, self.am, self.fm = pm4py.read_pnml(params.PATH_PETRINET)
        self.process = process
        self.start_time = params.START_SIMULATION
        self.pr_wip_initial = params.PR_WIP_INITIAL
        self.rp_feature = rp_feature
        self.params = params
        self.see_activity = False
        self.pos = 0
        self.attrib = attrib
        self.prefix = []
        self.last_role = 0
        self.NAME_EXPERIMENT = NAME_EXPERIMENT

    def read_json(self, path):
        with open(path) as file:
            data = json.load(file)
            self.ac_index = data['ac_index']

    def simulation(self, env: simpy.Environment, writer, type, syn=False):
        trans = self.next_transition(syn)
        ### register trace in process ###
        resource_trace = self.process.get_resource_trace()
        resource_trace_request = resource_trace.request()

        while trans is not None:
            if self.see_activity:
                yield resource_trace_request
            self.prefix.append(trans.name)
            features = None
            if trans.label is not None:
                buffer = [self.id, trans.label]
                ### call predictor for waiting time
                if trans.label in self.params.ROLE_ACTIVITY:
                    resource = self.process.get_resource(self.params.ROLE_ACTIVITY[trans.label])  ## definisco ruolo che deve eseguire (deve essere di tipo Resource
                else:
                    resource = self.process.get_resource('SYSTEM')
                transition = (self.params.INDEX_AC[trans.label], self.params.INDEX_ROLE[resource.get_name()])
                pr_wip_wait = self.pr_wip_initial + resource_trace.count
                rp_oc = self.process.get_occupations_resource(resource.get_name())
                if type == 'rims_plus':
                    request_resource = resource.request()
                if type == 'rims_plus':
                    if len(resource.queue) > 0:
                        queue = len(resource.queue[-1])
                    else:
                        queue = 0
                else:
                    queue = -1
                waiting = self.process.get_predict_waiting(str(self.id), pr_wip_wait, transition, rp_oc,
                                                           self.start_time + timedelta(seconds=env.now), queue)
                if self.see_activity:
                    yield env.timeout(waiting)
                if type == 'rims':
                    request_resource = resource.request()
                yield request_resource
                self.last_role = int(resource.get_name()[-1]) if resource.get_name() != 'SYSTEM' else 0
                ### register event in process ###
                resource_task = self.process.get_resource_event(trans.label)
                resource_task_request = resource_task.request()
                yield resource_task_request
                buffer.append(str(self.start_time + timedelta(seconds=env.now)))
                ### call predictor for processing time
                pr_wip = self.pr_wip_initial + resource_trace.count
                rp_oc = self.process.get_occupations_resource(resource.get_name())
                initial_ac_wip = self.params.AC_WIP_INITIAL[trans.label] if trans.label in self.params.AC_WIP_INITIAL else 0
                ac_wip = initial_ac_wip + resource_task.count
                duration = self.process.get_predict_processing(str(self.id), pr_wip, transition, ac_wip, rp_oc, self.start_time + timedelta(seconds=env.now), -1)
                if trans.label == 'Start' or trans.label == 'End':
                    yield env.timeout(0)
                else:
                    yield env.timeout(duration)
                buffer.append(str(self.start_time + timedelta(seconds=env.now)))
                buffer.append(resource.get_name())
                buffer.append(pr_wip_wait)
                buffer.append(ac_wip)
                buffer.append(queue)
                buffer.append(self.attrib)
                time = int((self.start_time + timedelta(seconds=env.now)).timestamp())

                
                features = [self.last_role, time, resource_trace.count + dict_mean['wip'], resource_task.count+ dict_mean[trans.label],
                            queue + dict_mean[resource.get_name()+ '_queue'],
                           self.process.get_occupations_single_resource(resource.get_name()) + dict_mean[trans.label+ '_oc']]
                #features = [self.last_role, time, pr_wip, ac_wip, queue,
                #                        self.process.get_occupations_single_resource(resource.get_name())]
                resource.release(request_resource)
                resource_task.release(resource_task_request)
                print(*buffer)
                writer.writerow(buffer)
                if trans.label != 'Start':
                    self.see_activity = True

            self.update_marking(trans)
            #trans = self.next_transition(syn)
            trans = self.next_transition_with_predictive(features)

        resource_trace.release(resource_trace_request)

    def update_marking(self, trans):
        self.am = semantics.execute(trans, self.net, self.am)

    def next_transition_with_predictive(self, features):
        all_enabled_trans = semantics.enabled_transitions(self.net, self.am)
        all_enabled_trans = list(all_enabled_trans)
        all_enabled_trans.sort(key=lambda x: x.name)
        label_element = str(list(self.am)[0])
        if len(all_enabled_trans) == 0:
            return None
        elif len(all_enabled_trans) > 1:
            path_decision_json = self.NAME_EXPERIMENT + '/decision_mining/last_payload/' + self.NAME_EXPERIMENT + '_decision_points_last_payload.json'
            #path_decision_json = self.NAME_EXPERIMENT + '/decision_mining/simple_index/' + self.NAME_EXPERIMENT + '_decision_points.json'
            with open(path_decision_json) as file:
                data = json.load(file)
                if data[label_element]["prediction"]:
                    path_model = self.NAME_EXPERIMENT + '/decision_mining/last_payload/' + label_element + '.pkl'
                    #path_model = self.NAME_EXPERIMENT + '/decision_mining/simple_index/' + label_element + '.pkl'
                    loaded_clf = pickle.load(open(path_model, 'rb'))
                    n2activity = data["encoding_transitions"]
                    prefix_encoding = self.prefix[-data[label_element]['padding']:] if len(self.prefix) > data[label_element]['padding'] else self.prefix
                    encoding = [n2activity[x] for x in prefix_encoding] + [n2activity['PAD']] * (
                                data[label_element]['padding'] - len(prefix_encoding)) #+ [self.last_role]
                    ###### aggiunta con feature di last payload
                    ##### ['last_resource', 'last_time', 'last_wip', 'last_task_wip', 'last_queue', 'last_rp_oc', .... attrib ....]
                    encoding += features
                    for a in self.attrib:
                        encoding.append(self.attrib[a])
                    #predicted = loaded_clf.predict(np.array(encoding).reshape(1, -1))[0]
                    prob = list(loaded_clf.predict_proba(np.array(encoding).reshape(1, -1)))
                    predicted = choice(loaded_clf.classes_, 1, prob)[0]
                    next = [y.name for y in all_enabled_trans].index(predicted)
                else:
                    #transitions = [key for key in data[label_element]["probability"]]
                    #prob = [data[label_element]["probability"][key] for key in data[label_element]["probability"]]
                    #label = random.choices(transitions, prob)[0]
                    #next = [y.name for y in all_enabled_trans].index(label)
                    next = random.choices(list(range(0, len(all_enabled_trans), 1)))[0]
            return all_enabled_trans[next]
        else:
            return all_enabled_trans[0]

    def next_transition(self, syn):
        all_enabled_trans = semantics.enabled_transitions(self.net, self.am)
        all_enabled_trans = list(all_enabled_trans)
        all_enabled_trans.sort(key=lambda x: x.name)
        label_element = str(list(self.am)[0])
        if len(all_enabled_trans) == 0:
            return None
        elif len(all_enabled_trans) > 1:
            if syn:
                prob = [0.50, 0.50]
                if label_element in ['exi_node_54ded9af-1e77-4081-8659-bd5554ae9b9d', 'exi_node_38c10378-0c54-4b13-8c4c-db3e4d952451', 'exi_node_52e223db-6ebf-4cc7-920e-9737fe97b655', 'exi_node_e55f8171-c3fc-4120-9ab1-167a472007b7']:
                    prob = [0.01, 0.99]
                elif label_element in ['exi_node_7af111c0-b232-4481-8fce-8d8278b1cb5a']:
                    prob = [0.99, 0.01]
                value = [*range(0, len(prob), 1)]
                next = int(random.choices(value, prob)[0])
            else:
                next = random.choices(list(range(0, len(all_enabled_trans), 1)))[0]
            return all_enabled_trans[next]
        else:
            return all_enabled_trans[0]

