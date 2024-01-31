'''
classe process che contiene risorse ed esegue task
'''
from datetime import datetime
import simpy
from resource import Resource
from MAINparameters import*
import math
from call_LSTM import Predictor


class SimulationProcess(object):

    def __init__(self, env: simpy.Environment, params):
        self.env = env
        self.params = params
        self.date_start = params.START_SIMULATION
        self.resource = self.define_single_resource()
        self.resource_events = self.define_resource_events(env)
        self.resource_trace = simpy.Resource(env, math.inf)
        self.buffer_traces = []
        self.predictor = Predictor((params.MODEL_PATH_PROCESSING, params.MODEL_PATH_WAITING), self.params)
        self.predictor.predict()

    def define_single_resource(self):
        set_resource = list(self.params.ROLE_CAPACITY.keys())
        dict_res = dict()
        for res in set_resource:
            res_simpy = Resource(self.env, res, self.params.ROLE_CAPACITY[res][0], self.params.ROLE_CAPACITY[res][1], self.date_start)
            print(res, res_simpy.capacity)
            dict_res[res] = res_simpy

        return dict_res

    def get_occupations_resource(self, resource):
        occup = []
        if self.params.FEATURE_ROLE == 'all_role':
            for key in self.resource:
                if key != 'SYSTEM':
                    occup.append(self.resource[key].get_resource().count / self.resource[key].capacity)
        else:
            occup.append(self.resource[resource].get_resource().count / self.resource[resource].capacity)
        return occup

    def get_occupations_single_resource(self, resource):
        return self.resource[resource].get_resource().count / self.resource[resource].capacity

    def get_resource(self, resource_label):
        return self.resource[resource_label] ### ritorna tipo resource mio

    def get_resource_event(self, task):
        return self.resource_events[task]

    def get_resource_trace(self):
        return self.resource_trace

    def define_resource_events(self, env):
        resources = dict()
        for key in self.params.INDEX_AC:
            resources[key] = simpy.Resource(env, math.inf)
        return resources

    def get_predict_processing(self, cid, pr_wip, transition, ac_wip, rp_oc, time, queue):
        return self.predictor.processing_time(cid, pr_wip, transition, ac_wip, rp_oc, time, queue)

    def get_predict_waiting(self, cid, pr_wip, transition, rp_oc, time, queue):
        if queue < 0:
            return self.predictor.predict_waiting(cid, pr_wip, transition, rp_oc, time, queue)
        else:
            return self.predictor.predict_waiting_queue(cid, pr_wip, transition, rp_oc, time, queue)
