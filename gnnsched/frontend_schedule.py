import threading
import torch
import importlib
import time

import torch.multiprocessing as mp

from util.util import timestamp
from gnnsched.worker import WorkerProc
from gnnsched.policy import *

class FrontendScheduleThd(threading.Thread):
    def __init__(self, model_list, qin):
        super(FrontendScheduleThd, self).__init__()
        self.model_list = model_list
        self.qin = qin

    def run(self):
        timestamp('schedule', 'start')

        # Load models
        comp_bytes, arrtimes, qos = {}, {}, {}
        
        for model_name in self.model_list:
            hash_name = hash('{}_{}_{}'.format(model_name[0], model_name[1], model_name[2]))
            comp_bytes[hash_name] = self._load_model(model_name)
            arrtimes[hash_name] = float(model_name[4])
            qos[hash_name] = float(model_name[3])

        timestamp('schedule', 'load_model')
       
        job_list = []
        while True:
            # Get request           
            agent, task_name, data_name, graph_name = self.qin.get()
            job_list.append([agent, task_name, data_name, graph_name])
            timestamp('schedule', 'get_request')
            if len(job_list) == len(self.model_list):
                break
        
        start = time.time()
        # schedule according to policies
        reorder_job_list = []
        if POLICY_OPTION == 0: # in-order
            reorder_job_list = policy_base(job_list, comp_bytes, arrtimes)
        elif POLICY_OPTION == 1: # shortest QoS target first
            reorder_job_list = policy_sqtf(job_list, comp_bytes, arrtimes, qos)
        elif POLICY_OPTION == 2: # balanced QoS target
            reorder_job_list = policy_bqt(job_list, comp_bytes, arrtimes, qos)
        else:  print('Wrong scheduling option!')
        print('Task scheduling time: {}'.format(time.time()-start))


#        for group_iter in range(len(reorder_job_list)):
#            print('num of elements: {}'.format(len(reorder_job_list[group_iter])))
#        print('after here')

        for group_iter in range(len(reorder_job_list)):
            comp_cache_offset = 0 # initialize offset of computations

            # Create workers
            worker_list, cur_w_idx = [], 0
            for _ in range(len(reorder_job_list[group_iter])):
                p_parent, p_child = mp.Pipe()
                worker = WorkerProc(p_child)
                worker.start()
                torch.cuda.send_shared_cache()
                worker_list.append((p_parent, worker))
                timestamp('frontend', 'create_worker')

            for job in reorder_job_list[group_iter]:
                agent, task_name, data_name, graph_name = job[0], job[1], job[2], job[3]
                # get next worker to work on request
                new_pipe, _ = worker_list[cur_w_idx]
                cur_w_idx += 1

                # send request to new worker
                model_name = []
                for model in self.model_list:
                    if model[0] == task_name and model[1] == data_name and model[2] == graph_name:
                        model_name = model

                comp_cache_size = comp_bytes[hash('{}_{}_{}'.format(task_name, data_name, graph_name))]

                comp_cache_info = [comp_cache_size, comp_cache_offset]
                comp_cache_offset += comp_cache_size # update comp cache offset

                new_pipe.send((agent, model_name, comp_cache_info))
                timestamp('schedule', 'notify_new_worker')

            res_counter = 0
            while True:
                cur_w_idx %= len(worker_list)
                new_pipe, worker = worker_list[cur_w_idx]
                cur_w_idx += 1
                # Recv response
                if new_pipe.poll():
                    res = new_pipe.recv()
                    res_counter += 1
                    worker.terminate()
               
                if res_counter == len(reorder_job_list[group_iter]):
                    break

    def _load_model(self, model_name):
        # Import parameters
        model_module = importlib.import_module('task.' + model_name[0])
        comp_total_bytes = model_module.import_parameters(model_name[1], model_name[2])

        return comp_total_bytes
