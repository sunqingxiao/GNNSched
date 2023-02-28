import importlib

import torch
import time

### Class
class ModelSummary():
    def __init__(self, model_name, comp_cache_info):
        """ """
        self.task_name, self.data_name, self.graph_name = model_name[0], model_name[1], model_name[2]
        print('{} {} {}'.format(self.task_name, self.data_name, self.graph_name))   
        self.comp_cache_size, self.comp_cache_offset = comp_cache_info[0], comp_cache_info[1]
        # Allocate fake memory for parameters
        self.cuda_stream_for_computation = torch.cuda.Stream()
        self.load_model()

    def execute(self):
        start = time.time()
        with torch.cuda.stream(self.cuda_stream_for_computation):
            torch.cuda.insert_shared_cache_for_computation(self.comp_cache_size, self.comp_cache_offset)

        return self.func(self.model, self.graph)

    def load_model(self):
        model_module = importlib.import_module('task.' + self.task_name)
        self.model, self.func, self.graph = model_module.import_task(self.data_name, self.graph_name)
