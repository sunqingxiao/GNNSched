from queue import Queue
from multiprocessing import Process
import torch
import time

from gnnsched.worker_common import ModelSummary
from util.util import timestamp

class WorkerProc(Process):
    def __init__(self, pipe):
        super(WorkerProc, self).__init__()
        self.pipe = pipe
        
    def run(self):
        timestamp('worker', 'start')

        # Warm up CUDA and get shared cache
        torch.randn(128, device='cuda')
        torch.cuda.recv_shared_cache() # pylint: disable=no-member
        timestamp('worker', 'share_gpu_memory')
        
        while True:  # dispatch workers for task execution
            agent, model_name, comp_cache_info = self.pipe.recv()
            model_summary = ModelSummary(model_name, comp_cache_info)
            timestamp('worker', 'import models')

            # start doing training
            with torch.cuda.stream(model_summary.cuda_stream_for_computation):
                infertime = model_summary.execute()

            print('{} {} {} slo: {} arritime: {} infertime: {}'.format(model_name[0], model_name[1], model_name[2], "%.4f"%(float(model_name[3])), "%.4f"%(float(model_name[4])), "%.4f"%(infertime)))

            self.pipe.send('FNSH')
            agent.send(b'FNSH')
            torch.cuda.clear_shared_cache()

            timestamp('worker_comp_thd', 'complete')
