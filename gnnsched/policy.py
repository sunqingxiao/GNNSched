import numpy as np
import collections
import math

#MEMORY_THRESH = 16 * 1024 * 1024 * 1024
MEMORY_ALLOCATED = 20 * 1024 * 1024 * 1024

MEMORY_THRESH = 20 * 1024 * 1024 * 1024
POLICY_OPTION = 2 # 0 BASE (in-order), 1 SQTF (shortest-QoS-target-first), 2 BQT (balanced-QoS-target)

# in-order
def policy_base(job_list, comp_bytes, arrtimes):
    reorder_job_list = []
    lasttime, group_counter, curtime, group_mc = -1, -1, 0, 0
    for i in range(len(job_list)):
        hash_name = hash('{}_{}_{}'.format(job_list[i][1], job_list[i][2], job_list[i][3]))
        group_mc += comp_bytes[hash_name]
        curtime = arrtimes[hash_name]
        if curtime != lasttime or group_mc > MEMORY_THRESH:
            reorder_job_list.append([])
            group_mc = comp_bytes[hash_name]
            group_counter += 1
        reorder_job_list[group_counter].append(job_list[i])
        lasttime = curtime

#    for i in range(len(reorder_job_list)):
#        print(reorder_job_list[i])

    return reorder_job_list


# shortest QoS target first
def policy_sqtf(job_list, comp_bytes, arrtimes, qos):
    par_job_list, par_comp_bytes, par_qos = [], [], []
    lasttime, group_counter, curtime = -1, -1, 0
    for i in range(len(job_list)):
        hash_name = hash('{}_{}_{}'.format(job_list[i][1], job_list[i][2], job_list[i][3]))
        curtime = arrtimes[hash_name]
        if curtime != lasttime:
            par_job_list.append([])
            par_comp_bytes.append([])
            par_qos.append([])
            group_counter += 1
        par_job_list[group_counter].append(job_list[i])
        par_comp_bytes[group_counter].append(comp_bytes[hash_name])
        par_qos[group_counter].append(qos[hash_name])
        lasttime = curtime
    
    reorder_job_list, group_counter = [], -1
    for i in range(len(par_job_list)):
        ascending = np.argsort(np.array(par_qos[i]))
        mcsum = sum(par_comp_bytes[i])
        groupthresh = math.ceil(mcsum / math.ceil(mcsum / MEMORY_THRESH))
        
        group_mc = MEMORY_THRESH
        for j in range(ascending.shape[0]):
            if group_mc > groupthresh:
                reorder_job_list.append([])
                group_mc = 0
                group_counter += 1

            reorder_job_list[group_counter].append(par_job_list[i][ascending[j]])
            group_mc += par_comp_bytes[i][ascending[j]]

#    for i in range(len(reorder_job_list)):
#        print(reorder_job_list[i])

    return reorder_job_list


# balenced QoS target
def policy_bqt(job_list, comp_bytes, arrtimes, qos):
    par_job_list, par_comp_bytes, par_qos = [], [], []
    lasttime, group_counter, curtime = -1, -1, 0
    for i in range(len(job_list)):
        hash_name = hash('{}_{}_{}'.format(job_list[i][1], job_list[i][2], job_list[i][3]))
        curtime = arrtimes[hash_name]
        if curtime != lasttime:
            par_job_list.append([])
            par_comp_bytes.append([])
            par_qos.append([])
            group_counter += 1
        par_job_list[group_counter].append(job_list[i])
        par_comp_bytes[group_counter].append(comp_bytes[hash_name])
        par_qos[group_counter].append(qos[hash_name])
        lasttime = curtime
    
    reorder_job_list, group_counter = [], -1
    for i in range(len(par_job_list)):
        ascending = np.argsort(np.array(par_qos[i]))
        ascDeque = collections.deque()
        for j in range(ascending.shape[0]):
            ascDeque.append(ascending[j])

        mcsum = sum(par_comp_bytes[i])
        groupthresh = math.ceil(mcsum / math.ceil(mcsum / MEMORY_THRESH))

        group_mc = MEMORY_THRESH
        for j in range(ascending.shape[0]):
            tmpIdx = ascDeque.pop() if j % 2 == 1 else ascDeque.popleft()
            if group_mc > groupthresh:
                reorder_job_list.append([])
                group_mc = 0
                group_counter += 1

            reorder_job_list[group_counter].append(par_job_list[i][tmpIdx])
            group_mc += par_comp_bytes[i][tmpIdx]

#    for i in range(len(reorder_job_list)):
#        print(reorder_job_list[i])

    return reorder_job_list
