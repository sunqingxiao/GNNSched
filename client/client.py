import sys
import time
import struct
import statistics
import argparse

from util.util import TcpClient, timestamp

def send_request(client, task_name, data, layers):
    timestamp('client', 'before_request_%s' % task_name)

    # Serialize task name
    task_name_b = task_name.encode()
    task_name_length = len(task_name_b)
    task_name_length_b = struct.pack('I', task_name_length)

    # Serialize data
    data_b = data.encode()
    data_length = len(data_b)
    data_length_b = struct.pack('I', data_length)

    # Serialize number of layers
    layers_b = layers.encode()
    layers_length = len(layers_b)
    layers_length_b = struct.pack('I', layers_length)

    timestamp('client', 'after_serialization')

    # Send task name / data
    client.send(task_name_length_b)
    client.send(task_name_b)
    client.send(data_length_b)
    client.send(data_b)
    client.send(layers_length_b)
    client.send(layers_b)

    timestamp('client', 'after_request_%s' % task_name)


def recv_response(client):
    reply_b = client.recv(4)
    reply = reply_b.decode()
    timestamp('client', 'after_reply')


def close_connection(client):
    model_name_length = 0
    model_name_length_b = struct.pack('I', model_name_length)
    client.send(model_name_length_b)
    timestamp('client', 'close_connection')


def main():
    # Load model list (task & data)
    model_list_file_name = sys.argv[1]
    model_list = []
    with open(model_list_file_name) as f:
        for line in f.readlines():
            seg = line.split()
            if len(line.split()) != 5:
                continue
            model_list.append([seg[0], seg[1], seg[2]])
    print(model_list)

    # Send inference request
    client_infer = []
    for i in range(len(model_list)):
        client_infer.append(TcpClient('localhost', 12345))
        send_request(client_infer[i], model_list[i][0], model_list[i][1], model_list[i][2])

#    time.sleep(4)

    timestamp('client', 'after_connect')
    time_1 = time.time()

    # Recv inference reply
    for i in range(len(model_list)):
        recv_response(client_infer[i])

    time_2 = time.time()
    duration = (time_2 - time_1) * 1000
    
    # Close connection
    for i in range(len(model_list)):
        close_connection(client_infer[i])

    time.sleep(1)
    timestamp('**********', '**********')


if __name__ == '__main__':
    main()
