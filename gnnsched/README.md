# GNNSched

This folder contains `GNNSched` code.

## Files

- main.py: accept connections from the client, create worker and scheduler processes.

- frontend\_tcp.py: accept client requests and send replies.

- frontend\_schedule.py: manage and schedule tasks according to the specified policy.

- worker.py: dispatch the tasks to worker threads and recycle results.

- worker\_common.py: attach the model to CUDA stream and execute it.

- policy.py: functions for scheduling policies.

- run\_server.sh: specify the model list to load and enable `GNNSched` server.

## Usage

```
./run_server.sh
```
