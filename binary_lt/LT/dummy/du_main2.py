import os
import time
import subprocess

import torch


def main(gpu_idx):
    os.system('CUDA_VISIBLE_DEVICES=%d python3 du_main.py &' % gpu_idx)


def parse_data(raw_data):
    raw_data = raw_data.split('\n')
    memory_usage_list = []

    for line in raw_data:
        if 'MiB / ' in line:
            memory_usage = int(line.split('|')[2].split('/')[0].strip().replace('MiB', ''))
            memory_usage_list.append(memory_usage)

    return memory_usage_list


MIN_MEM = 100
SLEEP_TIME = 600
MAX_COUNT = 3

gpu_count = torch.cuda.device_count()
print(gpu_count)
command = ['nvidia-smi']
empty_count_list = [MAX_COUNT - 1 for _ in range(gpu_count)]

while True:
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True)
    proc.wait()
    stdout = proc.stdout.read()
    memory_usage_list = parse_data(stdout)

    assert len(memory_usage_list) == gpu_count
    print(memory_usage_list)

    for idx in range(gpu_count):
        if memory_usage_list[idx] <= MIN_MEM:
            empty_count_list[idx] += 1
        else:
            empty_count_list[idx] = 0

    print(empty_count_list)

    for idx in range(gpu_count):
        if empty_count_list[idx] >= MAX_COUNT:
            main(idx)
            print("process run in %d GPU" % idx)

    time.sleep(SLEEP_TIME)
