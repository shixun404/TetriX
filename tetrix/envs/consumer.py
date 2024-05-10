import gym
import numpy as np
import torch as th
import networkx as nx
import random
import redis
class Consumer():
    # Load Task from Workload List
    def __init__(self, cpu_max_capacity=0, 
                        gpu_max_capacity=0,
                        worklist_host_ip='127.0.0.7',
                        worklist_host_port=6379):
        self.cpu_max_capacity = cpu_max_capacity
        self.gpu_max_capacity = gpu_max_capacity
        self.worklist_host_ip = worklist_host_ip
        self.worklist_host_port = worklist_host_port
        self.worklist = redis.Redis(host=self.worklist_host_ip, 
                        port=self.worklist_host_port,
                        decode_responses=True,
                        )
        self.node_status = None

    
    def run_consumer():
        while True:



    def pull_task():
        pass

    def execute_task():
        pass
    
    def finish_task():
        pass
