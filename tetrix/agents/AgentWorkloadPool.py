import gym
import numpy as np
import torch as th
import networkx as nx
import random

class WorkloadPool:
    def __init__(self, dag):
        self.dag = dag  # DAG representation: {task: [successors]}
        self.predecessor_count = self._calculate_predecessors()
        self.workload_pool = set([task for task, preds in self.predecessor_count.items() if preds == 0])
        self.running_pool = set()
        self.finished_pool = set()

    def _calculate_predecessors(self):
        """Calculate the number of predecessors for each task."""
        predecessor_count = {task: 0 for task in self.dag}
        for successors in self.dag.values():
            for successor in successors:
                if successor in predecessor_count:
                    predecessor_count[successor] += 1
                else:
                    predecessor_count[successor] = 1
        return  
    


if __name__ == "__main__":
    
    pool = WorkloadPool(dag)
    # Simulating task processing
    while pool.workload_pool or pool.running_pool:
        task = pool.get_task()
        if task:
            print(f"Processing {task}")
            # Simulate task completion
            pool.finish_task(task)