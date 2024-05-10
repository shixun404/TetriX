class Agent:
    def __init__(self, num_cpu, num_gpu):
        self.num_cpu = num_cpu
        self.num_gpu = num_gpu
    
    def get_task(self):
        """Get a task from the workload pool, if available, and move it to the running pool."""
        if not self.workload_pool:
            return None  # No available tasks
        task = self.workload_pool.pop()
        self.running_pool.add(task)
        return task

    def finish_task(self, task):
        """Mark a task as finished and move successors to the workload pool if ready."""
        if task not in self.running_pool:
            raise ValueError("Task not in running pool")
        self.running_pool.remove(task)
        self.finished_pool.add(task)

        # Check successors and move them to the workload pool if all their predecessors are finished
        for successor in self.dag.get(task, [ ]):
            self.predecessor_count[successor] -= 1
            if self.predecessor_count[successor] == 0:
                self.workload_pool.add(successor)
