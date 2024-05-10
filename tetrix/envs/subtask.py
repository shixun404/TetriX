class SubTask():
    def __init__(self, gid = None,
                    job_id=None,
                    num_cpu=None,
                    num_gpu=None,
                    status=None,
                    node=None,
                    max_time=None):
        self.gid = gid
        self.job_id = job_id
        self.num_cpu = num_cpu
        self.num_gpu = num_gpu
        self.status = status
        self.node = node
        self.max_time = max_time
    
