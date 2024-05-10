import pickle
import networkx as nx
import random
from subtask import SubTask
from node_status import NodeStatus
import yaml
import threading
import time

class Producer():
    def __init__(self, listen_port=None,
                send_port=None, 
                gossip_configuration_path=None,
                yaml_file_path='configuration.yaml'):
        self.ip = '127.0.0.7'
        self.listen_port = listen_port
        self.send_port = send_port
        self.subtask_list = redis.Redis(host='127.0.0.1', port=6379, decode_responses=True)
        self.cur_pid = 0
        self.status = 'health'
        with open(yaml_file_path, 'r') as file:
            self.config_dict = yaml.safe_load(file)
            # Create a UDP socket
        self.node_status = NodeStatus(ip=self.ip,
                                        listen_port=self.listen_port,
                                        status='Health',
                                        last_ping_timestamp=-1,
                                        last_pong_timestamp=-1,
                                        last_pfail_timestamp=-1)
        self.node_status_dict = {}
        self.node_status_dict[f"{self.ip}:{self.listen_port}"] = self.node_status
        self.node_status_lock = threading.Lock()
        self.node_key = f"{self.ip}:{self.listen_port}"
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Bind the socket to all available interfaces on port 37389
        server_address = ('', self.listen_port)
        # while(True):
        #     # Wait for a single packet
        #     data, address = sock.recvfrom(4096)

        #     print(f"Received packet from {address}")
        #     print(f"Data: {data}")

    def send_gossip(self, ):
        while True:
            # Simulate updating node status before sending gossip
            with self.node_status_lock:
                for key, value in self.node_status_dict.items():
                    cur_time = time.perf_counter()
                    
                    if cur_time - value.last_ping_timestamp > self.config_dict["PingTimeoutSeconds"] 
                    and value.last_pong_timestamp < value.last_ping_timestamp:
                        self.node_status_dict[key].status = 'PFAIL'
                        self.node_status_dict[key].last_pfail_timestamp = cur_time
                        self.node_status_dict[key].failure_report_list[key] = cur_time

                    for key_failure, value_failure in self.node_status_dict[key].failure_report_list.items():
                        if cur_time - value_failure >= self.config_dict["PossibleFailedReportTimeoutSeconds"]:
                            del self.node_status_dict[key].failure_report_list[key_failure]
                    
                    if self.node_status_dict[key].status == 'PFAIL' and 
                        len(self.node_status_dict[key].failure_report_list[key]) < 
                        (len(self.node_status_dict) + 1) / 2 and 
                        cur_time - self.node_status_dict[key].last_pfail_timestamp > 
                        self.config_dict["PossibleFailedTimeoutSeconds"]:
                        self.node_status_dict[key].status = 'Health'
                    

                        



                keys = self.node_status_dict.keys()
                target_nodes_id = random.choice([i for i in range(len(keys)) if keys[i] != self.node_key])
                print(f"{time.strftime('%X')}: Updated node status and sending gossip")
                for id in target_nodes_id:
                    key = keys[id]
                    self.node_status[key].last_ping_timestamp = time.perf_counter()



            time.sleep(self.config_dict["gossipIntervalSeconds"])
    
    def receive_gossip(self, ):
        while True:
            with self.node_status_lock:
                # Simulate checking and possibly modifying the node status
                if node_status[0] == 'Node1: Updated':
                    print(f"{time.strftime('%X')}: Received gossip, checking node status")
                    node_status[0] = 'Node1: OK'  # Reset status after checking
            
            # Simulate periodic check for incoming gossip
            time.sleep(1)
    
    def start(self,):
        threading.Thread(target=self.send_gossip).start()
        threading.Thread(target=self.receive_gossip).start()
    
    def load_submission(self, submission_path):
        for i in len(self.G_list):
            DAG = self.G_list[i]
            for key, value in DAG.items():
                if value["predecessors_num"] == 0 and value["status"] == "ready":
                    subtask = SubTask(gid=i,
                                    job_id=key,
                                    num_cpu=value["num_cpu"],
                                    num_gpu=value["num_gpu"],
                                    status=value["status"],
                                    max_time=value["max_time"],
                                    )
                    self.subtask_list.set(self.cur_pid, subtask)
                    value["status"] == "submitted"
    
    def scan_subtask_list(self,):
        # Iterate over all keys using scan_iter
        for key in self.subtask_list.scan_iter("*"):  # Adjust the pattern if you're targeting specific keys
            value = r.get(key)
            value = pickle.loads(str.encode(value))
            # Check if the status is 'finished'
            if value.status is "finished":
                # Delete the key if the status is 'finished'
                r.delete(key)
                print(f"Deleted key: {key}")
                self.G_list[value.gid][successors]["status"] = "finished"
                for successors in self.G_list[value.gid][value.job_id]["successor"]:
                    self.G_list[value.gid][successors]["predecessors_num"] -= 1

    def generate_random_dag(self, max_nodes=20, max_edges=50, batch_size):
        """Generates a random Directed Acyclic Graph (DAG) with a given number of nodes and edges."""
        for i in range(batch_size):
            G = nx.DiGraph()
            for i in range(max_nodes):
                G.add_node(i)
            
            while G.number_of_edges() < max_edges:
                a, b = random.sample(range(nodes), 2)
                if not nx.has_path(G, b, a) and a != b:
                    G.add_edge(a, b)
            G_successor_dict = {n: list(adj.keys()) for n, adj in G.adjacency()}
            G_reverse = G.reverse()
            G_predecessor_dict = {n: list(adj.keys()) for n, adj in G_reverse.adjacency()}
            G_dict = {}
            for key, value in G_successor_dict.items():
                subtask_status = {}
                subtask_status["num_cpu"] = random.randint(1, 5)
                subtask_status["num_gpu"] = random.randint(1, 8)
                subtask_status["max_time"] = random.randint(1, 8)
                subtask_status["successor"] = value
                subtask_status["predecessors"] = G_predecessor_dict[key]
                subtask_status["predecessors_num"] = len(G_predecessor_dict[key])
                subtask_status["status"] = "ready"
            self.G_list.append(G_dict)
    

if __name__ == "__main__":
    
    pool = WorkloadPool(dag)
    # Simulating task processing
    while pool.workload_pool or pool.running_pool:
        task = pool.get_task()
        if task:
            print(f"Processing {task}")
            # Simulate task completion
            pool.finish_task(task)