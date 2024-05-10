from multiprocessing import Process, Queue
import time
import random

def gossip(node_id, queues, state_queues):
    """
    Gossip process function
    :param node_id: Unique identifier for the node
    :param queues: Shared queues for communication (used for normal messages)
    :param state_queues: Shared queues for gossiping state information
    """
    node_states = {i: 'alive' for i in range(len(queues))}  # Initial state of all nodes
    
    while True:
        # Gossip about node states
        gossip_msg = (node_id, node_states)
        target_node = random.choice([i for i in range(len(queues)) if i != node_id])
        state_queues[target_node].put(gossip_msg)
        
        # Handle incoming gossip
        try:
            incoming_id, incoming_states = state_queues[node_id].get_nowait()  # Non-blocking get
            for node, state in incoming_states.items():
                if node != node_id:
                    # Update node state based on received gossip
                    node_states[node] = state
            print(f"Node {node_id} received gossip from Node {incoming_id} and updated states.")
        except:
            pass  # Do nothing if no gossip message
        
        time.sleep(random.randint(1, 4))  # Random sleep to simulate work and gossip interval

if __name__ == "__main__":
    num_nodes = 4  # Number of nodes in the P2P network
    queues = [Queue() for _ in range(num_nodes)]  # Create a queue for each node for normal messages
    state_queues = [Queue() for _ in range(num_nodes)]  # Create a queue for each node for gossip
    
    # Create and start a process for each node for gossip
    gossip_processes = []
    for i in range(num_nodes):
        p = Process(target=gossip, args=(i, queues, state_queues))
        gossip_processes.append(p)
        p.start()
    
    # Let the simulation run for some time
    time.sleep(20)
    
    # Stop all processes
    for p in gossip_processes:
        p.terminate()
        p.join()