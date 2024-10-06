# TetriX: A Decentralized and Resilient Multi-Agent Scheduler for HPC Clusters
TetriX is an advanced task scheduler designed for high-performance computing (HPC) clusters. Utilizing a decentralized and resilient approach, TetriX employs a network of agents for resource management, aiming for near-optimal outcomes and surpassing traditional human heuristics.

## Features (Under Development)

- **Efficient Scheduling:** Advanced algorithms using multi-agent reinforcement learning for near-optimal task allocation.
- **Decentralized Architecture:** Even distribution of tasks for enhanced robustness.
- **Resilience and Reliability:** Designed to maintain functionality under various scenarios.
- **Dynamic Adaptability:** Real-time adjustments to workload changes.
- **Scalability:** Effective for both small and large HPC clusters.

## File Structures
```
tetrix/
│
├── envs/                # Gym environments specific to TetriX
│   ├── __init__.py      # Makes envs a Python module
│   └── ...              # Environment implementation files
│
├── agents/              # Agents interacting in the environments
│   ├── __init__.py      # Makes agents a Python module
│   ├── dqn_agent.py     # Example: DQN agent implementation
│   └── ...              # Other agent types
│
├── train/               # Training scripts for the agents
│   ├── train_dqn.py     # Script to train DQN agent
│   └── ...              # Other training scripts
│
├── models/              # PyTorch models (Neural Networks)
│   ├── __init__.py      # Makes models a Python module
│   ├── dqn_model.py     # Example: DQN network architecture
│   └── ...              # Other model architectures
│
├── utils/               # Utility code
│   ├── __init__.py      # Makes utils a Python module
│   └── ...              # Helper functions and classes
│
├── requirements.txt     # Python dependencies, including PyTorch
├── README.md            # Project overview and setup instructions
└── LICENSE              # License information
```

## Current Progress
### Todo List
1. Test K-Ring, Q-learning, Chord, Nearest-Neighbor, Perigee on the Bitnodes network Dataset.
    - 2020. Bitnodes network. Data drawn from website, https://bitnodes.earn.com/.
    - Topology
    - Ring Neigbor Selection
2. Q-learning:
    - Weakness:
        - Centralized: Require global knowledge. However, I think this is not an issue. A strong-consistent membership protocol requires synchronization intrinsically.
        - Scalability: not scalable for 500 nodes.
    - Strongness:
        - 10%-20% lower diameter compared to greedy selection when latency are closed to each other.
        - Self Adaptive: The model can be distributed to all nodes. Once the number of inactive nodes exceeds a threshold, each node can inference the same topology individually.