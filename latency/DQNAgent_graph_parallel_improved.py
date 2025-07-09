import torch
import torch as th
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os
from torch.optim.lr_scheduler import StepLR
import copy

def graph_to_edge_index(G):
    edge_list = list(G.edges())
    edge_index = torch.tensor(edge_list).t().contiguous()
    return edge_index

class DQNNetwork(nn.Module):
    def __init__(self, feature_dim, output_dim, T=2, N=20,
                    device=torch.device('cuda:0')):
        super(DQNNetwork, self).__init__()
        self.T = T
        self.N = N
        self.p = feature_dim
        self.device = device
        self.theta_1 = nn.Linear(1, feature_dim)
        self.theta_2 = nn.Linear(feature_dim, feature_dim)
        self.theta_3 = nn.Linear(feature_dim, feature_dim)
        self.theta_4 = nn.Linear(1, feature_dim)

        self.fc_theta_1 = nn.Linear(1, feature_dim)
        self.fc_theta_2 = nn.Linear(feature_dim, feature_dim)
        self.fc_theta_3 = nn.Linear(feature_dim, feature_dim)
        self.fc_theta_4 = nn.Linear(1, feature_dim)

        self.theta_6 = nn.Linear(feature_dim, feature_dim)
        self.theta_7 = nn.Linear(feature_dim, feature_dim)
        self.theta_8 = nn.Linear(feature_dim, feature_dim)
        self.theta_9 = nn.Linear(feature_dim, feature_dim)
        self.fc_theta_7 = nn.Linear(feature_dim, feature_dim)
        self.fc_theta_8 = nn.Linear(feature_dim, feature_dim)
        self.theta_5 = nn.Linear(1 + 4 * feature_dim, feature_dim)
        self.theta_10 = nn.Linear(feature_dim, feature_dim // 2)
        self.theta_11 = nn.Linear(feature_dim // 2, feature_dim // 2)
        self.output = nn.Linear(feature_dim // 2, 1)
        
    def forward(self, state, start_id):
        bs = state.shape[0]
        x_fc = torch.zeros(bs, self.N, self.p, device=self.device)
        m_fc = torch.ones(bs, self.N, 1, device=self.device)
        x = torch.zeros(bs, self.N, self.p, device=self.device)
        adj_fc = state[:, :self.N * self.N].reshape(-1, self.N, self.N)
        adj = state[:, self.N * self.N:self.N * self.N * 2].reshape(-1, self.N, self.N)
        adj_fc_ = adj_fc
        adj_fc = (adj_fc - 5) / 10
        
        adj = (adj - 5) / 10
        reshaped_adj = adj.reshape(-1, 1)
        theta_4_o = torch.relu(self.theta_4(reshaped_adj))
        theta_4_o = theta_4_o.reshape(bs, self.N, self.N, self.p)
        m = state[:, self.N * self.N * 2:].reshape(-1, self.N, 1)
        fc_reshaped_adj = adj_fc.reshape(-1, 1)
            
        fc_theta_4_o = torch.relu(self.fc_theta_4(fc_reshaped_adj))

        theta_1_o = self.theta_1(m)
        theta_3_i = torch.relu(theta_4_o).sum(dim=2)
        theta_3_o = self.theta_3(theta_3_i)
        for i in range(self.T):
            theta_2_o = self.theta_2(torch.bmm(adj, x))
            x = torch.relu(theta_1_o + theta_2_o + theta_3_o)

        summed_x = torch.sum(x, dim=1, keepdim=True)
        summed_x = summed_x.repeat(1, self.N, 1)
        theta_6_o = self.theta_6(summed_x)
        fc_selected_adj = torch.gather(adj_fc_, 1,
                                        start_id.view(-1, 1, 1).expand(
                                            bs, 1, self.N)).reshape(-1, self.N, 1)

        selected_elements = torch.gather(x, 1, start_id.view(-1, 1, 1).expand(bs, 1, self.p))
        id_x = selected_elements.repeat(1, self.N, 1)
        fc_theta_8_o = self.fc_theta_8(x_fc)

        theta_7_o = self.theta_7(id_x)
        theta_8_o = self.theta_8(x)
        
        theta_5_in = torch.relu(torch.cat([fc_selected_adj,
                                            theta_6_o,
                                            fc_theta_8_o,
                                            theta_7_o,
                                            theta_8_o],
                                              dim=2))
        
        theta_10_in = torch.relu(self.theta_5(theta_5_in))
        theta_11_in = torch.relu(self.theta_10(theta_10_in))
        theta_11_out = torch.relu(self.theta_11(theta_11_in))
        output = self.output(theta_11_out).reshape(-1, self.N)
        
        return output

class ParallelDQNAgent:
    """
    Implements Algorithm 1: Parallel DGRO with individual Q-networks per partition
    """
    def __init__(self, M, state_size, action_size, replay_buffer, decay_gamma, device, 
                 experiment_name, sync_freq=1, target_update_freq=1000, shared_buffer=True):
        
        self.state_size = state_size
        self.N = action_size
        self.M = M  # Number of partitions
        self.replay_buffer = replay_buffer
        self.device = device
        self.sync_freq = sync_freq
        self.target_update_freq = target_update_freq
        self.shared_buffer = shared_buffer
        self.decay_gamma = decay_gamma
        self.experiment_name = experiment_name
        self.update_counter = 0
        
        # Create individual Q-networks for each partition
        self.models = []
        self.target_models = []
        self.optimizers = []
        self.schedulers = []
        
        for p in range(M):
            # Main Q-network
            model = DQNNetwork(state_size, action_size, N=action_size, device=device).to(device)
            self.models.append(model)
            
            # Target Q-network
            target_model = DQNNetwork(state_size, action_size, N=action_size, device=device).to(device)
            target_model.load_state_dict(model.state_dict())
            self.target_models.append(target_model)
            
            # Optimizer and scheduler for each network
            optimizer = optim.Adam(model.parameters())
            self.optimizers.append(optimizer)
            
            scheduler = StepLR(optimizer, step_size=2000, gamma=0.95)
            self.schedulers.append(scheduler)
        
        self.criterion = nn.MSELoss()
        self.min_lr = 1e-5
        
        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name)
    
    def generate_masked_one_hot(self, N, M):
        """Generate partition masks and starting nodes"""
        assert N % M == 0, "N must be divisible by M"
        partition_size = N // M

        indices = list(range(N))
        random.shuffle(indices)

        masks = []
        start_id = []
        for i in range(M):
            part_indices = indices[i * partition_size: (i + 1) * partition_size]
            mask = torch.zeros(N)
            mask[part_indices] = 1
            masks.append(mask)
            start_id.append(indices[i * partition_size])
        
        return masks, start_id

    def act(self, state, degree, G, mask, masks, start_id, vector=None, K=4, epsilon=0.95):
        """
        Each partition agent chooses action using its own Q-network
        """
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        parallel_action = []
        
        for i in range(self.M):
            mask_tensor = torch.tensor(mask, dtype=torch.bool, device=self.device)
            partition_mask = masks[i].to(dtype=torch.bool, device=self.device)
            final_mask = mask_tensor & partition_mask
            
            if random.random() > epsilon:
                # Use partition-specific Q-network
                with torch.no_grad():
                    action_values = self.models[i](state_tensor, 
                                                 torch.as_tensor([start_id[i]], device=self.device))
                    action_values = torch.where(final_mask, action_values, 
                                              torch.tensor(float('-inf'), device=self.device))
                    action = torch.argmax(action_values).item()
            else:
                valid_indices = torch.nonzero(final_mask, as_tuple=True)[0]
                action = valid_indices[torch.randint(len(valid_indices), (1,))].item()
            
            parallel_action.append(action)
        
        return parallel_action

    def store_experience(self, states, actions, rewards, next_states, dones, masks, steps):
        """
        Store experience for each partition in shared or individual buffers
        """
        if self.shared_buffer:
            # Store all partition experiences in shared buffer
            for p in range(self.M):
                self.replay_buffer.push(
                    states[p] if isinstance(states, list) else states,
                    actions[p],
                    rewards[p],
                    next_states[p] if isinstance(next_states, list) else next_states,
                    dones,
                    masks[p] if isinstance(masks, list) else masks,
                    steps
                )
        else:
            # Could implement individual buffers per partition here
            pass

    def learn(self, batch_size):
        """
        Update all partition Q-networks using experiences from shared buffer
        """
        if len(self.replay_buffer) < batch_size:
            return []
        
        losses = []
        
        # Update each partition's Q-network
        for p in range(self.M):
            loss = self._update_partition_network(p, batch_size)
            losses.append(loss)
        
        self.update_counter += 1
        
        # Periodic target network updates
        if self.update_counter % self.target_update_freq == 0:
            self._update_target_networks()
        
        return losses

    def _update_partition_network(self, partition_idx, batch_size):
        """Update Q-network for specific partition"""
        samples = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones, masks, steps = zip(*samples)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.uint8, device=self.device)
        masks = torch.tensor(np.array(masks), dtype=torch.bool, device=self.device)
        steps = torch.tensor(steps, dtype=torch.float, device=self.device)
        
        # Extract batch size and create dummy start_ids
        batch_size_actual = states.shape[0]
        dummy_start_ids = torch.zeros(batch_size_actual, dtype=torch.long, device=self.device)
        
        # Current Q-values
        current_q = self.models[partition_idx](states, dummy_start_ids).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q-values using target network
        with torch.no_grad():
            next_q_values = self.target_models[partition_idx](next_states, dummy_start_ids)
            next_q_values = torch.where(masks, next_q_values, torch.tensor(float('-inf')))
            next_q = next_q_values.max(1)[0]
        
        expected_q = rewards + (self.decay_gamma ** steps) * next_q * (1 - dones)
        
        loss = self.criterion(current_q, expected_q)

        self.optimizers[partition_idx].zero_grad()
        loss.backward()
        self.optimizers[partition_idx].step()
        self.schedulers[partition_idx].step()
        
        # Maintain minimum learning rate
        for param_group in self.optimizers[partition_idx].param_groups:
            if param_group['lr'] < self.min_lr:
                param_group['lr'] = self.min_lr
                
        return loss.item()

    def _update_target_networks(self):
        """Copy main networks to target networks"""
        for p in range(self.M):
            self.target_models[p].load_state_dict(self.models[p].state_dict())

    def save(self, episode=None):
        """Save all partition Q-networks"""
        for p in range(self.M):
            if episode is not None:
                save_path = os.path.join(self.experiment_name, f'model_partition_{p}_episode_{episode}.pth')
            else:
                save_path = os.path.join(self.experiment_name, f'model_partition_{p}.pth')
            torch.save(self.models[p].state_dict(), save_path)

    def load(self, load_paths=None, episode=None):
        """Load partition Q-networks"""
        if load_paths is None:
            # Auto-discover model files
            load_paths = []
            for p in range(self.M):
                if episode is not None:
                    filename = f'model_partition_{p}_episode_{episode}.pth'
                else:
                    filename = f'model_partition_{p}.pth'
                load_path = os.path.join(self.experiment_name, filename)
                load_paths.append(load_path)
        
        for p, load_path in enumerate(load_paths):
            if os.path.exists(load_path):
                self.models[p].load_state_dict(torch.load(load_path, map_location=self.device))
                self.target_models[p].load_state_dict(self.models[p].state_dict())
            else:
                print(f"Warning: Model file {load_path} not found for partition {p}")

    def sync_models(self):
        """
        Synchronize all partition models (optional, for experiments with shared weights)
        """
        # Average all model parameters
        avg_state_dict = {}
        for key in self.models[0].state_dict().keys():
            avg_state_dict[key] = torch.stack([model.state_dict()[key] for model in self.models]).mean(0)
        
        # Update all models with averaged parameters
        for model in self.models:
            model.load_state_dict(avg_state_dict)

# Backward compatibility: alias to original DQNAgent
DQNAgent = ParallelDQNAgent 