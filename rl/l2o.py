import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import pickle as pkl
import time
import argparse
import heatmap
import os

class MatrixGenerator(nn.Module):
    def __init__(self, N=16, device=torch.device('cuda:0')):
        super(MatrixGenerator, self).__init__()
        self.N = N
        self.fc1 = nn.Linear(N, (N ** 2) // 2, device=device)  # Example dimension
        self.fc2 = nn.Linear((N ** 2) // 2, N ** 2, device=device) # Output 256 to reshape into 16x16

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Sigmoid to keep output between 0 and 1
        return x.reshape(-1, self.N, self.N)  # Reshape into a 16x16 matrix


def custom_loss(output, N, fan_out, num_hop):
    # Constraint: Each column should sum to exactly 4
    col_sums = output.sum(dim=-2)
    constraint_loss = ((col_sums - fan_out) ** 2).sum()

    # Maximizing trace of A^4 (negative because we minimize loss)
    # A4 = torch.matrix_power(output, num_hop).clamp(min=0, max=1)
    A4 = torch.matrix_power(output, num_hop)
    trace_A4 = torch.sum(torch.exp(-A4))
    maximization_loss = trace_A4

    return constraint_loss + maximization_loss, constraint_loss, maximization_loss

def train(N=16, fan_out=4, num_hop=4, bs=128,lr=5e-4, seed=123, num_epochs=5010, load_path=None):
    device = torch.device('cuda:0')
    model = MatrixGenerator(N=N, device=device)
    if load_path is not None:
        model = torch.load(load_path, device=device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    test_inputs = torch.randn(bs, N, device=device)
    torch.manual_seed(seed)
    with open(f"test_{N}.pkl", 'rb') as f:
        test_inputs = pkl.load(f).to(device)
    start_time = time.time()
    directory = f"N={N}_fanout={fan_out}_numhop={num_hop}"
    path = os.path.join("figs", directory)

    # Check if the directory exists, if not, create it
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory '{path}' created.")
    else:
        print(f"Directory '{path}' already exists.")
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Generate matrix - input is not significant here, could be random noise or fixed
        input_noise = torch.randn(bs, N, device=device)  # Example input
        output = model(input_noise)

        # Apply threshold to make it binary for evaluating the real performance (post-training)
        # During training, work with continuous values to keep gradients
        loss, c_loss, max_loss = custom_loss(output, N=N, fan_out=fan_out, num_hop=num_hop)

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
        
        if epoch % 5 == 0:
            binarized_output = (output > 0.5).float() 
            test_output = model(test_inputs)
            test_loss, test_c_loss, test_max_loss = custom_loss(test_output, N=N, fan_out=fan_out, num_hop=num_hop)
            test_binarized_output = (test_output > 0.5).float()  # Threshold at 0.5
            wandb.log({ 
                        "c&m loss": loss.mean().item(), 
                        "constraint": c_loss.mean().item(),
                        "maximization": max_loss.mean().item(),
                        "#zero elements":  torch.sum(1 - torch.matrix_power(binarized_output, num_hop).clamp(min=0, max=1), dim=(1, 2)).mean(),
                        "test c&m loss": test_loss.mean().item(), 
                        "test constraint": test_c_loss.mean().item(),
                        "test maximization": test_max_loss.mean().item(),
                        "#zero elements":  torch.sum(1 - torch.matrix_power(test_binarized_output, num_hop).clamp(min=0, max=1), dim=(1, 2)).mean(),
                        "elapsed_time": (time.time() - start_time)})
            fig_path = os.path.join(path, f"N={N}_fanout={fan_out}_numhop={num_hop}_seed={seed}_epoch={epoch}.png")
            heatmap.plot_heatmap(test_binarized_output[0], 
                    torch.matrix_power(test_binarized_output[0], num_hop).clamp(min=0, max=1),
                     N=N, num_hop=num_hop, filename=fig_path)
        if epoch % 1000 == 0 and epoch != 0:
            store_path = f"N={N}_fanout={fan_out}_numhop={num_hop}_seed={seed}.pth"
            # Save only the state dictionary
            torch.save(model.state_dict(), store_path)

    with torch.no_grad():
        model.eval()
        final_output = model(test_inputs)
        binarized_output = (final_output > 0.5).float()  # Threshold at 0.5
        print(binarized_output.mean(dim=0))
        print(torch.matrix_power(binarized_output, num_hop).clamp(min=0, max=1).mean(dim=0))
        print(torch.sum(1 - torch.matrix_power(binarized_output, num_hop).clamp(min=0, max=1), dim=(1, 2)).mean())
        print("Sum of each column: ", binarized_output.sum(dim=-2).mean(dim=0))

if __name__ == "__main__":
    
    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser(description="Process some integers.")

    # 添加参数
    parser.add_argument("--N", type=int, help="Number of nodes", default=16)
    parser.add_argument("--fan_out", type=int, help="Fan out", default=4)
    parser.add_argument("--num_hop", type=int, help="#Hops to broadcast", default=4)
    parser.add_argument("--bs", type=int, help="Batch size", default=128)
    parser.add_argument("--lr", type=float, help="Learning rate", default=4e-5)
    parser.add_argument("--seed", type=int, help="Random seed", default=123)
    parser.add_argument("--load_path", type=str, help="Path to load the model", default=None)
    args = parser.parse_args()
    
    config = {
        'N': args.N,
        'fan_out': args.fan_out,
        'num_hop': args.num_hop,
        'lr': args.lr,
        'seed': args.seed,
        'bs': args.bs,
    }
    print(config)
    # assert 0
    wandb.init(
            project=f'gossip',
            sync_tensorboard=True,
            config=config,
            name=f"N={args.N}_fanout={args.fan_out}_numhop={args.num_hop}_seed={args.seed}.pth",
            # monitor_gym=True,
            save_code=True,
    )

    train(N=args.N, fan_out=args.fan_out, num_hop=args.num_hop, bs=args.bs, 
            lr=args.lr, seed=args.seed, load_path=args.load_path)
   