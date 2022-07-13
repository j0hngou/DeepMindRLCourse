from MAB import MultiArmedBandit
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def policy_gradient_solver(mab: MultiArmedBandit, n: int, e: int = 0) -> tuple[float, MultiArmedBandit, torch.Tensor]:
    k = mab.k
    weights = nn.Parameter(torch.ones(k))
    reward = torch.empty(1)
    action = torch.empty(1)
    optimizer = optim.Adam([weights], lr=1e-2)
    total_episodes = n
    total_reward = 0.0
    e = 0
    for i in range(total_episodes):
        if np.random.rand(1) < e:
            action = np.random.randint(k)
        else:
            distribution = weights.softmax(0)
            action = torch.multinomial(distribution, 1).item()
        reward = MAB.pull(action)
        total_reward += reward
        optimizer.zero_grad()
        loss = -torch.log(weights.softmax(0))[action] * reward
        loss.backward()
        optimizer.step()
        MAB.increment(action)
    return total_reward, MAB, weights

if __name__ == '__main__':
    MAB = MultiArmedBandit(k=10)
    total_reward, MAB, weights = policy_gradient_solver(mab=MAB, n=10000)
    print(f"Total reward: {total_reward}")
    print(f"Arms: {MAB.k}\nTimes: {MAB.times},\nMeans: {MAB.means},\nStds: {MAB.stds}")
    print(f"Weights: {weights}")
