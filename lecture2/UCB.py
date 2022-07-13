from MAB import MultiArmedBandit
import numpy as np

def UCBSolver(mab: MultiArmedBandit, k: int, n: int) -> list:
    # Pull once for each arm
    mean_estimates = np.zeros(mab.k)
    past_rewards = [np.zeros(n) for _ in range(mab.k)]

    for i in range(mab.k):
        mean_estimates[i] = mab.pull(i)
        past_rewards[i][0] = mean_estimates[i]
        past_rewards[i][1:] = np.nan
        mab.increment(arm=i)

    # On timesteps k+1 to k+n pick arm I_t = argmax(UCB_{i, t-1})
    for t in range(1, n):
        UCBs = np.zeros(mab.k)
        for i in range(mab.k):
            UCBs[i] = mean_estimates[i] + np.sqrt(2 * np.log(t) / mab.times[i])
        arm = np.argmax(UCBs)
        past_rewards[arm][t] = mab.pull(arm)
        mab.increment(arm)
        mean_estimates[arm] = np.nanmean(past_rewards[arm][:t+1]) # Use only the values that have been seen
    return mean_estimates

if __name__ == '__main__':
    MAB = MultiArmedBandit(k=10)
    mean_estimates = UCBSolver(mab=MAB, k=10, n=100)
    print(mean_estimates)
    print(f"Arms: {MAB.k}\nTimes: {MAB.times},\nMeans: {MAB.means},\nStds: {MAB.stds}")