from agent.agent import Agent
from functions import *

try:

    # stock to observe
    stock_name = "TSLA"
    # number of epochs
    episode_count = 1000

    # the 'trader'
    window_size = 10
    agent = Agent(window_size)

    data = getStockDataVec(stock_name)
    l = len(data) - 1
    batch_size = 32

    for e in range(episode_count + 1):
        print(str(e) + "/" + str(episode_count))
        state = getState(data, 0, window_size + 1)

        total_profit = 0
        agent.inventory = []

        for t in range(l):
            #take an action based on current state
            action = agent.act(state)

            # sit
            next_state = getState(data, t + 1, window_size + 1)
            reward = 0

            # buy
            if action == 1:
                agent.inventory.append(data[t])

            # sell
            # increase the reward
            elif action == 2 and len(agent.inventory) > 0:
                bought_price = agent.inventory.pop(0)
                reward = max(data[t] - bought_price, 0)
                total_profit += data[t] - bought_price

            # continue to the next state
            done = True if t == l - 1 else False
            agent.memory.append((state, action, reward, next_state, done))
            state = next_state

            if len(agent.memory) > batch_size:
                agent.expReplay(batch_size)

        # save every 10th model
        if e % 10 == 0:
            agent.model.save("models/model_ep" + str(e))
except Exception as e:
    print("Error: ".format(e))
finally:
    exit()
