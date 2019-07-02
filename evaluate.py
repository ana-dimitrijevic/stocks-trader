from keras.models import load_model

from agent.agent import Agent
from functions import *

try:

    # stock to observe
    stock_name = "TSLA"
    # model to use
    model_name = "model_ep20"

    model = load_model("models/" + model_name)
    window_size = model.layers[0].input.shape.as_list()[1]

    agent = Agent(window_size, True, model_name)
    data = getStockDataVec(stock_name)
    l = len(data) - 1
    batch_size = 32

    state = getState(data, 0, window_size + 1)
    total_profit = 0
    agent.inventory = []
    buy = [0] * len(data)
    sell = [0] * len(data)

    for t in range(l):
        action = agent.act(state)

        # sit
        next_state = getState(data, t + 1, window_size + 1)
        reward = 0

        # buy
        if action == 1:
            agent.inventory.append(data[t])
            buy[t] = data[t]

        # sell
        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            reward = max(data[t] - bought_price, 0)
            total_profit += data[t] - bought_price
            sell[t] = data[t]

        # timeseries_iter += 1
        done = True if t == l - 1 else False
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            print("Evaluation finished!")
            graph(data, buy, sell, model_name, total_profit, stock_name)

        if len(agent.memory) > batch_size:
            agent.expReplay(batch_size)


except Exception as e:
    print("Error: " + e)
finally:
    exit()
