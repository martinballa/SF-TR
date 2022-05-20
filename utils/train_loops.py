import numpy as np
from utils.evaluate import evaluate, target_evaluate
from utils.utils import *

def sample_task(args):
    w = np.zeros(args.phi_dims)
    task_id = np.random.randint(args.phi_dims)
    w[task_id] = 1
    return w

def train_target(args, agent, mem, env, train_logger, eval_env, eval_logger, results_dir):
    epsilon = args.epsilon
    results = []
    steps, rewards, task_rewards, phis, losses, task_vector = 0, 0, 0, [], [], []
    T, done, terminated = 0, True, True

    for T in range(1, args.T_max + 1):
        # Make a logging iteration every 1% of the progress
        if T % (args.T_max // 100) == 0:
            print(f"{int(T / args.T_max * 100)}% Step {T} saving logs")
            train_logger.write((results))
            results = []
        if done:
            if T != 1:
                # detach and log task vector
                if torch.is_tensor(task_vector):
                    task_vector = task_vector.detach().cpu().numpy().round(2)
                else:
                    task_vector = task_vector.round(2)
                results.append((T, steps, rewards, np.mean(losses), task_rewards, task_vector, *np.sum(phis, axis=0)))
                steps, rewards, task_rewards, phis, losses = 0, 0, 0, [], []
            state = env.reset()

        s = preprocess_obs(state, args)
        # non-stationary linear envs - task vectors are given for each episode
        if args.scenario in ["random", "random_pen"]:
            task_vector = env.get_task()
        else:
            task_vector = agent.w
        action = agent.act(s, epsilon=epsilon, w=task_vector)

        next_state, reward, done, phi = env.step(action)  # Step

        mem.append((state, action, next_state, reward, done, phi, task_vector))  # Append transition to memory

        steps += 1
        rewards += reward
        phis.append(phi)

        # Train and test
        if T >= args.learn_start:
            if T % args.replay_frequency == 0:
                loss = agent.learn(mem)
                losses.append(loss)
                epsilon = get_epsilon(T, args.T_max, 0.1)

            if T % args.evaluation_interval == 0:
                # to N evaluation episodes for each task
                agent.eval()
                eval_results, avg_reward = target_evaluate(args, eval_env, T, agent)
                eval_logger.write((eval_results))
                agent.train()

            # Update target network
            if args.soft_update:
                agent.soft_update_target_net(args.tau)
            elif T % args.target_update == 0:
                agent.update_target_net()

            # Checkpoint the network
            if (args.checkpoint_interval != 0) and (T % args.checkpoint_interval == 0):
                agent.save_model(results_dir, name=f"checkpoint_{T}.pth")

        state = next_state
    agent.save_model(results_dir, name=f"final_{T}.pth")

def pretrain(args, agent, mem, env, train_logger, eval_env, eval_logger, results_dir):
    epsilon = args.epsilon
    results = []

    steps, rewards, task_rewards, phis, losses = 0, 0, 0, [], []
    T, done, = 0, True

    for T in range(1, args.T_max + 1):
        # Make a logging iteration every 1% of the progress
        if T % (args.T_max // 100) == 0:
            print(f"{int(T / args.T_max * 100)}% Step {T} saving logs")
            train_logger.write((results))
            results = []
        if done:
            # sample a new task vector
            w = sample_task(args)

            if T != 1:
                results.append(
                    (T, steps, rewards, np.mean(losses), task_rewards, np.argmax(w), *np.sum(phis, axis=0)))
                steps, rewards, task_rewards, phis, losses = 0, 0, 0, [], []
            state = env.reset()


        s = preprocess_obs(state, args)
        if args.n_policies == 1:
            option = 0
        else:
            option = np.argmax(w)
        action = agent.act(s, w, epsilon=epsilon, policy=option)

        next_state, reward, done, phi = env.step(action)

        reward = 0 # pretraining - no reward provided
        task_reward = np.dot(w, phi)

        mem.append((state, action, next_state, reward, done, phi, w))

        steps += 1
        rewards += reward
        task_rewards += task_reward
        phis.append(phi)

        # Train and test
        if T >= args.learn_start:

            if T % args.replay_frequency == 0:
                loss = agent.learn(mem)
                losses.append(loss)
                epsilon = get_epsilon(T, args.T_max, 0.1)

            if T % args.evaluation_interval == 0:
                # to N evaluation episodes for each task
                agent.eval()
                for task in range(args.phi_dims):
                    eval_results, avg_reward = evaluate(args, eval_env, T, agent, task=task)
                    eval_logger.write((eval_results))
                agent.train()

            # Update target network
            if args.soft_update:
                agent.soft_update_target_net(args.tau)
            elif T % args.target_update == 0:
                agent.update_target_net()

            # Checkpoint the network
            if (args.checkpoint_interval != 0) and (T % args.checkpoint_interval == 0):
                agent.save_model(results_dir, name=f"checkpoint_{T}.pth")

        state = next_state
    agent.save_model(results_dir, name=f"final_{T}.pth")