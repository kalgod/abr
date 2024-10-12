import os
import sys
import numpy as np
import argparse
import load_trace
import ppo2 as network  # 根据需要可以替换为其他ABR算法模块
import fixed_env as env
from const import *
from abr_baseline.bba import BBA
from abr_baseline.llm import LLM

def test_algorithm(args,model):
    all_reward=[]
    np.random.seed(RANDOM_SEED)
    
    assert len(VIDEO_BIT_RATE) == A_DIM

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TEST_TRACES)
    net_env = env.Environment(all_cooked_time=all_cooked_time, all_cooked_bw=all_cooked_bw)

    log_path = LOG_FILE+"log_sim_"+args.alg+"_"+all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'w')

    time_stamp = 0
    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY
    action_vec = np.zeros(A_DIM)
    action_vec[bit_rate] = 1

    s_batch = [np.zeros((S_INFO, S_LEN))]
    r_batch = []
    entropy_record = []
    entropy_ = 0.5
    video_count = 0

    while True:
        # print(all_file_names[net_env.trace_idx])
        delay, sleep_time, buffer_size, rebuf, video_chunk_size, next_video_chunk_sizes, end_of_video, video_chunk_remain = net_env.get_video_chunk(bit_rate)

        time_stamp += delay + sleep_time

        reward = (VIDEO_BIT_RATE[bit_rate] / M_IN_K -
                  REBUF_PENALTY * rebuf -
                  SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K)

        r_batch.append(reward)
        last_bit_rate = bit_rate

        log_file.write(f"{time_stamp / M_IN_K}\t{VIDEO_BIT_RATE[bit_rate]}\t{buffer_size}\t{rebuf}\t{video_chunk_size}\t{delay}\t{entropy_}\t{reward}\n")
        log_file.flush()

        state = np.array(s_batch[-1], copy=True) if s_batch else np.zeros((S_INFO, S_LEN))
        state = np.roll(state, -1, axis=1)

        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))
        state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
        state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K
        state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR
        state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K
        state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

        action_prob = model.predict(np.reshape(state, (1, S_INFO, S_LEN)))
        # noise = np.random.gumbel(size=len(action_prob))
        bit_rate = np.argmax(action_prob)

        s_batch.append(state)
        # print(action_prob,bit_rate)
        entropy_ = -np.sum(action_prob*np.log(action_prob+1e-9))
        entropy_record.append(entropy_)

        if end_of_video:
            mean_reward=np.mean(r_batch)
            # log_file.write(f"Average Reward: {mean_reward}")
            log_file.write("\n")
            log_file.close()
            all_reward.append(mean_reward)

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY

            s_batch.clear()
            r_batch.clear()
            action_vec = np.zeros(A_DIM)
            action_vec[bit_rate] = 1
            s_batch.append(np.zeros((S_INFO, S_LEN)))
            entropy_record = []

            video_count += 1

            if video_count >= len(all_file_names):
                break

            log_path = LOG_FILE+"log_sim_"+args.alg+"_"+all_file_names[net_env.trace_idx]
            log_file = open(log_path, 'w')
    all_reward=np.array(all_reward)
    print(f"Average reward of all traces for Algorithm {args.alg}: {np.mean(all_reward)}")


def main():
    parser = argparse.ArgumentParser(description='Test different ABR algorithms.')
    parser.add_argument('-nn_model', type=str,default="./pretrain/nn_model_ep_155400.pth", help='Path to the neural network model.')
    parser.add_argument('-alg', type=str,default="ppo", help='Choose the ABR algorithm to test.')

    args = parser.parse_args()

    # 根据算法选择进行不同的调用
    if args.alg == "ppo":
        actor = network.Network(state_dim=[S_INFO, S_LEN], action_dim=A_DIM, learning_rate=ACTOR_LR_RATE)
        if args.nn_model is not None:
            actor.load_model(args.nn_model)
            print("Testing model restored.")
        model=actor
    elif args.alg == "bb":
        model=BBA()
    elif args.alg == "llm":
        model=LLM()
    else:
        raise ValueError("Invalid algorithm.")
    
    test_algorithm(args,model)


if __name__ == '__main__':
    main()