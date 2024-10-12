from poe_api_wrapper import PoeApi
import time, threading
import cv2
import os
import moviepy.editor as mp
import numpy as np
import re
from const import *

class LLM():
    def __init__(self):
        tokens = {
            'p-b': "i7JZq6B4feBIrYx2Ao8ewQ==", 
            'p-lat': "cuHNeEA7iTTHDGTzlgzdCa1b6niVntYU6vSQmOqr5g==",
            'formkey': 'c045ce1e4e5ba81ffa20da079984c515',
        }

        tokens_jlc = {
            'p-b': "iG6CkxM-wGYwbqiYIkFDrA==", 
            'p-lat': "moQLC5Bz/NR66fad6mHHR8kQ9Vtecv492PW5a8c6AQ==",
            'formkey': 'c8ceeb7e8b2c67b13aceb888ddbfa20d',
        }
        self.client=PoeApi(tokens=tokens_jlc)
        # self.get_data()

    def get_data(self):
        # Get chat data of all bots (this will fetch all available threads)
        print(self.client.get_chat_history("gpt4_o_128k")['data'])
        # Get chat data of a bot (this will fetch all available threads)
        print(self.client.get_chat_history("gpt4_o_mini")['data'])
        data = self.client.get_settings()
        print(data)
        # print(client.get_botInfo(handle="gpt4_o"))
        # print(client.get_available_creation_models())
        # print(client.get_available_bots())

    def send_message(self,bit_rate,buffer_size,download_speed,delay,next_chunk_size,remain_chunks):
        fixed_msg = "This is an adaptive bitrate streaming optimization problem in video-on-demand mode. Based on the past state information and the target QoE objective, answer the optimal next video chunk bitrate to maximize QoE."
        msg1="The available bitrates are among [300, 750, 1200, 1850, 2850, 4300] Kbps. The past states are as follows: The last selected video chunk bitrates is "+str(bit_rate)+" Kbps. "
        msg2="The current buffer size is "+str(buffer_size)+" seconds."
        msg3="The last "+str(len(download_speed))+" chunks' download speeds (bandwidth) are "+str(download_speed)+" MB/s."
        msg4="The last "+str(len(delay))+" chunks' download time are "+str(delay)+" ms."
        msg5="The next available chunk size are "+str(next_chunk_size)+" MB, each representing different bitrates-encoded chunks."
        msg6="The remaining chunk number is "+str(remain_chunks)+"."
        msg7="The target QoE for each chunk is to maximize: bitrates/1000 - 4.3*rebuffering_time - 1*(bitrates-last_bitrates)/1000. The units for bitrates and rebuffering time are Kbps and seconds, respectively. You need to consider long-term QoE for all remaining chunks that requires future bandwidth prediction and future chunk size prediction."
        msg8="Your answer output format is: {'bitrate': 0-5,'reason':Your analysis}."
        message=fixed_msg+msg1+msg2+msg3+msg4+msg5+msg6+msg7+msg8
        print(message)
        # for chunk in self.client.send_message("gpt4_o_128k", message,chatId= 649712483): pass
        for chunk in self.client.send_message("gpt4_o_mini", message,chatId= 649104377,timeout=30): pass
        res=chunk["text"]
        return res

    def predict(self, state):
        # select bit_rate according to decision tree
        bit_rate=state[0,0,-1]*float(np.max(VIDEO_BIT_RATE))
        buffer_size=state[0,1,-1]*BUFFER_NORM_FACTOR
        download_speed=state[0,2,:] #MB/s
        delay=state[0,3,:]*M_IN_K / BUFFER_NORM_FACTOR #ms
        next_chunk_size=state[0,4, :A_DIM] #MB
        remain_chunks=state[0,5,-1]*float(CHUNK_TIL_VIDEO_END_CAP)

        valid_bandwidth=download_speed[download_speed>0]
        valid_time=delay[delay>0]

        print(f"bit_rate: {bit_rate}, buffer_size: {buffer_size}, download_speed: {valid_bandwidth}, delay: {valid_time}, next_chunk_size: {next_chunk_size}, remain_chunks: {remain_chunks}")
        text=self.send_message(bit_rate,buffer_size,valid_bandwidth,valid_time,next_chunk_size,remain_chunks)
        
        match = re.search(r'"bitrate":\s*(\d+)', text)
        if match:
            bit_rate = int(match.group(1))  # 转换为整数
        else:
            raise ValueError("没有找到 bitrate 的值")
        bit_rate = int(bit_rate)
        print(text,bit_rate)
        res=np.zeros((state.shape[0],A_DIM))
        res[:,bit_rate]=1
        return res