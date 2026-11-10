from model_predicting import record_audio
import time
from tqdm import tqdm
from datetime import datetime
import os
from multiprocessing import Process

# todo: 需要在更新model时修改audio_length的值
def listen_to_audio(working_dir, max_time=-1, audio_length=10):
    # max_time: the maximum time of recording audio, unit is hour, if max_time = -1, then record audio continuously
    max_time_seconds = max_time * 3600
    num_of_audios = 0

    # 计算最大的进程数
    if max_time_seconds == -1:
        max_audio_num = -1
    else:
        max_audio_num = max_time_seconds // audio_length + 1

    # 开始收集数据
    function_start_time = time.time()
    print(f"Start recording audio at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # 每十秒添加一个进程来进行录音，直到达到最大进程数
    while max_audio_num == -1 or max_audio_num > 0:
        num_of_audios += 1
        print(f"\rRecording {num_of_audios}th audio, {int(max_audio_num)} audios ({int(max_audio_num * audio_length)}s) remaining ...", end='')
        record_audio(audio_length, 
                     os.path.join(working_dir, 'original_audios'), 
                     f"{datetime.now().strftime('%Y-%m-%d_%H.%M.%S')}.wav")
        if max_audio_num != -1:
                max_audio_num -= 1
    return num_of_audios


    # 结束收集数据
    function_end_time = time.time()
    print(f"Stop recording audio at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time taken: {function_end_time - function_start_time} seconds")

if __name__ == '__main__':
    working_dir = "/Users/zhangmengqi/Documents/PhD/SnoreDetector/long_time_predict"
    max_time = 0.01
    num_of_recorded_audio = listen_to_audio(working_dir, max_time)
    print("All processes are done!")
    print(f"{num_of_recorded_audio} unit of audio recorded!\ntotal time: {max_time} hours")