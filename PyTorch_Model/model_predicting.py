from model_training import process_audio, normalize_audio, SnoreNet
import librosa
import soundfile as sf
import os
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import torch
import time
from datetime import datetime
from multiprocessing import Process
from tqdm import tqdm

# todo: 需要在更新model时修改为model的名称
def predict_single_wav_file(file_path, model_path, show_log=True, show_true=True, show_false=True):
    # the file_path is the path of the audio file
    # the audio file should be in .wav format and have a sample rate of 44100 Hz
    # the length of the audio file should be 1s
    # 读取音频文件并提取MFCC特征
    mfcc = process_audio(file_path)
    # 对MFCC特征进行归一化处理
    normalized_mfcc = normalize_audio(model_path, mfcc)
    # 将MFCC特征转换成tensor
    tensor_mfcc = torch.tensor(normalized_mfcc, dtype=torch.float)
    # batch size = 1
    tensor_mfcc = tensor_mfcc.unsqueeze(0)
    # 使用模型进行预测
    model = SnoreNet()
    trained_model_path = os.path.join(model_path, 'model.pth')
    model.load_state_dict(torch.load(trained_model_path))
    model.eval()
    # print(f"tensor_mfcc shape: {tensor_mfcc.shape}")
    # print(f"tensor_mfcc shape: {tensor_mfcc.shape}\n\n\n")
    output = model(tensor_mfcc)
    if show_log:
        if show_true and output >= 0.5:
            print(f"prediction of {file_path.split('/')[-1]}: This is snoring sound, output: {round(output.item()*100, 2)}%")
        if show_false and output < 0.5:
            print(f"prediction of {file_path.split('/')[-1]}: This is non-snoring sound, output: {round(output.item()*100, 2)}%")
    return output

def transform_m4a_to_wav(file_path):
    # 读取音频文件
    y, sr = librosa.load(file_path)
    # 如果采样率不是44100，则将其转换成44100，方法是使用librosa库的resample函数
    if sr != 44100:
        y = librosa.resample(y, orig_sr = sr, target_sr= 44100)
        sr = 44100
    # 将音频文件写入.wav文件
    wav_file_path = file_path.replace('.m4a', '.wav')
    sf.write(wav_file_path, y, sr)

    return wav_file_path

# todo: to参数的默认值需要在更新model时修改为model的默认输入长度（秒）
def crop_wav_file(wav_file_path, output_dir="", to=1):
    if output_dir == "":
        output_dir = os.path.dirname(wav_file_path)
    y, sr = librosa.load(wav_file_path)
    total_length = len(y)
    if total_length > sr * to:
        # 如果音频文件长度大于1s，则将其分割成多个1s的音频文件，如果最后一个音频文件长度小于1s，则使用空白音频填充
        num = total_length // sr
        output_file_paths = []
        for i in range(num):
            # output_file_path = output_dir + 原文件名 + _ + i + .wav
            output_file_path = os.path.join(output_dir, f"{os.path.basename(wav_file_path).replace('.wav', f'_{i}.wav')}")
            sf.write(output_file_path, y[i*sr:(i+1)*sr], sr)
            output_file_paths.append(output_file_path)
        if total_length % sr != 0:
            padding_length = sr * to - total_length % sr
            padding = np.zeros(padding_length)
            # output_file_path = output_dir + 原文件名 + _ + num + .wav
            output_file_path = os.path.join(output_dir, f"{os.path.basename(wav_file_path).replace('.wav', f'_{num}.wav')}")
            sf.write(output_file_path, np.concatenate((y[num*sr:], padding)), sr)
            output_file_paths.append(output_file_path)
        return output_file_paths
    
    elif total_length < sr * to:
        # 如果音频文件小于1s，则使用空白音频填充其不足1s的部分，使其长度为1s
        padding_length = sr * to - total_length
        padding = np.zeros(padding_length)
        padded_y = np.concatenate((y, padding))
        # output_file_path = output_dir + 原文件名 + 0 + .wav
        output_file_path = os.path.join(output_dir, f"{os.path.basename(wav_file_path).replace('.wav', '0.wav')}")
        sf.write(output_file_path, padded_y, sr)
        return [output_file_path]
    else:
        # 如果音频文件长度等于1s，则直接将其写入输出文件夹
        # output_file_path = output_dir + 原文件名
        output_file_path = os.path.join(output_dir, os.path.basename(wav_file_path))
        sf.write(output_file_path, y, sr)
        return [output_file_path]

# todo: output_dir的默认值需要在更新model时修改为model的默认输入长度（秒），在model的输入音频长度变为10秒后，此处可以考虑将seconds修改为60
def record_audio(seconds = 10, output_dir = '/Users/zhangmengqi/Documents/PhD/SnoreDetector/my_own_data', output_name = 'output.wav'):
    fs = 44100  # Sample rate

    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    output_path = os.path.join(output_dir, output_name)
    write(output_path, fs, myrecording)  # Save as WAV file
    return output_path

def predict_uncropped_wav_file(file_path, model_path, output_dir, threshold=0.15):
    cropped_files = crop_wav_file(file_path, output_dir = output_dir)
    results = []
    for file in cropped_files:
        prediction = predict_single_wav_file(file, model_path, show_log=False)
        # 根据predict的结果来修改文件名，如果是打鼾声音，则在文件名开头加上1_，否则加上0_
        new_file_name = f"1_{os.path.basename(file)}" if prediction > 0.5 else f"0_{os.path.basename(file)}"
        new_file_name = os.path.join(output_dir, new_file_name)
        os.rename(file, new_file_name)
        results.append(prediction)
    # 根据预测结果的平均值修改原文件的文件名，如果平均值大于0.5，则在文件名开头加上1_，否则加上0_
    # 最后返回所有预测结果decision的平均值
    file_result = torch.mean(torch.tensor(results))
    new_file_name = f"1_{os.path.basename(file_path)}" if file_result > threshold else f"0_{os.path.basename(file_path)}"
    new_file_name = os.path.join(os.path.dirname(file_path), new_file_name)
    os.rename(file_path, new_file_name)
    print(f"prediction of {file_path.split('/')[-1]}: This is snoring sound, output: {round(file_result.item()*100, 2)}%" if file_result > threshold else f"prediction of {file_path.split('/')[-1]}: This is non-snoring sound, output: {round(file_result.item()*100, 2)}%")
    return file_result.item()

def listen_to_audio_and_predict(working_dir, model_path, max_time=-1):
    # max_time: the maximum time of recording audio, unit is hour, if max_time = -1, then record audio continuously
    max_time_seconds = max_time * 3600
    processes = []
    time_start = time.time()

    if max_time_seconds == -1:
        pbar = tqdm(total=None)
    else:
        pbar = tqdm(total=max_time_seconds)
    time_elapsed = round(time.time() - time_start, 2)
    time_remaining = "∞ " if max_time == -1 else round(max_time_seconds - (time.time() - time_start), 2)
    pbar.set_description(f"time elapsed: {time_elapsed}s, time remaining: {time_remaining}s, {len(processes)} unit of audio recorded")

    # todo: 此处可以考虑将录音和预测都改为多进程
    while max_time_seconds == -1 or max_time_seconds > 0:
        start_time = time.time()
        recorded_audio_file = record_audio(output_dir = os.path.join(working_dir, 'original_audios'), 
                     output_name=f"{datetime.now().strftime('%Y-%m-%d_%H.%M.%S')}.wav")
        
        # predict here
        output_dir = os.path.join(working_dir, 'cropped_audios')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        p = Process(target=predict_uncropped_wav_file, args=(recorded_audio_file, model_path, output_dir))
        p.start()
        processes.append(p)

        time_elapsed = round(time.time() - time_start, 2)
        time_remaining = "∞ " if max_time == -1 else round(max_time*3600 - (time.time() - time_start), 2)
        pbar.set_description(f"time elapsed: {time_elapsed}s, time remaining: {time_remaining}s, {len(processes)} unit of audio recorded")
        pbar.update(round(time.time() - start_time, 0))

        max_time_seconds -= time.time() - start_time
    return processes

if __name__ == '__main__':
    # 持续实施录音和预测
    working_dir = "/Users/zhangmengqi/Documents/PhD/SnoreDetector/long_time_predict"
    model_path = "/Users/zhangmengqi/Documents/PhD/SnoreDetector/PyTorch_Model"
    max_time = 0.1
    processes = listen_to_audio_and_predict(working_dir, model_path, max_time)
    for p in processes:
        p.join()
    print("All predictions are completed!")

    # 预测文件夹中的所有音频文件
    # dir_path = "/Users/zhangmengqi/Documents/PhD/SnoreDetector/long_time_predict/original_audios"
    # for file in sorted(os.listdir(dir_path)):
    #     if file.endswith('.wav'):
    #         predict_single_wav_file(os.path.join(dir_path, file), "/Users/zhangmengqi/Documents/PhD/SnoreDetector/PyTorch_Model", show_true=False)

    # # 预测单个音频文件
    # predict_uncropped_wav_file("/Users/zhangmengqi/Documents/PhD/SnoreDetector/long_time_predict/original_audios/2024-04-02_21.13.49.wav", 
    #                         "/Users/zhangmengqi/Documents/PhD/SnoreDetector/PyTorch_Model",
    #                         "/Users/zhangmengqi/Documents/PhD/SnoreDetector/long_time_predict/cropped_audios")



