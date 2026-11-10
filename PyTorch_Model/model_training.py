import numpy as np
import speechpy
import scipy.io.wavfile as wav
import os
import json
import warnings
import soundfile as sf
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

class GPUMemoryDataset(Dataset):
    def __init__(self, data, targets):
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            # raise warning
            warnings.warn("No GPU or MPS available, using CPU")
        self.data = data.to(device)
        self.labels = targets.to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def process_audio_with_soundfile(file_path, 
                  frame_length=0.03, 
                  frame_stride=0.03, 
                  num_filters=32, 
                  fft_points=512, 
                  num_cepstral=32):
    # 使用 soundfile 库来读取音频文件
    signal, fs = sf.read(file_path)
    if len(signal.shape) > 1:
        signal = np.mean(signal, axis=1)
    
    # 振幅归一化
    # 通过查看源码，发现speechpy.feature.mfcc函数并未进行振幅归一化处理，因此需要在调用该函数之前进行振幅归一化处理
    # 对于空白音频文件，振幅归一化处理会导致除以0的情况，因此需要捕获异常，避免对这类音频文件进行振幅归一化处理
    try:
        signal = signal / np.max(np.abs(signal))
    except RuntimeWarning as e:
        # print(f"waring: {e}")
        # print(f"file_path: {file_path}")
        # print(f"signal: {signal}")
        # print(f"np.max(np.abs(signal)): {np.max(np.abs(signal))}")
        # print(f"signal.shape: {signal.shape}")
        pass

    # 将音频信号分帧，每帧的长度为frame_length秒，步长为frame_stride秒
    # 计算每帧的功率谱，FFT点数设置为fft_points
    # 对于power_spectrum中的每一帧应用Mel滤波器组，滤波器的数量设置为num_filters
    # 计算MFCC，倒谱系数的数量设置为num_cepstral
    # 通过查看源码，发现speechpy.feature.mfcc函数并未进行振幅归一化处理，因此需要在调用该函数之前进行振幅归一化处理
    mfcc = speechpy.feature.mfcc(signal,
                                 sampling_frequency=fs, 
                                 frame_length=frame_length, 
                                 frame_stride=frame_stride,
                                 num_filters=num_filters, 
                                 fft_length=fft_points,
                                 num_cepstral=num_cepstral)
    
    # mfcc = mfcc.astype(float)
    # print(f"mfcc.shape before adding one dimention: {mfcc.shape}")
    mfcc = np.array([mfcc])
    # print(f"mfcc.shape after adding one dimention: {mfcc.shape}")

    return mfcc

def process_audio(file_path, 
                  frame_length=0.03, 
                  frame_stride=0.03, 
                  num_filters=32, 
                  fft_points=512, 
                  num_cepstral=32):
    # 读取音频文件
    # 如果文件是多声道音频文件，则取平均值将其转换成单声道音频文件
    fs, signal = wav.read(file_path)
    if len(signal.shape) > 1:
        signal = np.mean(signal, axis=1)

    # 振幅归一化
    # 通过查看源码，发现speechpy.feature.mfcc函数并未进行振幅归一化处理，因此需要在调用该函数之前进行振幅归一化处理
    # 对于空白音频文件，振幅归一化处理会导致除以0的情况，因此需要捕获异常，避免对这类音频文件进行振幅归一化处理
    try:
        signal = signal / np.max(np.abs(signal))
    except RuntimeWarning as e:
        # print(f"waring: {e}")
        # print(f"file_path: {file_path}")
        # print(f"signal: {signal}")
        # print(f"np.max(np.abs(signal)): {np.max(np.abs(signal))}")
        # print(f"signal.shape: {signal.shape}")
        pass

    # 将音频信号分帧，每帧的长度为frame_length秒，步长为frame_stride秒
    # 计算每帧的功率谱，FFT点数设置为fft_points
    # 对于power_spectrum中的每一帧应用Mel滤波器组，滤波器的数量设置为num_filters
    # 计算MFCC，倒谱系数的数量设置为num_cepstral
    # 通过查看源码，发现speechpy.feature.mfcc函数并未进行振幅归一化处理，因此需要在调用该函数之前进行振幅归一化处理
    mfcc = speechpy.feature.mfcc(signal,
                                 sampling_frequency=fs, 
                                 frame_length=frame_length, 
                                 frame_stride=frame_stride,
                                 num_filters=num_filters, 
                                 fft_length=fft_points,
                                 num_cepstral=num_cepstral)
    
    # set data type of mfcc to float 
    # mfcc = mfcc.astype(float)
    # print(f"mfcc.shape before adding one dimention: {mfcc.shape}")
    mfcc = np.array([mfcc])
    # print(f"mfcc.shape after adding one dimention: {mfcc.shape}")

    # return mfcc
    return mfcc

def process_audio_dir(dir_path):
    # 读取目录下的所有音频文件
    # 对每个音频文件调用process_audio函数，得到MFCC特征
    # 将所有音频文件的MFCC特征合并成一个numpy数组并返回
    mfccs = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if not file.endswith('.wav'):
                continue
            file_path = os.path.join(root, file)
            mfcc = process_audio(file_path)
            mfccs.append(mfcc)
    return np.array(mfccs)

def normalize_audio(model_path, data):
    # parameter file is a json file that contains the mean and std of the training data
    # data is np.array of shape (500, 64, 32)
    # parameter_file = model_path + '/training_data_parameters.json'
    parameter_file = os.path.join(model_path, 'training_data_parameters.json')
    with open(parameter_file, 'r') as f:
        params = json.load(f)
    mean = params['mean']
    std = params['std']
    
    # Normalize the data using the mean and std
    normalized_data = (data - mean) / std
    
    return normalized_data

def generate_training_data(true_data, false_data, train_percentage=0.9, val_percentage=0.1):
    # 按照true_data和false_data各占50%的比例生成训练数据、验证数据和测试数据，返回这三个数据集
    # 这三个数据集的比例分别为8:2:0
    total_samples = len(true_data) + len(false_data)
    train_samples = int(total_samples * train_percentage)
    val_samples = int(total_samples * val_percentage)
    test_samples = total_samples - train_samples - val_samples
    
    # Shuffle the data
    np.random.shuffle(true_data)
    np.random.shuffle(false_data)
    
    # Split the data into training, validation, and test sets
    train_data = np.concatenate((true_data[:train_samples//2], false_data[:train_samples//2]))
    val_data = np.concatenate((true_data[train_samples//2:train_samples//2+val_samples//2], false_data[train_samples//2:train_samples//2+val_samples//2]))
    test_data = np.concatenate((true_data[train_samples//2+val_samples//2:], false_data[train_samples//2+val_samples//2:]))

    train_label = np.concatenate((np.ones(train_samples//2), np.zeros(train_samples//2)))
    val_label = np.concatenate((np.ones(val_samples//2), np.zeros(val_samples//2)))
    test_label = np.concatenate((np.ones(test_samples//2), np.zeros(test_samples//2)))
    
    return train_data, val_data, test_data, train_label, val_label, test_label

def output_data_parameters(model_path, mean, std):
    # model_path = model_path + '/training_data_parameters.json'
    model_path = os.path.join(model_path, 'training_data_parameters.json')
    # 创建保存参数的字典
    params = {
        'mean': mean,
        'std': std
    }
    
    # 将字典保存为json文件
    with open(model_path, 'w') as f:
        json.dump(params, f)

def get_mean_std(data):
    # 计算数据的均值和标准差
    # 返回均值和标准差
    # data.shape = (500, 64,32)
    if np.isnan(data).any():
        print("data contains NaN values")
        raise ValueError("data contains NaN values")
    mean = np.mean(data)
    std = np.std(data)
    # print(f"mean: {mean}, std: {std}")
    return mean, std

class SnoreNet(nn.Module):
    def __init__(self):
        super(SnoreNet, self).__init__()
        # 输入图像大小为32*32，输入通道数为1，输出通道数（即filter数量）为32，卷积核大小为3*3，步长为1，padding为1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.activation = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout_25 = nn.Dropout(0.25)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64*6*6, 512)
        self.dropout_50 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 1)
        self.activation_sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.maxpool(x)
        x = self.dropout_25(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.conv4(x)
        x = self.activation(x)
        x = self.maxpool(x)
        x = self.dropout_25(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout_50(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.activation_sigmoid(x)
        return x

def binary_accuracy(preds, y):
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

def train(model, iterator, optimizer, criterion, device='cpu'):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in iterator:
        inputs, labels = batch.data
        inputs=inputs.float().to(device)
        labels=labels.float().to(device)
        optimizer.zero_grad()
        predictions = model(input).squeeze(1)
        loss = criterion(predictions, labels)
        acc = binary_accuracy(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def validate(model, iterator, criterion, device='cpu'):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.float().to(device)).squeeze(1)
            loss = criterion(predictions, batch.labels)
            acc = binary_accuracy(predictions, batch.labels)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

if __name__ == '__main__':
    # 使用 soundfile 库来读取音频文件
    snore_set_path = '/Users/zhangmengqi/Documents/PhD/SnoreDetector/PyTorch_Model/Snoring_Dataset/1'
    # snore_set_path = r"C:\Users\jiuji\Desktop\SnoreDetector\PyTorch_Model\Snoring_Dataset\1"
    non_snore_set_path = '/Users/zhangmengqi/Documents/PhD/SnoreDetector/PyTorch_Model/Snoring_Dataset/0'
    # non_snore_set_path = r"C:\Users\jiuji\Desktop\SnoreDetector\PyTorch_Model\Snoring_Dataset\0"
    model_path = "/Users/zhangmengqi/Documents/PhD/SnoreDetector/PyTorch_Model"
    # model_path = r"C:\Users\jiuji\Desktop\SnoreDetector\PyTorch_Model"
    batch_size = 4
    epoch = 6000
    validate_every = 20
    use_gpu = True

    # 如果要求使用gpu，检查是否有cuda可用, 如果有cuda可用，将device设置为"cuda"
    # 如果没有cuda可用，检查是否有MPS可用，如果有MPS可用，将device设置为"mps"
    device = "cpu"
    if use_gpu:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
    print(f"device: {device}")

    # 只将UserWarning转化为异常
    warnings.filterwarnings('error', category=RuntimeWarning)

    # 读取音频文件的MFCC特征，并打印
    mfccs_snore = process_audio_dir(snore_set_path)
    print(f"mfccs_snore.shape: {mfccs_snore.shape}")
    mfccs_non_snore = process_audio_dir(non_snore_set_path)

    # 生成训练数据、验证数据和测试数据
    train_mfccs, val_mfccs, test_mfccs, train_labels, val_labels, test_labels = generate_training_data(mfccs_snore, mfccs_non_snore)
    print(f"train_mfccs.shape: {train_mfccs.shape}")

    # 计算训练数据的均值和标准差
    mean, std = get_mean_std(train_mfccs)

    # 输出数据参数到文件
    output_data_parameters(model_path, mean, std)

    # 对数据进行归一化处理
    train_mfccs = normalize_audio(model_path, train_mfccs)
    print(f"train_mfccs.shape: {train_mfccs.shape}")
    val_mfccs = normalize_audio(model_path, val_mfccs)
    test_mfccs = normalize_audio(model_path, test_mfccs)
    # print(f"train_mfccs.shape: {train_mfccs.shape}")
    # print(f"val_mfccs.shape: {val_mfccs.shape}")
    # print(f"test_mfccs.shape: {test_mfccs.shape}")

    # 创建数据集
    if device == 'cuda' or device == 'mps':
        train_dataset = GPUMemoryDataset(torch.tensor(train_mfccs, dtype=torch.float), torch.tensor(train_labels, dtype=torch.float))
        val_dataset = GPUMemoryDataset(torch.tensor(val_mfccs, dtype=torch.float), torch.tensor(val_labels, dtype=torch.float))
        test_dataset = GPUMemoryDataset(torch.tensor(test_mfccs, dtype=torch.float), torch.tensor(test_labels, dtype=torch.float))
    else:
        train_dataset = TensorDataset(torch.tensor(train_mfccs, dtype=torch.float), torch.tensor(train_labels, dtype=torch.float))
        val_dataset = TensorDataset(torch.tensor(val_mfccs, dtype=torch.float), torch.tensor(val_labels, dtype=torch.float))
        test_dataset = TensorDataset(torch.tensor(test_mfccs, dtype=torch.float), torch.tensor(test_labels, dtype=torch.float))
    # print(f"train_data tensor shape: {torch.tensor(train_mfccs).shape}")
    # print(f"train_labels tensor shape: {torch.tensor(train_labels).shape}")
    # print(f"train_dataset[0][0] shape: {train_dataset[0][0].shape}")
    # print(f"train_dataset[0][1] shape: {train_dataset[0][1].shape}, value: {train_dataset[0][1]}")


    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


    
    # 创建模型、损失函数和优化器
    model = SnoreNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=1e-5)

    train_losses_epoch = []
    train_accuracies_epoch = []
    val_losses_epoch = []
    val_accuracies_epoch = []

    pbar = tqdm(total=epoch*train_mfccs.shape[0]//batch_size)
    for epoch in range(epoch):
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        for i, data in enumerate(train_loader, 0):
            pbar.set_description(f"Epoch {epoch+1}, batch {str(i).zfill(2)}/{len(train_loader)}")
            inputs, labels = data[0].to(device), data[1].to(device).unsqueeze(1)

            optimizer.zero_grad()

            # print(f"inputs shape: {inputs.shape}")

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_accuracy = binary_accuracy(outputs, labels).item()

            if device == 'cuda':
                torch.cuda.empty_cache()

            validate_every = min(validate_every, len(train_loader))
            if i%validate_every == validate_every-1:
                train_accuracy = binary_accuracy(outputs, labels)
                train_accuracies.append(train_accuracy.item())

                for i_val, data_val in enumerate(val_loader, 0):
                    inputs_val, labels_val = data_val[0].to(device), data_val[1].to(device).unsqueeze(1)
                    # inputs_val = inputs_val.float().to(device)
                    # labels_val = labels_val.float().to(device)

                    outputs_val = model(inputs_val)
                    loss_val = criterion(outputs_val, labels_val)
                    val_losses.append(loss_val.item())

                    # calculate accuracy
                    acc = binary_accuracy(outputs, labels)
                    val_accuracies.append(acc.item())

                    if device == 'cuda':
                        torch.cuda.empty_cache()

            pbar.update(1)

        train_losses_epoch.append(np.mean(train_losses))
        train_accuracies_epoch.append(np.mean(train_accuracies))
        val_losses_epoch.append(np.mean(val_losses))
        val_accuracies_epoch.append(np.mean(val_accuracies))

        # train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        # train_losses.append(train_loss)
        # train_accuracies.append(train_acc)
        # val_loss, val_acc = validate(model, val_loader, criterion, device)
        # val_losses.append(val_loss)
        # val_accuracies.append(val_acc)

    # use plt to generate 4 sub-plots, the first and second sub-plot show the change of training and validation loss over epochs
    # for each epoch, show the average loss of all batches in this epoch
    # the third and fourth sub-plot show the change of training and validation accuracy over epochs
    # for each epoch, show the average accuracy of all batches in this epoch
    # note that the graph should be line graph
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(train_losses_epoch)
    axs[0, 0].set_title('Training Loss')
    axs[0, 1].plot(val_losses_epoch)
    axs[0, 1].set_title('Validation Loss')
    axs[1, 0].plot(train_accuracies_epoch)
    axs[1, 0].set_title('Training Accuracy')
    axs[1, 1].plot(val_accuracies_epoch)
    axs[1, 1].set_title('Validation Accuracy')
    plt.subplots_adjust(hspace=0.4)

    plt.savefig(os.path.join(model_path, 'training_result.png'))
    output_path = os.path.join(model_path, 'model.pth')
    torch.save(model.state_dict(), output_path)
    
    plt.show()


