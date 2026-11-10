//
//  SnoreAudioUnit+CoreDataClass.swift
//  SnoreDetector
//
//  Created by 张萌琦 on 2024/4/6.
//
//

import Foundation
import CoreData
import AVFoundation

@objc(SnoreAudioUnit)
public class SnoreAudioUnit: NSManagedObject {
    override public func awakeFromInsert() {
        super.awakeFromInsert()
        self.id = UUID()
        self.timestamp = Date()
        
        // create 10s of audio record
        var audioRecorder = AudioRecorder()
        do {
            let audioURL = try audioRecorder.recordForTenSec()
            self.fileURL = audioURL.absoluteString
            // Use audioURLString
        } catch AudioRecorderError.startRecordingFailed(let underlyingError) {
            print("Failed to start recording: \(underlyingError)")
            // Handle the error. For example, you can show an alert to the user.
            // todo: 使用NotificationCenter来发送错误消息给用户
        } catch {
            print("An unexpected error occurred: \(error)")
            // Handle the error.
            // todo: 使用NotificationCenter来发送错误消息给用户
        }
    }
    
    public func predict() {
        // todo:
        // 在此处调用coreML使用训练好的模型对十个 已经完成了与处理的的tensor进行预测
        // 如果有两个及以上的tensor的预测结果为鼾声阳性，那么整个predict的结果是true，否则的话是false
    }
}

enum AudioRecorderError: Error {
    case startRecordingFailed(Error)
}

private class AudioRecorder: NSObject, AVAudioRecorderDelegate {
    var audioRecorder: AVAudioRecorder?

    func recordForTenSec() throws -> URL {
        let recording_name = "SnoreAudioUnit_\(fileNameDateTimeFormatter.string(from: Date())).m4a"
        let audioFilename = getDocumentsDirectory().appendingPathComponent(recording_name)

        let settings = [
            AVFormatIDKey: Int(kAudioFormatMPEG4AAC),
            AVSampleRateKey: 44100,
            AVNumberOfChannelsKey: 1,
            AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue
        ]

        do {
            audioRecorder = try AVAudioRecorder(url: audioFilename, settings: settings)
            audioRecorder?.delegate = self
            audioRecorder?.record(forDuration: 10.0)
        } catch {
            print("Could not start recording: \(error)")
            throw AudioRecorderError.startRecordingFailed(error)
        }
        return audioFilename
    }
    
    func audioRecorderDidFinishRecording(_ recorder: AVAudioRecorder, successfully flag: Bool) {
        // This method will be called when the recording finishes.
        // You can perform your operations here.
        // todo:
        // 应尝试在此处处理音频文件，将音频文件分割为10份，
        // 每一份都转换为tensor的形式，进行预处理，并保存在对象的属性中
    }
    
    func getDocumentsDirectory() -> URL {
        let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
        return paths[0]
    }
}
