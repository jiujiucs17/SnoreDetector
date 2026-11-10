//
//  SnoreRecordingSession+CoreDataClass.swift
//  SnoreDetector
//
//  Created by 张萌琦 on 2024/4/6.
//
//

import Foundation
import CoreData

@objc(SnoreRecordingSession)
public class SnoreRecordingSession: NSManagedObject {
    override public func awakeFromInsert() {
        super.awakeFromInsert()
        self.id = UUID()
        self.timestamp = Date()
        self.name = "Snore Session " + dateTimeFormatter.string(from: self.timestamp!)
    }
}
