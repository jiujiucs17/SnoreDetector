//
//  SnoreEvent+CoreDataClass.swift
//  SnoreDetector
//
//  Created by 张萌琦 on 2024/4/6.
//
//

import Foundation
import CoreData

@objc(SnoreEvent)
public class SnoreEvent: NSManagedObject {
    override public func awakeFromInsert() {
        super.awakeFromInsert()
        self.id = UUID()
        self.startTimestamp = Date()
    }
}
