//
//  SnoreAudioUnit+CoreDataProperties.swift
//  SnoreDetector
//
//  Created by 张萌琦 on 2024/4/6.
//
//

import Foundation
import CoreData


extension SnoreAudioUnit {

    @nonobjc public class func fetchRequest() -> NSFetchRequest<SnoreAudioUnit> {
        return NSFetchRequest<SnoreAudioUnit>(entityName: "SnoreAudioUnit")
    }

    @NSManaged public var fileURL: String?
    @NSManaged public var id: UUID?
    @NSManaged public var timestamp: Date?
    @NSManaged public var belongTo: SnoreEvent?
    @NSManaged public var isDoneRecording: Bool?
    @NSManaged public var isDoneProcessing: Bool?

}

extension SnoreAudioUnit : Identifiable {

}
