//
//  SnoreEvent+CoreDataProperties.swift
//  SnoreDetector
//
//  Created by 张萌琦 on 2024/4/6.
//
//

import Foundation
import CoreData


extension SnoreEvent {

    @nonobjc public class func fetchRequest() -> NSFetchRequest<SnoreEvent> {
        return NSFetchRequest<SnoreEvent>(entityName: "SnoreEvent")
    }

    @NSManaged public var endTimestamp: Date?
    @NSManaged public var id: UUID?
    @NSManaged public var startTimestamp: Date?
    @NSManaged public var belongTo: SnoreRecordingSession?
    @NSManaged public var contain: NSSet?

}

// MARK: Generated accessors for contain
extension SnoreEvent {

    @objc(addContainObject:)
    @NSManaged public func addToContain(_ value: SnoreAudioUnit)

    @objc(removeContainObject:)
    @NSManaged public func removeFromContain(_ value: SnoreAudioUnit)

    @objc(addContain:)
    @NSManaged public func addToContain(_ values: NSSet)

    @objc(removeContain:)
    @NSManaged public func removeFromContain(_ values: NSSet)

}

extension SnoreEvent : Identifiable {

}
