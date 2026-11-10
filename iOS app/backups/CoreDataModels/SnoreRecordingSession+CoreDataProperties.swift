//
//  SnoreRecordingSession+CoreDataProperties.swift
//  SnoreDetector
//
//  Created by 张萌琦 on 2024/4/6.
//
//

import Foundation
import CoreData


extension SnoreRecordingSession {

    @nonobjc public class func fetchRequest() -> NSFetchRequest<SnoreRecordingSession> {
        return NSFetchRequest<SnoreRecordingSession>(entityName: "SnoreRecordingSession")
    }

    @NSManaged public var id: UUID?
    @NSManaged public var lengthInSeconds: Double
    @NSManaged public var name: String?
    @NSManaged public var timestamp: Date?
    @NSManaged public var contain: NSSet?

}

// MARK: Generated accessors for contain
extension SnoreRecordingSession {

    @objc(addContainObject:)
    @NSManaged public func addToContain(_ value: SnoreEvent)

    @objc(removeContainObject:)
    @NSManaged public func removeFromContain(_ value: SnoreEvent)

    @objc(addContain:)
    @NSManaged public func addToContain(_ values: NSSet)

    @objc(removeContain:)
    @NSManaged public func removeFromContain(_ values: NSSet)

}

extension SnoreRecordingSession : Identifiable {

}
