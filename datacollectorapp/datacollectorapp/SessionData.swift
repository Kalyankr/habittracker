//
//  SessionData.swift
//  datacollectorapp
//
//  Created by Kalyan reddy Katla on 12/30/25.
//

import Foundation

/// Represents a collected motion data session received from the watch as ZIP
struct SessionData: Identifiable {
    let id = UUID()
    let sessionTimestamp: String
    let zipFileURL: URL
    let sampleCount: Int
    let annotationCount: Int
    let receivedAt: Date
    
    var formattedDate: String {
        let formatter = DateFormatter()
        formatter.dateStyle = .short
        formatter.timeStyle = .medium
        return formatter.string(from: receivedAt)
    }
    
    var fileSizeString: String {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: zipFileURL.path),
              let size = attrs[.size] as? Int64 else {
            return "Unknown"
        }
        
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        return formatter.string(fromByteCount: size)
    }
    
    var fileName: String {
        zipFileURL.lastPathComponent
    }
}
