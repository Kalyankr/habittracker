//
//  WatchConnectivityManager.swift
//  datacollector
//
//  Created by Kalyan reddy Katla on 12/25/25.
//


import WatchConnectivity
import Foundation

class WatchConnectivityManager: NSObject, WCSessionDelegate {

    static let shared = WatchConnectivityManager()
    
    // Callback for transfer completion
    var onTransferComplete: ((Bool, String?) -> Void)?

    override init() {
        super.init()
        if WCSession.isSupported() {
            WCSession.default.delegate = self
            WCSession.default.activate()
        }
    }
    
    /// Create ZIP and transfer to iPhone
    func transferSessionAsZip(sensorData: Data, annotationsData: Data, sampleCount: Int, annotationCount: Int) {
        let tempDir = FileManager.default.temporaryDirectory
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        let timestamp = dateFormatter.string(from: Date())
        
        // Create a session folder
        let sessionFolder = tempDir.appendingPathComponent("session_\(timestamp)")
        
        do {
            // Create folder
            try FileManager.default.createDirectory(at: sessionFolder, withIntermediateDirectories: true)
            
            // Write sensor data CSV
            let sensorFile = sessionFolder.appendingPathComponent("sensor_data.csv")
            try sensorData.write(to: sensorFile)
            
            // Write annotations CSV (tab-separated)
            let annotationsFile = sessionFolder.appendingPathComponent("annotations.csv")
            try annotationsData.write(to: annotationsFile)
            
            // Create ZIP
            let zipURL = tempDir.appendingPathComponent("session_\(timestamp).zip")
            
            // Remove existing zip if present
            if FileManager.default.fileExists(atPath: zipURL.path) {
                try FileManager.default.removeItem(at: zipURL)
            }
            
            // Use NSFileCoordinator to create zip
            let coordinator = NSFileCoordinator()
            var error: NSError?
            
            coordinator.coordinate(readingItemAt: sessionFolder, options: .forUploading, error: &error) { zipSourceURL in
                do {
                    try FileManager.default.copyItem(at: zipSourceURL, to: zipURL)
                } catch {
                    print("Failed to create zip: \(error)")
                }
            }
            
            if let error = error {
                throw error
            }
            
            // Transfer ZIP file
            let metadata: [String: Any] = [
                "fileName": "session_\(timestamp).zip",
                "fileType": "session_zip",
                "sampleCount": sampleCount,
                "annotationCount": annotationCount,
                "sessionTimestamp": timestamp,
                "timestamp": Date().timeIntervalSince1970
            ]
            
            WCSession.default.transferFile(zipURL, metadata: metadata)
            print("Started ZIP transfer: session_\(timestamp).zip")
            
            // Cleanup session folder
            try? FileManager.default.removeItem(at: sessionFolder)
            
        } catch {
            print("Failed to create/transfer ZIP: \(error)")
            onTransferComplete?(false, error.localizedDescription)
        }
    }

    func session(_: WCSession,
                 activationDidCompleteWith _: WCSessionActivationState,
                 error _: Error?) {}
    
    // Called when file transfer completes
    func session(_ session: WCSession, didFinish fileTransfer: WCSessionFileTransfer, error: Error?) {
        DispatchQueue.main.async {
            if let error = error {
                print("File transfer failed: \(error)")
                self.onTransferComplete?(false, error.localizedDescription)
            } else {
                print("ZIP transfer completed successfully")
                self.onTransferComplete?(true, nil)
            }
        }
    }
}
