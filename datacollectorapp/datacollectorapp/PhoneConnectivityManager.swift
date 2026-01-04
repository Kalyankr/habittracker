//
//  PhoneConnectivityManager.swift
//  habittracker
//
//  Created by Kalyan reddy Katla on 12/25/25.
//

import WatchConnectivity
import SwiftUI
import Combine

class PhoneConnectivityManager: NSObject, ObservableObject, WCSessionDelegate {
    
    static let shared = PhoneConnectivityManager()
    
    @Published var sessions: [SessionData] = []
    @Published var isConnected: Bool = false
    @Published var lastReceivedMessage: String = ""

    override init() {
        super.init()

        if WCSession.isSupported() {
            let session = WCSession.default
            session.delegate = self
            session.activate()
        }
        
        // Load existing sessions from documents
        loadExistingSessions()
    }
    
    private func loadExistingSessions() {
        let documentsURL = getDocumentsDirectory()
        
        do {
            let files = try FileManager.default.contentsOfDirectory(at: documentsURL, includingPropertiesForKeys: [.creationDateKey])
            
            for fileURL in files where fileURL.pathExtension == "zip" {
                let attrs = try FileManager.default.attributesOfItem(atPath: fileURL.path)
                let creationDate = attrs[.creationDate] as? Date ?? Date()
                
                // Extract timestamp from filename: session_YYYY-MM-DD_HH-mm-ss.zip
                let timestamp = extractTimestamp(from: fileURL.lastPathComponent)
                
                let session = SessionData(
                    sessionTimestamp: timestamp,
                    zipFileURL: fileURL,
                    sampleCount: 0, // Unknown for loaded files
                    annotationCount: 0,
                    receivedAt: creationDate
                )
                
                DispatchQueue.main.async {
                    self.sessions.append(session)
                }
            }
            
            // Sort by date, newest first
            DispatchQueue.main.async {
                self.sessions.sort { $0.receivedAt > $1.receivedAt }
            }
        } catch {
            print("Error loading existing sessions: \(error)")
        }
    }
    
    private func extractTimestamp(from fileName: String) -> String {
        // Extract YYYY-MM-DD_HH-mm-ss from filenames
        let pattern = #"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})"#
        if let regex = try? NSRegularExpression(pattern: pattern),
           let match = regex.firstMatch(in: fileName, range: NSRange(fileName.startIndex..., in: fileName)),
           let range = Range(match.range(at: 1), in: fileName) {
            return String(fileName[range])
        }
        return fileName
    }
    
    private func getDocumentsDirectory() -> URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    // REQUIRED by WCSessionDelegate (iOS)
    func session(_ session: WCSession,
                 activationDidCompleteWith activationState: WCSessionActivationState,
                 error: Error?) {
        DispatchQueue.main.async {
            self.isConnected = activationState == .activated
        }
    }

    func sessionDidBecomeInactive(_ session: WCSession) {
        DispatchQueue.main.async {
            self.isConnected = false
        }
    }

    func sessionDidDeactivate(_ session: WCSession) {
        WCSession.default.activate()
    }
    
    func sessionReachabilityDidChange(_ session: WCSession) {
        DispatchQueue.main.async {
            self.isConnected = session.isReachable
        }
    }

    // FILE RECEIVER - receives ZIP from watch
    func session(_ session: WCSession, didReceive file: WCSessionFile) {
        let metadata = file.metadata ?? [:]
        let fileName = metadata["fileName"] as? String ?? "session_\(Date().timeIntervalSince1970).zip"
        let sessionTimestamp = metadata["sessionTimestamp"] as? String ?? "unknown"
        let sampleCount = metadata["sampleCount"] as? Int ?? 0
        let annotationCount = metadata["annotationCount"] as? Int ?? 0
        
        let documentsURL = getDocumentsDirectory()
        let destinationURL = documentsURL.appendingPathComponent(fileName)
        
        do {
            // Remove existing file if present
            if FileManager.default.fileExists(atPath: destinationURL.path) {
                try FileManager.default.removeItem(at: destinationURL)
            }
            
            // Copy file to documents
            try FileManager.default.copyItem(at: file.fileURL, to: destinationURL)
            print("ZIP saved: \(destinationURL)")
            
            let newSession = SessionData(
                sessionTimestamp: sessionTimestamp,
                zipFileURL: destinationURL,
                sampleCount: sampleCount,
                annotationCount: annotationCount,
                receivedAt: Date()
            )
            
            DispatchQueue.main.async {
                self.sessions.insert(newSession, at: 0)
                self.lastReceivedMessage = "Received: \(fileName)"
            }
            
        } catch {
            print("Failed to save received file: \(error)")
        }
    }
    
    // Delete a session
    func deleteSession(_ session: SessionData) {
        do {
            try FileManager.default.removeItem(at: session.zipFileURL)
            sessions.removeAll { $0.id == session.id }
        } catch {
            print("Failed to delete session: \(error)")
        }
    }
}

