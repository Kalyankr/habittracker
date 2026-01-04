//
//  ContentView.swift
//  datacollector
//
//  Created by Kalyan reddy Katla on 12/25/25.
//

import SwiftUI

struct ContentView: View {
    @EnvironmentObject var connectivityManager: PhoneConnectivityManager
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Connection status header
                HStack {
                    Circle()
                        .fill(connectivityManager.isConnected ? Color.green : Color.red)
                        .frame(width: 10, height: 10)
                    Text(connectivityManager.isConnected ? "Watch Connected" : "Watch Not Connected")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                    Spacer()
                }
                .padding(.horizontal)
                .padding(.vertical, 8)
                .background(Color(.systemGray6))
                
                if connectivityManager.sessions.isEmpty {
                    // Empty state
                    Spacer()
                    VStack(spacing: 16) {
                        Image(systemName: "applewatch")
                            .font(.system(size: 60))
                            .foregroundColor(.secondary)
                        Text("No Sessions Yet")
                            .font(.title2)
                            .fontWeight(.medium)
                        Text("Start collecting data on your Apple Watch.\nSessions will appear here as ZIP files.")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal, 40)
                    }
                    Spacer()
                } else {
                    // Session list
                    List {
                        ForEach(connectivityManager.sessions) { session in
                            SessionRowView(session: session)
                                .swipeActions(edge: .trailing, allowsFullSwipe: true) {
                                    Button(role: .destructive) {
                                        connectivityManager.deleteSession(session)
                                    } label: {
                                        Label("Delete", systemImage: "trash")
                                    }
                                }
                        }
                    }
                    .listStyle(.insetGrouped)
                }
            }
            .navigationTitle("Motion Data")
        }
    }
}

struct SessionRowView: View {
    let session: SessionData
    
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            // Session info
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Image(systemName: "doc.zipper")
                        .foregroundColor(.blue)
                    Text(session.fileName)
                        .font(.headline)
                        .lineLimit(1)
                }
                
                HStack(spacing: 16) {
                    if session.sampleCount > 0 {
                        Label("\(session.sampleCount) samples", systemImage: "waveform.path")
                    }
                    if session.annotationCount > 0 {
                        Label("\(session.annotationCount) labels", systemImage: "tag")
                    }
                    Label(session.fileSizeString, systemImage: "doc")
                }
                .font(.caption)
                .foregroundColor(.secondary)
                
                Text(session.formattedDate)
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
            
            // Action buttons
            HStack(spacing: 12) {
                ShareLink(item: session.zipFileURL) {
                    Label("Share", systemImage: "square.and.arrow.up")
                        .font(.subheadline)
                }
                .buttonStyle(.bordered)
                
                Button {
                    saveToFiles(session: session)
                } label: {
                    Label("Save to Files", systemImage: "folder")
                        .font(.subheadline)
                }
                .buttonStyle(.bordered)
            }
        }
        .padding(.vertical, 4)
    }
    
    private func saveToFiles(session: SessionData) {
        let picker = UIDocumentPickerViewController(forExporting: [session.zipFileURL], asCopy: true)
        
        if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
           let window = windowScene.windows.first,
           let rootVC = window.rootViewController {
            rootVC.present(picker, animated: true)
        }
    }
}

#Preview {
    ContentView()
        .environmentObject(PhoneConnectivityManager.shared)
}
