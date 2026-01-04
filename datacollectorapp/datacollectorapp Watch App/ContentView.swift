//
//  ContentView.swift
//  Data Collector
//
//  Created by Kalyan reddy Katla on 12/25/25.
//


import SwiftUI

struct ContentView: View {

    @State private var isRunning = false
    @State private var sampleCount = 0
    @State private var annotationCount = 0
    @State private var transferStatus: String = ""
    @State private var isTransferring = false
    @State private var lastLabel: String = ""

    private let motionManager = WatchMotionManager()

    var body: some View {
        ScrollView {
            VStack(spacing: 8) {
                
                // Status display
                VStack(spacing: 2) {
                    Text(isRunning ? "🔴 Recording" : "⏸ Stopped")
                        .font(.headline)
                    
                    HStack(spacing: 8) {
                        Text("\(sampleCount)")
                            .font(.caption2)
                            .monospacedDigit()
                        Text("samples")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                        Text("•")
                            .foregroundColor(.secondary)
                        Text("\(annotationCount)")
                            .font(.caption2)
                            .monospacedDigit()
                        Text("labels")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                }
                
                // Annotation buttons - only enabled while recording
                if isRunning {
                    VStack(spacing: 6) {
                        if !lastLabel.isEmpty {
                            Text("Last: \(lastLabel)")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                        }
                        
                        HStack(spacing: 6) {
                            Button {
                                recordLabel("yes")
                            } label: {
                                Text("✋ Yes")
                                    .font(.caption)
                                    .frame(maxWidth: .infinity)
                            }
                            .buttonStyle(.borderedProminent)
                            .tint(.red)
                            
                            Button {
                                recordLabel("no")
                            } label: {
                                Text("🙅 No")
                                    .font(.caption)
                                    .frame(maxWidth: .infinity)
                            }
                            .buttonStyle(.borderedProminent)
                            .tint(.green)
                        }
                    }
                }

                // Start/Stop button
                Button {
                    toggleCollection()
                } label: {
                    HStack {
                        Image(systemName: isRunning ? "stop.fill" : "play.fill")
                        Text(isRunning ? "Stop" : "Start")
                    }
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .tint(isRunning ? .orange : .blue)
                .disabled(isTransferring)
                
                // Transfer status
                if !transferStatus.isEmpty {
                    Text(transferStatus)
                        .font(.caption2)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                }
                
                if isTransferring {
                    ProgressView()
                        .scaleEffect(0.8)
                }
            }
            .padding(.horizontal, 4)
        }
        .onAppear {
            _ = WatchConnectivityManager.shared
            
            // Update sample count
            motionManager.onSampleCountUpdate = { count in
                sampleCount = count
            }
            
            // Handle transfer completion
            WatchConnectivityManager.shared.onTransferComplete = { success, error in
                DispatchQueue.main.async {
                    isTransferring = false
                    if success {
                        transferStatus = "✅ Sent ZIP!"
                        motionManager.clearData()
                        sampleCount = 0
                        annotationCount = 0
                    } else {
                        transferStatus = "❌ \(error ?? "Failed")"
                    }
                }
            }
        }
    }
    
    private func recordLabel(_ label: String) {
        motionManager.buttonDown()
        // Small delay to simulate press duration
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
            motionManager.recordAnnotation(label: label)
            annotationCount = motionManager.annotationCount
            lastLabel = label
        }
    }

    private func toggleCollection() {
        if isRunning {
            // Stopping - transfer data as ZIP
            motionManager.stop()
            isRunning = false
            
            if motionManager.sampleCount > 0 {
                isTransferring = true
                transferStatus = "Creating ZIP..."
                
                let sensorData = motionManager.getSensorCSVData()
                let annotationsData = motionManager.getAnnotationsCSVData()
                
                WatchConnectivityManager.shared.transferSessionAsZip(
                    sensorData: sensorData,
                    annotationsData: annotationsData,
                    sampleCount: motionManager.sampleCount,
                    annotationCount: motionManager.annotationCount
                )
            } else {
                transferStatus = "No data"
            }
        } else {
            // Starting
            transferStatus = ""
            lastLabel = ""
            annotationCount = 0
            isRunning = true
            motionManager.start()
        }
    }
}
