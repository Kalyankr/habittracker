//
//  ContentView.swift
//  Data Collector
//
//  Created by Kalyan reddy Katla on 12/25/25.
//


import SwiftUI

struct ContentView: View {

    @State private var isRunning = false
    @State private var currentLabel = 0

    private let motionManager = WatchMotionManager()

    var body: some View {
        VStack(spacing: 12) {

            Text(isRunning ? "Collecting..." : "Stopped")
                .font(.headline)

            Button("🔴 Hair Touch") {
                currentLabel = 1
                motionManager.currentLabel = 1
            }
            .buttonStyle(.borderedProminent)

            Button("🟢 Normal") {
                currentLabel = 0
                motionManager.currentLabel = 0
            }
            .buttonStyle(.bordered)

            Button(isRunning ? "⏹ Stop" : "▶️ Start") {
                toggleCollection()
            }
            .buttonStyle(.bordered)
        }
        .onAppear {
            _ = WatchConnectivityManager.shared
            motionManager.onSample = { sample in
                WatchConnectivityManager.shared.send(sample: sample)
            }
        }
    }

    private func toggleCollection() {
        isRunning.toggle()
        if isRunning {
            motionManager.start()
        } else {
            motionManager.stop()
        }
    }
}
