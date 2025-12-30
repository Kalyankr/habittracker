//
//  WatchSensorSample.swift
//  datacollector
//
//  Created by Kalyan reddy Katla on 12/25/25.
//


import Foundation

struct WatchSensorSample: Codable {
    let timestamp: Double
    let accX: Double
    let accY: Double
    let accZ: Double
    let gyroX: Double
    let gyroY: Double
    let gyroZ: Double
    let label: Int
}
