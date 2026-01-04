//
//  SensorSample.swift
//  datacollectorapp
//
//  Created by Kalyan reddy Katla on 12/30/25.
//

import Foundation

/// Mirror of WatchSensorSample for iPhone app
struct SensorSample: Codable, Identifiable {
    var id: Double { timestamp }
    
    let timestamp: Double
    let accX: Double
    let accY: Double
    let accZ: Double
    let gyroX: Double
    let gyroY: Double
    let gyroZ: Double
    let label: Int
}
