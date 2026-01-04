//
//  WatchSensorSample.swift
//  datacollector
//
//  Created by Kalyan reddy Katla on 12/25/25.
//


import Foundation

/// Complete sensor data including all available motion data
struct WatchSensorSample: Codable {
    // Timestamp
    let timestamp: Double           // Unix timestamp (seconds)
    let timestampNanos: Int64       // Nanoseconds since epoch
    
    // User Acceleration (gravity removed)
    let accX: Double
    let accY: Double
    let accZ: Double
    
    // Rotation Rate (gyroscope)
    let gyroX: Double
    let gyroY: Double
    let gyroZ: Double
    
    // Gravity vector
    let gravityX: Double
    let gravityY: Double
    let gravityZ: Double
    
    // Attitude (quaternion)
    let quaternionW: Double
    let quaternionX: Double
    let quaternionY: Double
    let quaternionZ: Double
    
    // Attitude (Euler angles in radians)
    let pitch: Double
    let roll: Double
    let yaw: Double
}
