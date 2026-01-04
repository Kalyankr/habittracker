//
//  WatchMotionManager.swift
//  datacollector
//
//  Created by Kalyan reddy Katla on 12/25/25.
//


import CoreMotion

class WatchMotionManager {

    private let motionManager = CMMotionManager()
    private let queue = OperationQueue()

    var isCollecting = false
    
    // Session start time
    private var sessionStartTime: Date?
    private var sessionStartNanos: Int64 = 0
    
    // Buffer to store samples during session
    private(set) var samples: [WatchSensorSample] = []
    
    // Annotations storage
    private(set) var annotations: [Annotation] = []
    
    // For tracking button press duration
    private var buttonPressStart: Date?

    var onSampleCountUpdate: ((Int) -> Void)?

    func start() {
        guard motionManager.isDeviceMotionAvailable else { return }

        // Clear previous session data
        samples.removeAll()
        annotations.removeAll()
        
        // Record session start
        sessionStartTime = Date()
        sessionStartNanos = Int64(Date().timeIntervalSince1970 * 1_000_000_000)
        
        isCollecting = true
        motionManager.deviceMotionUpdateInterval = 1.0 / 50.0 // 50 Hz

        motionManager.startDeviceMotionUpdates(
            using: .xArbitraryZVertical,
            to: queue
        ) { [weak self] motion, _ in
            guard let self = self,
                  let motion = motion,
                  self.isCollecting else { return }

            let now = Date()
            let timestamp = now.timeIntervalSince1970
            let timestampNanos = Int64(timestamp * 1_000_000_000)
            
            let acc = motion.userAcceleration
            let gyro = motion.rotationRate
            let gravity = motion.gravity
            let attitude = motion.attitude
            let quat = attitude.quaternion

            let sample = WatchSensorSample(
                timestamp: timestamp,
                timestampNanos: timestampNanos,
                accX: acc.x,
                accY: acc.y,
                accZ: acc.z,
                gyroX: gyro.x,
                gyroY: gyro.y,
                gyroZ: gyro.z,
                gravityX: gravity.x,
                gravityY: gravity.y,
                gravityZ: gravity.z,
                quaternionW: quat.w,
                quaternionX: quat.x,
                quaternionY: quat.y,
                quaternionZ: quat.z,
                pitch: attitude.pitch,
                roll: attitude.roll,
                yaw: attitude.yaw
            )

            // Buffer sample
            self.samples.append(sample)
            
            // Notify count update on main thread
            DispatchQueue.main.async {
                self.onSampleCountUpdate?(self.samples.count)
            }
        }
    }

    func stop() {
        isCollecting = false
        motionManager.stopDeviceMotionUpdates()
    }
    
    // MARK: - Annotation Management
    
    /// Called when button is pressed down
    func buttonDown() {
        buttonPressStart = Date()
    }
    
    /// Called when button is released - records annotation
    func recordAnnotation(label: String) {
        guard let startTime = sessionStartTime else { return }
        
        let now = Date()
        let timeNanos = Int64(now.timeIntervalSince1970 * 1_000_000_000)
        let secondsElapsed = now.timeIntervalSince(startTime)
        
        // Calculate press duration
        let pressDuration: Int
        if let pressStart = buttonPressStart {
            pressDuration = Int(now.timeIntervalSince(pressStart) * 1000)
        } else {
            pressDuration = 50 // Default
        }
        buttonPressStart = nil
        
        let annotation = Annotation(
            time: timeNanos,
            secondsElapsed: secondsElapsed,
            text: label,
            millisecondPressDuration: pressDuration
        )
        annotations.append(annotation)
    }
    
    /// Get the sample count
    var sampleCount: Int {
        return samples.count
    }
    
    /// Get annotation count
    var annotationCount: Int {
        return annotations.count
    }
    
    /// Convert buffered samples to CSV format with all sensor data
    func getSensorCSVData() -> Data {
        var csv = "timestamp,timestampNanos,accX,accY,accZ,gyroX,gyroY,gyroZ,gravityX,gravityY,gravityZ,quaternionW,quaternionX,quaternionY,quaternionZ,pitch,roll,yaw\n"
        
        for s in samples {
            csv += "\(s.timestamp),\(s.timestampNanos),\(s.accX),\(s.accY),\(s.accZ),\(s.gyroX),\(s.gyroY),\(s.gyroZ),\(s.gravityX),\(s.gravityY),\(s.gravityZ),\(s.quaternionW),\(s.quaternionX),\(s.quaternionY),\(s.quaternionZ),\(s.pitch),\(s.roll),\(s.yaw)\n"
        }
        
        return csv.data(using: .utf8) ?? Data()
    }
    
    /// Convert annotations to CSV format matching user's existing format
    func getAnnotationsCSVData() -> Data {
        var csv = "time\tseconds_elapsed\ttext\tmillisecond_press_duration\n"
        
        for a in annotations {
            csv += "\(a.time)\t\(a.secondsElapsed)\t\(a.text)\t\(a.millisecondPressDuration)\n"
        }
        
        return csv.data(using: .utf8) ?? Data()
    }
    
    /// Clear all buffered data
    func clearData() {
        samples.removeAll()
        annotations.removeAll()
        sessionStartTime = nil
    }
}
