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
    var currentLabel: Int = 0

    var onSample: ((WatchSensorSample) -> Void)?

    func start() {
        guard motionManager.isDeviceMotionAvailable else { return }

        isCollecting = true
        motionManager.deviceMotionUpdateInterval = 1.0 / 50.0 // 50 Hz

        motionManager.startDeviceMotionUpdates(
            using: .xArbitraryZVertical,
            to: queue
        ) { [weak self] motion, _ in
            guard let self = self,
                  let motion = motion,
                  self.isCollecting else { return }

            let acc = motion.userAcceleration
            let gyro = motion.rotationRate

            let sample = WatchSensorSample(
                timestamp: Date().timeIntervalSince1970,
                accX: acc.x,
                accY: acc.y,
                accZ: acc.z,
                gyroX: gyro.x,
                gyroY: gyro.y,
                gyroZ: gyro.z,
                label: self.currentLabel
            )

            self.onSample?(sample)
        }
    }

    func stop() {
        isCollecting = false
        motionManager.stopDeviceMotionUpdates()
    }
}
