//
//  WatchConnectivityManager.swift
//  datacollector
//
//  Created by Kalyan reddy Katla on 12/25/25.
//


import WatchConnectivity

class WatchConnectivityManager: NSObject, WCSessionDelegate {

    static let shared = WatchConnectivityManager()

    override init() {
        super.init()
        if WCSession.isSupported() {
            WCSession.default.delegate = self
            WCSession.default.activate()
        }
    }

    func send(sample: WatchSensorSample) {
        guard WCSession.default.isReachable else { return }

        if let data = try? JSONEncoder().encode(sample),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            WCSession.default.sendMessage(json, replyHandler: nil)
        }
    }

    func session(_: WCSession,
                 activationDidCompleteWith _: WCSessionActivationState,
                 error _: Error?) {}
}
