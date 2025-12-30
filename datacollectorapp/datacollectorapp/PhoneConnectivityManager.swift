//
//  PhoneConnectivityManager.swift
//  habittracker
//
//  Created by Kalyan reddy Katla on 12/25/25.
//

import WatchConnectivity

class PhoneConnectivityManager: NSObject, WCSessionDelegate {

    override init() {
        super.init()

        if WCSession.isSupported() {
            let session = WCSession.default
            session.delegate = self
            session.activate()
        }
    }

    // REQUIRED by WCSessionDelegate (iOS)
    func session(_ session: WCSession,
                 activationDidCompleteWith activationState: WCSessionActivationState,
                 error: Error?) {
        // No-op (required)
    }

    func sessionDidBecomeInactive(_ session: WCSession) {
        // No-op (required)
    }

    func sessionDidDeactivate(_ session: WCSession) {
        // Required to reactivate
        WCSession.default.activate()
    }

    // DATA RECEIVER
    func session(_ session: WCSession,
                 didReceiveMessage message: [String : Any]) {

        print("Received sample:", message)
        // Later: write to CSV
    }
}
