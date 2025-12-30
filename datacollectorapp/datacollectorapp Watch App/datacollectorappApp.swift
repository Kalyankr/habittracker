//
//  datacollectorappApp.swift
//  datacollectorapp Watch App
//
//  Created by Kalyan reddy Katla on 12/30/25.
//

import SwiftUI

@main
struct datacollectorapp_Watch_AppApp: App {
    init() {
            _ = WatchConnectivityManager.shared
        }

        var body: some Scene {
            WindowGroup {
                ContentView()
            }
        }
}
