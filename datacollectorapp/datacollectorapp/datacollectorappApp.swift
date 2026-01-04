//
//  datacollectorappApp.swift
//  datacollectorapp
//
//  Created by Kalyan reddy Katla on 12/30/25.
//

import SwiftUI

@main
struct datacollectorappApp: App {
    @StateObject private var connectivityManager = PhoneConnectivityManager.shared
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(connectivityManager)
        }
    }
}
