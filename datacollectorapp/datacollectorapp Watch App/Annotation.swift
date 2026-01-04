//
//  Annotation.swift
//  datacollectorapp Watch App
//
//  Created by Kalyan reddy Katla on 01/04/26.
//

import Foundation

/// Annotation matching existing format: time, seconds_elapsed, text, millisecond_press_duration
struct Annotation: Codable {
    let time: Int64              // Nanoseconds since epoch
    let secondsElapsed: Double   // Seconds since session start
    let text: String             // "yes" or "no"
    let millisecondPressDuration: Int // Duration of button press in ms
}
