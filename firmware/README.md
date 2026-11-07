# ESP32-S3 firmware: event-driven SNN leak detector (Wokwi or real board)

The trained `LeakSNN` runs as plain C++ on an ESP32-S3. Encoder spike streams of real test clips
from the Python pipeline are played back in real time (1 step = 1 ms), and the result is shown on LEDs
and the serial monitor.

```
firmware/
  wokwi_esp32s3/          <- everything Wokwi / Arduino needs
    sketch.ino            demo: playback, LEDs, button, serial report
    leak_snn.h            event-driven SNN + alarm logic (portable C++, no dependencies)
    snn_weights.h         GENERATED: weights, decays, thresholds      (export_firmware.py)
    test_streams.h        GENERATED: 6 test clips + PyTorch logits     (export_firmware.py)
    diagram.json          Wokwi circuit: ESP32-S3 DevKitC-1, red/green LED, button
  host_test/host_test.cpp checks the C++ engine against PyTorch on a PC
```

## Circuit
| Part | Pin | Meaning |
|---|---|---|
| Red LED (+220 Ω) | GPIO 4 | **LEAK ALARM** |
| Green LED (+220 Ω) | GPIO 5 | listening, no leak |
| Push-button to GND | GPIO 6 | skip to the next recording |

## Run in the browser (Wokwi, no hardware needed)
1. Open <https://wokwi.com> → **New project** → **ESP32-S3** (Arduino).
2. Replace the contents of `sketch.ino` and `diagram.json` with the files from `wokwi_esp32s3/`.
3. Use the file-tab menu → **New file…** to add `leak_snn.h`, `snn_weights.h` and `test_streams.h` (paste the contents).
4. Press ▶. The serial monitor prints one block per recording:
   ```
   [4/6] leak, SNR 0 dB  (3142 input events in 998 ms)
     !! LEAK ALARM at t = 443 ms
     leak onset 396 ms -> detected after 47 ms
     result CORRECT | alarm time matches PyTorch: yes (py 443, esp 443) | max |logit diff| 3.8e-05
     compute per 1 ms step: event-driven .. us (..% busy) | dense .. us (..% busy) | ..x less
   ```
   The red LED lights up at the moment of detection. Press **NEXT** to skip.

## Run on a real ESP32-S3
Arduino IDE (esp32 core by Espressif) → board **ESP32S3 Dev Module** → open `wokwi_esp32s3/sketch.ino`
(rename the folder to `sketch` if the IDE asks) → Upload → Serial Monitor at 115200 baud.
Wire the LEDs and the button as in the table. To measure real power, power the board through a
Nordic PPK2 / INA226 / Joulescope.

## What this demo shows and what it does not
* **Shows:** the trained network runs on the target MCU with the same alarm decision as PyTorch, bit for bit.
  It also shows how much of each 1 ms the CPU is busy, event-driven vs. the same network computed frame-based.
* **Does not show:** power in mW. Wokwi does not model current draw, and its timing is only approximately
  cycle-accurate. The input is recorded encoder spikes, not a live ultrasonic microphone. The front-end
  (FFT and encoder) is not yet on the MCU. Both are next steps; see `docs/PROJECT_GUIDE.md`.

## Regenerate after training
```bash
python export_firmware.py      # uses runs/seeds/0/snn.pt + its FA-5 % threshold once they exist
c++ -O2 -std=c++17 firmware/host_test/host_test.cpp -o host_test && ./host_test   # must print PASS
```
The committed headers come from the first training run (`runs/snn.pt`, threshold P(leak) > 0.9).
