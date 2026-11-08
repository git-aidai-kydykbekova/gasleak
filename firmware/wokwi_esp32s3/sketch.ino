// Ultrasonic gas-leak detector demo on ESP32-S3 (runs on a real board or in the Wokwi simulator).
//
// Plays recorded encoder spike streams (test clips exported from the Python pipeline) through the
// event-driven SNN in real time (1 time step = 1 ms) and shows the decision on two LEDs:
//   green  (GPIO 5) : listening, no leak         red (GPIO 4) : LEAK ALARM
//   button (GPIO 6) : skip to the next recording
// For every recording it prints over Serial:
//   * detection result and delay vs. the true leak onset
//   * max |logit(ESP32) - logit(PyTorch)| and whether the alarm time matches Python exactly
//   * measured compute time per 1 ms step, event-driven vs. the same network run frame-based (dense)
//     -> fraction of each millisecond the CPU is busy (the rest could be spent in light sleep)
#include <Arduino.h>
#include "leak_snn.h"
#include "test_streams.h"

const int LED_ALARM = 4, LED_LISTEN = 5, BTN_NEXT = 6;

LeakDetector det;
uint8_t active[SNN_N_IN];

// collect the input channels that spiked at step t; ev is sorted by time, encoded (t << 5) | channel
int gather(const SpikeStream& s, uint32_t& p, int t) {
  int n = 0;
  while (p < s.n_ev && (s.ev[p] >> 5) == t) active[n++] = s.ev[p++] & 31;
  return n;
}

bool skip_requested() {
  static uint32_t last = 0;
  if (digitalRead(BTN_NEXT) == LOW && millis() - last > 300) { last = millis(); return true; }
  return false;
}

// returns busy microseconds; real_time = pace at 1 step per ms and drive the LEDs
uint32_t run(const SpikeStream& s, bool dense, bool real_time, float& max_err) {
  det.reset();
  uint32_t p = 0, busy = 0, next = micros();
  max_err = 0.f;
  for (int t = 0; t < STREAM_T; t++) {
    uint32_t t0 = micros();
    int n = gather(s, p, t);
    bool alarm = det.step(active, n, dense);
    busy += micros() - t0;
    float err = fabsf(det.last_logit - s.ref_logit[t]);
    if (err > max_err) max_err = err;
    if (real_time) {
      digitalWrite(LED_ALARM, alarm); digitalWrite(LED_LISTEN, !alarm);
      if (alarm && det.alarm_t == t) Serial.printf("  !! LEAK ALARM at t = %d ms\n", t);
      if (skip_requested()) return busy;
      next += 1000;
      while ((int32_t)(micros() - next) < 0) {}
    }
  }
  return busy;
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_ALARM, OUTPUT); pinMode(LED_LISTEN, OUTPUT); pinMode(BTN_NEXT, INPUT_PULLUP);
  delay(200);
  Serial.printf("\n=== Event-driven SNN gas-leak detector on %s @ %d MHz ===\n", ESP.getChipModel(), getCpuFrequencyMhz());
  Serial.printf("model: %s | threshold logit %.3f | K = %d ms | warm-up %d ms | %d recordings\n",
                SNN_SOURCE, SNN_TAU, SNN_K, SNN_WARMUP, N_STREAMS);
}

void loop() {
  int ok = 0;
  for (int i = 0; i < N_STREAMS; i++) {
    const SpikeStream& s = STREAMS[i];
    Serial.printf("\n[%d/%d] %s  (%u input events in %d ms)\n", i + 1, N_STREAMS, s.name, (unsigned)s.n_ev, STREAM_T);
    float err_ev, err_dense;
    uint32_t busy_ev = run(s, false, true, err_ev);
    SnnStats st = det.snn.stats;
    int alarm = det.alarm_t;
    uint32_t busy_dense = run(s, true, false, err_dense);
    digitalWrite(LED_ALARM, LOW); digitalWrite(LED_LISTEN, HIGH);

    bool correct = s.label ? (alarm >= s.onset_ms) : (alarm < 0);
    if (s.label && alarm >= s.onset_ms) Serial.printf("  leak onset %d ms -> detected after %d ms\n", s.onset_ms, alarm - s.onset_ms);
    else if (s.label && alarm >= 0)     Serial.printf("  leak onset %d ms -> FALSE ALARM before onset (t = %d ms)\n", s.onset_ms, alarm);
    else if (s.label)                   Serial.printf("  leak onset %d ms -> MISSED\n", s.onset_ms);
    else                                Serial.printf("  no leak -> %s\n", alarm < 0 ? "no alarm (correct)" : "FALSE ALARM");
    bool same = alarm == s.py_alarm_ms;
    ok += same;
    Serial.printf("  result %s | alarm time matches PyTorch: %s (py %d, esp %d) | max |logit diff| %.2e\n",
                  correct ? "CORRECT" : "WRONG", same ? "yes" : "NO", s.py_alarm_ms, alarm, err_ev);
    Serial.printf("  compute per 1 ms step: event-driven %.1f us (%.1f %% busy) | dense %.1f us (%.1f %% busy) | %.1fx less\n",
                  busy_ev / (float)STREAM_T, busy_ev / (10.f * STREAM_T), busy_dense / (float)STREAM_T,
                  busy_dense / (10.f * STREAM_T), busy_dense / (float)(busy_ev ? busy_ev : 1));
    Serial.printf("  spikes: in %u, L1 %u, L2 %u | synaptic ops %u (dense would use %u MACs)\n",
                  (unsigned)st.in_events, (unsigned)st.s1_spikes, (unsigned)st.s2_spikes, (unsigned)st.synops,
                  (unsigned)(STREAM_T * (SNN_N_IN * SNN_N_HID + SNN_N_HID * SNN_N_HID + SNN_N_HID)));
    delay(1500);
  }
  Serial.printf("\n=== %d/%d recordings match the PyTorch model exactly; replaying ===\n", ok, N_STREAMS);
  delay(3000);
}
