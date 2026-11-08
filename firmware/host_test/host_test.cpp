// Host-side check of the firmware SNN (same leak_snn.h as on the ESP32-S3) against PyTorch.
// Build & run from the repo root after `python export_firmware.py`:
//   c++ -O2 -std=c++17 firmware/host_test/host_test.cpp -o host_test && ./host_test
#include <chrono>
#include <cmath>
#include <cstdio>
#include "../wokwi_esp32s3/leak_snn.h"
#include "../wokwi_esp32s3/test_streams.h"

static double run(LeakDetector& det, const SpikeStream& s, bool dense, float& max_err) {
  uint8_t active[SNN_N_IN];
  uint32_t p = 0;
  max_err = 0.f;
  det.reset();
  auto t0 = std::chrono::steady_clock::now();
  for (int t = 0; t < STREAM_T; t++) {
    int n = 0;
    while (p < s.n_ev && (s.ev[p] >> 5) == t) active[n++] = s.ev[p++] & 31;
    det.step(active, n, dense);
    max_err = std::fmax(max_err, std::fabs(det.last_logit - s.ref_logit[t]));
  }
  return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
}

int main() {
  LeakDetector det;
  int fails = 0;
  std::printf("model %s, tau %.3f\n", SNN_SOURCE, SNN_TAU);
  for (int i = 0; i < N_STREAMS; i++) {
    const SpikeStream& s = STREAMS[i];
    float err_ev, err_dense;
    double us_ev = run(det, s, false, err_ev);
    int alarm_ev = det.alarm_t;
    uint32_t synops = det.snn.stats.synops;
    double us_dense = run(det, s, true, err_dense);
    int alarm_dense = det.alarm_t;
    bool ok = alarm_ev == s.py_alarm_ms && alarm_dense == s.py_alarm_ms && err_ev < 1e-3f && err_dense < 1e-3f;
    fails += !ok;
    std::printf("%-24s events %5u  alarm py %4d / c++ event %4d / c++ dense %4d  max|dlogit| %.1e / %.1e  "
                "synops %7u  time %.0f / %.0f us  %s\n",
                s.name, (unsigned)s.n_ev, s.py_alarm_ms, alarm_ev, alarm_dense, err_ev, err_dense,
                (unsigned)synops, us_ev, us_dense, ok ? "OK" : "MISMATCH");
  }
  std::printf("%s: %d/%d recordings match PyTorch\n", fails ? "FAIL" : "PASS", N_STREAMS - fails, N_STREAMS);
  return fails ? 1 : 0;
}
