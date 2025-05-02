#include <chrono>

class Timer {
    using Clock = std::chrono::steady_clock;
    using TimePoint = std::chrono::time_point<Clock>;
    using Duration = std::chrono::microseconds;

public:
    Timer(): startTime{Clock::now()} {}

    void start() {
        startTime = Clock::now();
    }

    long long reset() {
        auto ret = std::chrono::duration_cast<Duration>(Clock::now() - startTime).count();
        start();
        return ret;
    }

    long long getTime() {
        return std::chrono::duration_cast<Duration>(Clock::now() - startTime).count();
    }

private:
    TimePoint startTime;
};
