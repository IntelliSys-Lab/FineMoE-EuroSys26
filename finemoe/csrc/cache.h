#pragma once

#include "cuda_runtime.h"
#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <utility>
#include <vector>

namespace finemoe {

struct Expert {
    int layer;
    const void* source;
    int slot = -1;
    double probability = 1.0;
    std::uint64_t frequency = 0;
    bool protected_copy = false;
    bool protection_tracked = false;
};

struct Slot {
    void* destination;
    void* ready = nullptr;
    void* used = nullptr;
    void* graph = nullptr;
    void* waited_stream = nullptr;
    int expert = -1;
    bool leased = false;
    bool ready_complete = false;
    bool use_recorded = false;
    bool wait_recorded = false;
};

struct Workspace {
    void* input = nullptr;
    void* weight = nullptr;
    void* accumulator = nullptr;
    std::size_t bytes = 0;
};

struct Forecast {
    std::vector<std::pair<int, double>> experts;
    std::size_t next = 0;
};

class ExpertCache {
    CudaRuntime cuda_;
    int device_;
    void* transfer_stream_;
    std::size_t bytes_;
    std::vector<Expert> experts_;
    std::vector<Slot> slots_;
    std::vector<double> demand_scores_, prefetch_scores_;
    std::vector<int> protected_experts_;
    std::size_t free_slot_ = 0;
    std::map<int, Forecast> forecasts_;
    Workspace workspace_;
    std::mutex mutex_;
    std::atomic<unsigned int> demand_{0};
    int speculative_slot_ = -1;
    std::uint64_t hits_ = 0, misses_ = 0;

    int load(int expert, bool speculative);
    void refresh(int slot);
    void rebuild();
    void protect(int expert);
    void expire_locked(int layer, bool all);
    void probabilities_locked(const std::vector<int>& indices, const std::vector<double>& values);
    int acquire_locked(int expert, void* stream, bool wait);
    void release_locked(int slot, void* stream);
    void wait_locked(Slot& slot, void* stream);
    void lookahead_locked(int expert);
    void destroy_events() noexcept;

public:
    ExpertCache(const std::string& library, int device, void* stream, std::size_t bytes,
                std::vector<Expert> experts, const std::vector<void*>& destinations);
    ~ExpertCache();
    std::size_t size() const { return experts_.size(); }
    std::size_t capacity() const { return slots_.size(); }
    bool contains(int expert);
    bool prefetch(int expert);
    void expire(int layer, bool all);
    void probabilities(const std::vector<int>& indices, const std::vector<double>& values);
    void plan(int layer, const std::vector<int>& indices, const std::vector<double>& values,
              const std::vector<int>& order);
    void advance(int layer);
    void reset_probabilities(double value);
    int acquire(int expert, int next, void* stream);
    void release(int slot, void* stream);
    void configure(const std::vector<void*>& graphs, Workspace workspace);
    void dispatch(const std::vector<std::pair<int, int>>& ranked_experts,
                  const void* hidden, const void* weights, std::size_t weight_stride,
                  void* shared_graph, void* stream);
    std::pair<std::uint64_t, std::uint64_t> stats();
    void clear();
};

} // namespace finemoe
