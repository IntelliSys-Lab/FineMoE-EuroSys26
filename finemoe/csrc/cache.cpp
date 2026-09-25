#include "cache.h"
#include <algorithm>
#include <limits>

namespace finemoe {
namespace {
class DemandGuard {
    std::atomic<unsigned int>& demand_;
public:
    explicit DemandGuard(std::atomic<unsigned int>& demand) : demand_(demand) { ++demand_; }
    ~DemandGuard() { --demand_; }
};
}

ExpertCache::ExpertCache(const std::string& library, int device, void* stream,
                        std::size_t bytes, std::vector<Expert> experts,
                        const std::vector<void*>& destinations)
    : cuda_(library), device_(device), transfer_stream_(stream), bytes_(bytes),
      experts_(std::move(experts)) {
    if (experts_.empty() || destinations.empty() || !bytes_)
        throw std::invalid_argument("expert cache requires weights and slots");
    DeviceGuard guard(cuda_, device_);
    slots_.reserve(destinations.size());
    demand_scores_.assign(destinations.size(), std::numeric_limits<double>::infinity());
    prefetch_scores_ = demand_scores_;
    protected_experts_.reserve(experts_.size());
    try {
        for (void* destination : destinations) {
            slots_.push_back(Slot{destination});
            cuda_.check(cuda_.create_event(&slots_.back().ready, 2));
            cuda_.check(cuda_.create_event(&slots_.back().used, 2));
        }
    } catch (...) {
        destroy_events();
        throw;
    }
}

void ExpertCache::destroy_events() noexcept {
    for (auto& slot : slots_) {
        if (slot.ready) cuda_.destroy_event(slot.ready);
        if (slot.used) cuda_.destroy_event(slot.used);
    }
}

ExpertCache::~ExpertCache() {
    try {
        DeviceGuard guard(cuda_, device_);
        // Raw CUDA submissions must finish before Python releases their storage.
        cuda_.synchronize();
        destroy_events();
    } catch (...) {}
}

void ExpertCache::refresh(int index) {
    const auto& slot = slots_[index];
    double score = std::numeric_limits<double>::infinity();
    if (slot.expert >= 0 && !slot.leased) {
        const auto& expert = experts_[slot.expert];
        score = expert.probability * expert.frequency;
    }
    demand_scores_[index] = score;
    prefetch_scores_[index] = slot.expert >= 0 && experts_[slot.expert].protected_copy
        ? std::numeric_limits<double>::infinity() : score;
}

void ExpertCache::rebuild() {
    for (std::size_t i = 0; i < slots_.size(); ++i) refresh(static_cast<int>(i));
}

void ExpertCache::protect(int index) {
    auto& expert = experts_[index];
    if (expert.protected_copy) return;
    expert.protected_copy = true;
    if (!expert.protection_tracked) {
        protected_experts_.push_back(index);
        expert.protection_tracked = true;
    }
    if (expert.slot >= 0) prefetch_scores_[expert.slot] = std::numeric_limits<double>::infinity();
}

int ExpertCache::load(int index, bool speculative) {
    auto& expert = experts_[index];
    if (expert.slot >= 0) return expert.slot;
    const bool free = free_slot_ < slots_.size();
    int victim = free ? static_cast<int>(free_slot_) : -1;
    if (!free) {
        const auto& scores = speculative ? prefetch_scores_ : demand_scores_;
        double best = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < scores.size(); ++i) {
            const double score = scores[i];
            if (score < best || (victim >= 0 && score == best && slots_[i].expert < slots_[victim].expert)) {
                best = score;
                victim = static_cast<int>(i);
            }
        }
    }
    if (victim < 0) return -1;
    auto& slot = slots_[victim];
    if (slot.use_recorded) cuda_.check(cuda_.wait(transfer_stream_, slot.used, 0));
    cuda_.check(cuda_.copy(slot.destination, expert.source, bytes_, 1, transfer_stream_));
    cuda_.check(cuda_.record(slot.ready, transfer_stream_));
    if (slot.expert >= 0) experts_[slot.expert].slot = -1;
    slot.expert = index;
    slot.ready_complete = false;
    slot.wait_recorded = false;
    expert.slot = victim;
    if (free) ++free_slot_;
    refresh(victim);
    return victim;
}

bool ExpertCache::contains(int expert) {
    std::lock_guard<std::mutex> lock(mutex_);
    return experts_[expert].slot >= 0;
}

bool ExpertCache::prefetch(int expert) {
    if (demand_.load()) return false;
    DeviceGuard guard(cuda_, device_);
    std::lock_guard<std::mutex> lock(mutex_);
    if (demand_.load()) return false;
    if (experts_[expert].slot >= 0) return true;
    if (speculative_slot_ >= 0 && !cuda_.complete(slots_[speculative_slot_].ready)) return false;
    int slot = load(expert, true);
    if (slot < 0) return false;
    speculative_slot_ = slot;
    protect(expert);
    return true;
}

void ExpertCache::expire_locked(int layer, bool all) {
    auto end = std::remove_if(protected_experts_.begin(), protected_experts_.end(), [&](int index) {
        auto& expert = experts_[index];
        if (expert.protected_copy && !all && expert.layer > layer) return false;
        expert.protected_copy = false;
        expert.protection_tracked = false;
        if (expert.slot >= 0) prefetch_scores_[expert.slot] = demand_scores_[expert.slot];
        return true;
    });
    protected_experts_.erase(end, protected_experts_.end());
}

void ExpertCache::expire(int layer, bool all) {
    std::lock_guard<std::mutex> lock(mutex_);
    expire_locked(layer, all);
    if (all) forecasts_.clear();
}

void ExpertCache::probabilities_locked(const std::vector<int>& indices, const std::vector<double>& values) {
    for (std::size_t i = 0; i < indices.size(); ++i) {
        auto& expert = experts_[indices[i]];
        if (expert.probability == values[i]) continue;
        expert.probability = values[i];
        if (expert.slot >= 0) refresh(expert.slot);
    }
}

void ExpertCache::probabilities(const std::vector<int>& indices, const std::vector<double>& values) {
    for (double value : values)
        if (!(value >= 0.0 && value <= 1.0)) throw std::invalid_argument("invalid cache probability");
    std::lock_guard<std::mutex> lock(mutex_);
    probabilities_locked(indices, values);
}

void ExpertCache::plan(int layer, const std::vector<int>& indices, const std::vector<double>& values,
                       const std::vector<int>& order) {
    for (std::size_t i = 0; i < indices.size(); ++i) {
        if (!(values[i] >= 0.0 && values[i] <= 1.0) || experts_[indices[i]].layer != layer)
            throw std::invalid_argument("invalid layer forecast");
    }
    Forecast forecast;
    forecast.experts.reserve(order.size());
    for (int position : order) forecast.experts.emplace_back(indices[position], values[position]);
    std::lock_guard<std::mutex> lock(mutex_);
    forecasts_.insert_or_assign(layer, std::move(forecast));
    probabilities_locked(indices, values);
}

void ExpertCache::advance(int layer) {
    std::lock_guard<std::mutex> lock(mutex_);
    expire_locked(layer, false);
    forecasts_.erase(forecasts_.begin(), forecasts_.upper_bound(layer));
    if (forecasts_.empty()) return;
    DeviceGuard guard(cuda_, device_);
    if (speculative_slot_ >= 0 && !cuda_.complete(slots_[speculative_slot_].ready)) return;
    Forecast* best = nullptr;
    double priority = -1.0;
    for (auto& entry : forecasts_) {
        auto& forecast = entry.second;
        while (forecast.next < forecast.experts.size() &&
               experts_[forecast.experts[forecast.next].first].slot >= 0) ++forecast.next;
        if (forecast.next == forecast.experts.size()) continue;
        double urgency = forecast.experts[forecast.next].second / (entry.first - layer);
        if (urgency > priority) {
            best = &forecast;
            priority = urgency;
        }
    }
    if (!best || demand_.load()) return;
    int expert = best->experts[best->next].first;
    int slot = load(expert, true);
    if (slot < 0) return;
    speculative_slot_ = slot;
    protect(expert);
    ++best->next;
}

void ExpertCache::reset_probabilities(double value) {
    if (!(value >= 0.0 && value <= 1.0)) throw std::invalid_argument("invalid cache probability");
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& expert : experts_) expert.probability = value;
    rebuild();
}

void ExpertCache::wait_locked(Slot& slot, void* stream) {
    if (!slot.ready_complete && (!slot.wait_recorded || slot.waited_stream != stream)) {
        cuda_.check(cuda_.wait(stream, slot.ready, 0));
        slot.waited_stream = stream;
        slot.wait_recorded = true;
    }
}

int ExpertCache::acquire_locked(int expert, void* stream, bool wait) {
    int index = experts_[expert].slot;
    if (index >= 0) {
        auto& slot = slots_[index];
        if (slot.leased) throw std::runtime_error("expert is already executing");
        slot.ready_complete = slot.ready_complete || cuda_.complete(slot.ready);
        if (slot.ready_complete) ++hits_; else ++misses_;
    } else {
        ++misses_;
        index = load(expert, false);
        if (index < 0) throw std::runtime_error("all expert slots are leased");
    }
    auto& slot = slots_[index];
    if (wait) wait_locked(slot, stream);
    slot.leased = true;
    experts_[expert].protected_copy = false;
    ++experts_[expert].frequency;
    refresh(index);
    return index;
}

void ExpertCache::lookahead_locked(int expert) {
    if (expert >= 0 && experts_[expert].slot < 0 && load(expert, false) >= 0)
        protect(expert);
}

int ExpertCache::acquire(int expert, int next, void* stream) {
    DemandGuard demand(demand_);
    DeviceGuard guard(cuda_, device_);
    std::lock_guard<std::mutex> lock(mutex_);
    int index = acquire_locked(expert, stream, true);
    try { lookahead_locked(next); }
    catch (...) { release_locked(index, stream); throw; }
    return index;
}

void ExpertCache::release_locked(int index, void* stream) {
    auto& slot = slots_[index];
    cuda_.check(cuda_.record(slot.used, stream));
    slot.use_recorded = true;
    slot.leased = false;
    refresh(index);
}

void ExpertCache::release(int index, void* stream) {
    DeviceGuard guard(cuda_, device_);
    std::lock_guard<std::mutex> lock(mutex_);
    if (!slots_[index].leased) throw std::runtime_error("expert slot is not leased");
    release_locked(index, stream);
}

void ExpertCache::configure(const std::vector<void*>& graphs, Workspace workspace) {
    DeviceGuard guard(cuda_, device_);
    std::lock_guard<std::mutex> lock(mutex_);
    cuda_.check(cuda_.synchronize());
    for (std::size_t i = 0; i < slots_.size(); ++i) slots_[i].graph = graphs[i];
    workspace_ = workspace;
}

void ExpertCache::dispatch(const std::vector<std::pair<int, int>>& ranked_experts,
                           const void* hidden, const void* weights, std::size_t stride,
                           void* shared_graph, void* stream) {
    DeviceGuard guard(cuda_, device_);
    if (!workspace_.input) throw std::runtime_error("expert graphs have not been warmed up");
    cuda_.check(cuda_.copy(workspace_.input, hidden, workspace_.bytes, 3, stream));
    cuda_.check(cuda_.zero(workspace_.accumulator, 0, workspace_.bytes, stream));
    for (std::size_t i = 0; i < ranked_experts.size(); ++i) {
        int index;
        {
            DemandGuard demand(demand_);
            std::lock_guard<std::mutex> lock(mutex_);
            index = acquire_locked(ranked_experts[i].second, stream, false);
            try {
                if (i + 1 < ranked_experts.size()) lookahead_locked(ranked_experts[i + 1].second);
            } catch (...) { release_locked(index, stream); throw; }
        }
        try {
            if (i == 0) cuda_.check(cuda_.launch(shared_graph, stream));
            auto& slot = slots_[index];
            {
                std::lock_guard<std::mutex> lock(mutex_);
                wait_locked(slot, stream);
            }
            const char* weight = static_cast<const char*>(weights) + ranked_experts[i].first * stride;
            cuda_.check(cuda_.copy(workspace_.weight, weight, 2, 3, stream));
            cuda_.check(cuda_.launch(slot.graph, stream));
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex_);
            release_locked(index, stream);
            throw;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        release_locked(index, stream);
    }
}

std::pair<std::uint64_t, std::uint64_t> ExpertCache::stats() {
    std::lock_guard<std::mutex> lock(mutex_);
    return {hits_, misses_};
}

void ExpertCache::clear() {
    DemandGuard demand(demand_);
    DeviceGuard guard(cuda_, device_);
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& slot : slots_)
        if (slot.leased) throw std::runtime_error("cannot clear a cache while an expert is executing");
    cuda_.check(cuda_.synchronize());
    for (auto& slot : slots_) slot.expert = -1;
    for (auto& expert : experts_) {
        expert.slot = -1;
        expert.frequency = 0;
        expert.probability = 1.0;
        expert.protected_copy = false;
        expert.protection_tracked = false;
    }
    protected_experts_.clear();
    forecasts_.clear();
    std::fill(demand_scores_.begin(), demand_scores_.end(), std::numeric_limits<double>::infinity());
    std::fill(prefetch_scores_.begin(), prefetch_scores_.end(), std::numeric_limits<double>::infinity());
    free_slot_ = 0;
    speculative_slot_ = -1;
    hits_ = misses_ = 0;
}

} // namespace finemoe
