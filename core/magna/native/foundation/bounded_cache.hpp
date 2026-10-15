// Small least-recently-used caches bounded by entry count and retained bytes.
// Lookups scan linearly; every owner keeps at most a few dozen entries.
// Bounds are passed per insertion because some owners read them from run
// options that can change between runs.
#pragma once
#include <cstddef>
#include <limits>
#include <list>
#include <utility>
namespace magic_native {
// Retained-byte bound shared by the preparation caches of exact algebra.
constexpr std::size_t default_cache_bytes = 16 * 1024 * 1024;
constexpr std::size_t unbounded_cache_bytes = std::numeric_limits<std::size_t>::max();

template <class Entry> class BoundedCache {
    struct Slot {
        Entry entry;
        std::size_t bytes;
    };
    std::list<Slot> slots;
    std::size_t bytes = 0;

public:
    // Return the first entry satisfying match, now the most recently used.
    template <class Match> Entry* find(Match&& match) {
        for(auto it = slots.begin(); it != slots.end(); it++)
            if(match(std::as_const(it->entry))) {
                slots.splice(slots.end(), slots, it);
                return &slots.back().entry;
            }
        return nullptr;
    }
    // Retain entry, evicting the least recently used ones so that both bounds
    // hold. An entry that alone exceeds max_bytes is not retained.
    Entry* insert(Entry entry, std::size_t max_entries, std::size_t entry_bytes = 0,
            std::size_t max_bytes = unbounded_cache_bytes) {
        if(!max_entries || entry_bytes > max_bytes) return nullptr;
        while(!slots.empty() && (slots.size() >= max_entries || bytes + entry_bytes > max_bytes)) {
            bytes -= slots.front().bytes;
            slots.pop_front();
        }
        bytes += entry_bytes;
        slots.push_back({std::move(entry), entry_bytes});
        return &slots.back().entry;
    }
    std::size_t size() const { return slots.size(); }
    void clear() {
        slots.clear();
        bytes = 0;
    }
};
} // namespace magic_native
