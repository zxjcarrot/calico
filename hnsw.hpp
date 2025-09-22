#ifndef HNSW_HPP
#define HNSW_HPP

#include <atomic>
#include <cmath>
#include <functional>
#include <queue>
#include <random>
#include <vector>
#include <span>

#include "calico.hpp"
#include "distance.hpp"

// --- HNSW Metadata ---

struct HNSWMetadataPage {
    bool dirty;
    PID base_pid;                   // The starting Page ID for HNSW data pages.
    u64 max_elements;               // The maximum number of elements the index can hold.
    std::atomic<u64> node_count;    // The total number of nodes currently inserted.
    
    // Entry point information.
    u32 enter_point_node_id;        // The global entry point node ID for the graph.
    std::atomic<u32> max_level;     // The highest level currently present in the graph.
};


// --- HNSW Main Class ---

template <
    size_t Dim,                     // Vector dimension.
    // 2. Use the new SIMD-accelerated distance metric as the default.
    typename DistanceMetric = hnsw_distance::SIMDAcceleratedL2, 
    size_t M = 16,                  // Max number of neighbors for layers > 0.
    size_t M0 = 32,                 // Max number of neighbors for layer 0.
    size_t efConstruction = 200,    // Number of neighbors to search during index construction.
    size_t MaxLevel = 16            // Pre-estimated max level for random level generation.
>
class HNSW {
public:
    // Node slot within a page: a small, fixed-size metadata structure.
    struct HNSWNodeSlot {
        u16 vector_offset;      // Offset to the vector data in the page's heap area.
        u16 neighbors_offset;   // Offset to the neighbor list in the page's heap area.
        u32 level;              // The actual highest level of this node.
    };

    // HNSW data page: uses a slot/heap layout for high space efficiency.
    struct HNSWPage {
        bool dirty;
        u16 node_count;         // Number of nodes actually used on this page.
        u16 data_offset;        // Starting offset of the heap area (grows downwards from pageSize).
        u32 padding;            // Ensure alignment and size.
        
        static constexpr size_t HeaderSize = sizeof(dirty) + sizeof(node_count) + sizeof(data_offset) + sizeof(padding);
        // Calculate how many fixed-size slots can fit.
        static constexpr size_t MaxNodesPerPage = (pageSize - HeaderSize) / sizeof(HNSWNodeSlot);

        static_assert(MaxNodesPerPage > 0, "HNSWNodeSlot is too large to fit in a page.");

        HNSWNodeSlot slots[MaxNodesPerPage];
        // The rest of the page is the heap area, accessed via offsets.
    };

    static constexpr size_t NodesPerPage = HNSWPage::MaxNodesPerPage;

private:
    PID metadata_pid;
    PID base_pid; // The starting PID for HNSW data pages, determined at initialization.
    u64 max_elements_;

    // Random level generator.
    std::mt19937 level_generator;
    double mult_factor;

public:
    // Constructor: initializes and pre-allocates all necessary pages.
    HNSW(PID meta_pid, u64 max_elements);

    // Add a point to the index. The new node's ID is returned via the output parameter.
    void addPoint(const float* point, u32& node_id_out);

    // Search for the K nearest neighbors. Returns pairs of (distance, node_id).
    std::vector<std::pair<float, u32>> searchKnn(const float* query, size_t k);


private:
    // --- Node Location Calculation (O(1) Mapping) ---
    
    // Calculates the Page ID for a given node ID.
    PID getNodePID(u32 node_id) const {
        return base_pid + (node_id / NodesPerPage);
    }
    // Calculates the slot index within a page for a given node ID.
    u32 getSlotIndexInPage(u32 node_id) const {
        return node_id % NodesPerPage;
    }

    // --- Internal Helper: For safe and convenient access to node data within a page ---
    
    // This helper class provides a clean, read-only interface to a node's data,
    // abstracting away the details of the slot/heap layout.
    class NodeAccessor {
    private:
        const HNSWPage* page;
        const HNSWNodeSlot* slot;

    public:
        NodeAccessor(const HNSWPage* p, u32 slot_index) : page(p), slot(&p->slots[slot_index]) {}

        u32 getLevel() const { return slot->level; }
        
        const float* getVector() const {
            return reinterpret_cast<const float*>(reinterpret_cast<const u8*>(page) + slot->vector_offset);
        }

        // Gets the neighbor list for a specific level.
        // The implementation in hnsw.cpp will calculate the correct offset and length
        // within the node's flat neighbor list stored in the heap.
        std::span<const PID> getNeighbors(u32 level) const;
    };
    
    // --- Core Algorithm Implementations ---

    // Generates a random level for a new node based on the HNSW probability distribution.
    u32 getRandomLevel();

    // Greedily searches for the nearest neighbor to the query on a given layer.
    u32 searchLayer(const float* query, u32 entry_point_id, u32 target_level);
    
    // Selects the best neighbors from a candidate set using the HNSW heuristic.
    std::vector<u32> selectNeighborsHeuristic(const float* query, const std::vector<u32>& candidates, size_t M_limit);
};

#endif // HNSW_HPP