#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h> // Needed for py::function
#include <optional>
#include <iostream>
#include <memory>
#include <vector>

#include "calico.hpp"

namespace py = pybind11;

// --- Forward declare init/shutdown functions from calico.cpp ---
void initialize_system();
void shutdown_system();

// --- Automatic Initializer ---
struct SystemInitializer {
    SystemInitializer() {
        std::cout << "Calico module loading: Initializing system..." << std::endl;
        initialize_system();
        std::cout << "Calico system initialized." << std::endl;
    }
    ~SystemInitializer() {
        std::cout << "Calico module unloading: Shutting down system..." << std::endl;
        shutdown_system();
    }
};
static SystemInitializer initializer;

// --- Global BTree Object ---
static BTree global_btree;

// --- Python Module Definition ---
PYBIND11_MODULE(calico_py, m) {
    m.doc() = "Python bindings for the Calico B-Tree (callback-based scan)";

    // --- Module-level functions operating on the single global BTree instance ---

    m.def("insert", [](py::bytes key, py::bytes val) {
        std::string key_str = key;
        std::string val_str = val;
        global_btree.insert(
            {(u8*)key_str.data(), key_str.size()},
            {(u8*)val_str.data(), val_str.size()}
        );
    }, py::arg("key"), py::arg("value"), "Insert a key-value pair into the active B-Tree.");

    m.def("lookup", [](py::bytes key) -> std::optional<py::bytes> {
        std::string key_str = key;
        std::vector<u8> result;
        bool found = global_btree.lookup(
            {(u8*)key_str.data(), key_str.size()},
            [&](std::span<u8> payload) {
                result.assign(payload.begin(), payload.end());
            }
        );
        if (!found) {
            return std::nullopt;
        }
        return py::bytes(reinterpret_cast<const char*>(result.data()), result.size());
    }, py::arg("key"), "Retrieve the value for a given key. Returns None if not found.");

    m.def("remove", [](py::bytes key) -> bool {
        std::string key_str = key;
        return global_btree.remove({(u8*)key_str.data(), key_str.size()});
    }, py::arg("key"), "Delete a key-value pair. Returns True if successful, False otherwise.");
    
    m.def("contains", [](py::bytes key) -> bool {
        std::string key_str = key;
        return global_btree.lookup({(u8*)key_str.data(), key_str.size()}, [](std::span<u8>){});
    }, py::arg("key"), "Check if a key exists in the tree.");
    
    // --- NEW: Callback-based scan function ---
    m.def("scan", [](py::bytes start_key, const py::function& callback) {
        std::string key_str = start_key;
        std::span<u8> key_span(reinterpret_cast<u8*>(key_str.data()), key_str.size());

        // This C++ lambda will be called by BTree::scanAsc for each item.
        global_btree.scanAsc(key_span, 
            [&](BTreeNode& node, unsigned slot) -> bool {
                // 1. Reconstruct the full key from the node's prefix and the slot's key part.
                u16 prefixLen = node.prefixLen;
                u16 keyLen = node.slot[slot].keyLen;
                std::vector<u8> full_key(prefixLen + keyLen);
                memcpy(full_key.data(), node.getPrefix(), prefixLen);
                memcpy(full_key.data() + prefixLen, node.getKey(slot), keyLen);

                // 2. Get the payload.
                std::span<u8> payload = node.getPayload(slot);

                // 3. Convert C++ data to Python bytes objects.
                py::bytes py_key(reinterpret_cast<const char*>(full_key.data()), full_key.size());
                py::bytes py_payload(reinterpret_cast<const char*>(payload.data()), payload.size());

                // 4. Call the Python callback function.
                py::object result = callback(py_key, py_payload);
                
                // 5. Check the return value from Python to see if we should continue.
                // We explicitly check for a `False` return. `None` or other truthy values continue.
                if (py::isinstance<py::bool_>(result) && !result.cast<bool>()) {
                    return false; // Stop scanning.
                }

                return true; // Continue scanning.
            }
        );
    }, py::arg("start_key"), py::arg("callback"), 
       "Scans the B-Tree starting from a key, invoking a callback for each item. "
       "The callback should return False to stop the scan.");
}