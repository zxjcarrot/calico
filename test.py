import calico_py
import os

# -----------------------------------------------------------------------------
# NOTE: This test script assumes a single, persistent B-Tree.
# Tests are designed to run sequentially, and the state from one test
# carries over to the next. A dictionary `expected_db_state` is used
# to track the expected contents of the B-Tree throughout the run.
# -----------------------------------------------------------------------------

def run_test(test_name, test_func, *args):
    """Helper function to run a test and print its status."""
    print(f"--- Running test: {test_name} ---")
    try:
        test_func(*args)
        print(f"[SUCCESS] Test '{test_name}' passed.\n")
    except Exception as e:
        print(f"[FAILURE] Test '{test_name}' failed: {e}\n")
        exit(1)

def test_insert_and_lookup(state):
    """Tests basic insertion and lookup, updating the shared state."""
    key1 = b'hello'
    val1 = b'world'
    
    print(f"  Inserting {key1!r} -> {val1!r}")
    calico_py.insert(key1, val1)
    state[key1] = val1 # Update our expected state
    
    ret_val = calico_py.lookup(key1)
    assert ret_val is not None, "Lookup should find the inserted key"
    assert ret_val == val1, f"Value mismatch for {key1!r}"
    
    ret_val_none = calico_py.lookup(b'does_not_exist')
    assert ret_val_none is None, "Lookup for a non-existent key should return None"

def test_contains(state):
    """Tests the 'contains' functionality, updating the shared state."""
    key = b'existence_test'
    val = b'i_am_here'
    
    print(f"  Inserting {key!r} -> {val!r}")
    calico_py.insert(key, val)
    state[key] = val # Update our expected state

    assert calico_py.contains(key) is True, "Contains should be True for an existing key"
    assert calico_py.contains(b'non_existent_key_for_contains') is False, "Contains should be False for a non-existing key"

def test_remove(state):
    """Tests the 'remove' functionality, updating the shared state."""
    key = b'key_to_be_deleted'
    val = b'temporary_value'
    
    print(f"  Inserting and then removing {key!r}")
    calico_py.insert(key, val)
    # This key is temporary, so we don't add it to the final expected state.
    
    assert calico_py.contains(key) is True, "Key should exist before deletion"

    remove_success = calico_py.remove(key)
    assert remove_success is True, "Remove should return True for an existing key"

    assert calico_py.contains(key) is False, "Key should not exist after deletion"
    
    # Verify that removing a key that was never there also works as expected
    remove_fail = calico_py.remove(b'non_existent_key_for_remove')
    assert remove_fail is False, "Remove should return False for a non-existent key"

def test_scan_with_callback(state):
    """
    Tests the callback-based 'scan' function.
    This test is crucial as it verifies the combined state from all previous tests.
    """
    # 1. Insert more data for the scan test
    scan_data = {
        b'apple': b'red',
        b'banana': b'yellow',
        b'cherry': b'red',
        b'date': b'brown',
        b'elderberry': b'purple'
    }
    print(f"  Inserting {len(scan_data)} new items for scan test...")
    for k, v in scan_data.items():
        calico_py.insert(k, v)
    
    # 2. Update the total expected state with this new data
    state.update(scan_data)
    
    # 3. Prepare the final, sorted list of expected key-value pairs
    final_expected_data = sorted(state.items())
    print(f"  Total expected items in B-Tree: {len(final_expected_data)}")

    # 4. Perform a full scan and verify against the complete expected state
    print("  Testing full scan...")
    scanned_items = []
    def full_scan_cb(key, value):
        scanned_items.append((key, value))
        return True # Continue
    
    calico_py.scan(start_key=b'', callback=full_scan_cb)
    assert scanned_items == final_expected_data, (
        f"Full scan mismatch!\n"
        f"Expected: {final_expected_data}\n"
        f"Got:      {scanned_items}"
    )

def main():
    print("Starting B-Tree Python interface tests...")
    
    # For a clean test run, delete the database file before starting.
    db_file = "/tmp/bm_py"
    if os.path.exists(db_file):
        os.remove(db_file)
        print(f"Removed old database file: {db_file}\n")

    # This dictionary will track the state of the B-Tree across tests.
    expected_db_state = {}

    # Run tests sequentially, passing the state dictionary to each.
    run_test("Insertion and Lookup", test_insert_and_lookup, expected_db_state)
    run_test("Contains Check", test_contains, expected_db_state)
    run_test("Removal", test_remove, expected_db_state)
    run_test("Callback Scan", test_scan_with_callback, expected_db_state)
    
    print("="*30)
    print("All tests completed successfully!")
    print("="*30)

if __name__ == "__main__":
    main()