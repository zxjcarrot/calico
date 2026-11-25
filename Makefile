ARCH_FLAGS ?= -march=native
CXX ?= g++
STATIC_RUNTIME ?= 1
CXXFLAGS :=  -DNDEBUG -O3 -std=c++23 -g3 -fno-omit-frame-pointer -fnon-call-exceptions -fasynchronous-unwind-tables $(ARCH_FLAGS)
ifeq ($(PATH_CACHE),1)
CXXFLAGS += -DCALICO_PATH_CACHE
endif
ifeq ($(TRADHASH),0)
CXXFLAGS += -DCALICO_DIRECT_PAGE_STATE
endif
LDFLAGS :=
ifeq ($(STATIC_RUNTIME),1)
LDFLAGS += -static-libgcc -static-libstdc++
endif

all: calico_oltp_query calico_olap_query calico_oltp_query_special calico_oltp_query_special_pad64

calico.o: calico.cpp calico.hpp hash_tables.hpp third_party/robin_hood.h third_party/concurrentqueue.hpp tpcc/*pp
	$(CXX) $(CXXFLAGS) -c calico.cpp -o calico.o
# -DCALICO_SPECIALIZATION_CALICO

calico_special.o: calico.cpp calico.hpp hash_tables.hpp third_party/robin_hood.h third_party/concurrentqueue.hpp tpcc/*pp
	$(CXX) $(CXXFLAGS) -c calico.cpp -o calico_special.o  -DCALICO_SPECIALIZATION_CALICO

calico_special_pad64.o: calico.cpp calico.hpp hash_tables.hpp third_party/robin_hood.h third_party/concurrentqueue.hpp tpcc/*pp
	$(CXX) $(CXXFLAGS) -c calico.cpp -o calico_special_pad64.o -DCALICO_SPECIALIZATION_CALICO -DCALICO_PAGESTATE_ALIGN64

calico_oltp_query: calico_oltp_query.cpp tpcc/*pp calico.o
	$(CXX) $(CXXFLAGS) calico.o calico_oltp_query.cpp $(LDFLAGS) -o calico_oltp_query -laio -luring -labsl_hash -labsl_raw_hash_set -labsl_raw_logging_internal -pthread #

calico_oltp_query_special: calico_oltp_query.cpp tpcc/*pp calico_special.o
	$(CXX) $(CXXFLAGS) calico_special.o calico_oltp_query.cpp $(LDFLAGS) -o calico_oltp_query_special -laio -luring -labsl_hash -labsl_raw_hash_set -labsl_raw_logging_internal -pthread #

calico_oltp_query_special_pad64: calico_oltp_query.cpp tpcc/*pp calico_special_pad64.o
	$(CXX) $(CXXFLAGS) calico_special_pad64.o calico_oltp_query.cpp $(LDFLAGS) -o calico_oltp_query_special_pad64 -laio -luring -labsl_hash -labsl_raw_hash_set -labsl_raw_logging_internal -pthread #

calico_olap_query: calico_olap_query.cpp tpcc/*pp calico.o
	$(CXX) $(CXXFLAGS) calico.o calico_olap_query.cpp $(LDFLAGS) -o calico_olap_query -laio -luring -labsl_hash -labsl_raw_hash_set -labsl_raw_logging_internal -pthread

clean:
	rm -f calico_oltp_query_special
	rm -f calico_oltp_query_special_pad64
	rm -f calico_oltp_query
	rm -f calico_olap_query
	rm -f calico.o
	rm -f calico_special.o
	rm -f calico_special_pad64.o
