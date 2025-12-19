all: calico_oltp_query calico_olap_query

calico.o: calico.cpp tpcc/*pp
	g++ -DNDEBUG -O3 -std=c++20 -g -fnon-call-exceptions -fasynchronous-unwind-tables -c calico.cpp -o calico.o 
#-DCALICO_SPECIALIZATION_CALICO

calico_oltp_query: calico_oltp_query.cpp tpcc/*pp calico.o
	g++ -DNDEBUG -O3 -std=c++20 -g -fnon-call-exceptions -fasynchronous-unwind-tables calico.o calico_oltp_query.cpp -o calico_oltp_query -laio -luring -labsl_hash -labsl_raw_hash_set -labsl_raw_logging_internal -pthread #
calico_olap_query: calico_olap_query.cpp tpcc/*pp calico.o
	g++ -DNDEBUG -O3 -std=c++20 -g -fnon-call-exceptions -fasynchronous-unwind-tables calico.o calico_olap_query.cpp -o calico_olap_query -laio -luring -labsl_hash -labsl_raw_hash_set -labsl_raw_logging_internal -pthread

clean:
	rm -f calico_oltp_query
	rm -f calico_olap_query
	rm -f calico.o
