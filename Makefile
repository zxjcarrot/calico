all: calico calico_oltp_query calico_olap_query

calico: calico.cpp tpcc/*pp
	g++ -DNDEBUG -O3 -std=c++20 -g -fnon-call-exceptions -fasynchronous-unwind-tables -c calico.cpp -o calico.o

calico_oltp_query: calico_oltp_query.cpp tpcc/*pp calico.o
	g++ -DNDEBUG -O3 -std=c++20 -g -fnon-call-exceptions -fasynchronous-unwind-tables calico.o calico_oltp_query.cpp -o calico -laio -luring

calico_olap_query: calico_olap_query.cpp tpcc/*pp calico.o
	g++ -DNDEBUG -O3 -std=c++20 -g -fnon-call-exceptions -fasynchronous-unwind-tables calico.o calico_olap_query.cpp -o calico_olap_query -laio -luring

clean:
	rm -f calico
	rm -f calico_olap_query
