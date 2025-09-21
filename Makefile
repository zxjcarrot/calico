calico: calico.cpp tpcc/*pp
	g++ -DNDEBUG -O3 -std=c++20 -g -fnon-call-exceptions -fasynchronous-unwind-tables calico.cpp -o calico -laio

clean:
	rm calico
