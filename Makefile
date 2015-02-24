PROG	= cpp_sim
CU_PROG	= cu_sim


CC		= g++
CFLAGS	+= -Wall -std=c++11
SRCS	= simulation.cpp save_results.cpp 
CU_SRC	= cu_simulation.cu
OBJS	:= ${SRCS:.cpp=.o}
LD_PATH	+= -L/home/kosta/usr/lib
LIBS	+= -lboost_system
CU_LIBS += -lcurand

GRAPH_SCRIPT=plot.gp
GRAPH_FILE=graph.eps

V_DATA=V_sim_cu V_sim_nif_cu V_sim_nif_cpp V_sim_cpp


.o: $(SRCS)
	$(CC) -c $(CFLAGS) $^

cpp_sim: .o
	$(CC) -o $@ $(OBJS) $(LD_PATH) $(LIBS)

cu_sim: save_results.o
	nvcc $(CU_SRC) -o $@ --compiler-options $(CFLAGS) $(LD_PATH) $< $(CU_LIBS) $(LIBS)

all: cpp_sim cu_sim

cpp_exec:
		./$(OUTPUTCPP)

cu_exec:
		./$(OUTPUTCU)

graph:
	gnuplot $(GRAPH_SCRIPT)

send:
	scp $(V_DATA) selyunin@phobos:/home/selyunin/Dropbox/thesis/work/papers/TW_simulation/TW_GPU/

view:
	evince $(GRAPH_FILE)

clean:
	@rm -f *~ .*~
	@rm -f *.o
	@rm -f $(PROG) $(CU_PROG)
	@rm -f $(GRAPH_FILE)
	@rm -f $(V_DATA)
