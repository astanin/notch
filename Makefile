default: rosenblatt.exe

%.exe: %.cc *.hh
	g++ -g -lm -Wall -Wextra -pedantic --std=c++11 $< -o $@
