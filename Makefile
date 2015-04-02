default: rosenblatt.exe

%.exe: %.cc
	g++ -lm -Wall -Wextra -pedantic --std=c++11 $< -o $@
