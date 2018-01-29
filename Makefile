
kalman: Makefile kalman.c kalman.h main.c
	gcc -o kalman kalman.c main.c -lgsl -lgslcblas -lm
