#!/bin/bash

bear -- clang++ main.cc -fsanitize=address -o a.out `pkg-config --cflags --libs opencv4` && ./a.out
