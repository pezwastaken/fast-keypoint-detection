#!/bin/bash

bear -- clang++-19 -std=c++20 main.cc -fsanitize=address -o a.out `pkg-config --cflags --libs opencv4`
./a.out
