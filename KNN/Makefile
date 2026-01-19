CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2

TARGET = main
SRC = main.cpp

# Default arguments (can be overridden)
N ?= 100
D ?= 3
NTEST ?= 20
K ?= 3

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

run: $(TARGET)
	./$(TARGET) $(N) $(D) $(NTEST) $(K)

clean:
	rm -f $(TARGET)

