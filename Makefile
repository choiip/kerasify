CC=g++
CFLAGS=-c -std=c++17 -O3 -ffast-math -fno-rtti \
    -I./include \
    -W -Wall -Wextra -Wpedantic \
    -Wc++17-compat
LDFLAGS=

SOURCES=$(shell find . -type f -name '*.cpp' ! -path './build/*')
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=kerasify

all: clean_build

minimal: CFLAGS := $(CFLAGS) -Os \
    -fno-math-errno \
    -fno-stack-protector -fno-ident -fomit-frame-pointer
minimal: clean_build

clean_build: $(SOURCES) $(EXECUTABLE)
	find . -type f -name '*.o' ! -path './build/*' -delete

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	find . -type f -name '*.o' ! -path './build/*' -delete
	rm -f kerasify

