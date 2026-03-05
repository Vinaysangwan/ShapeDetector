.PHONY: build

EXE = ShapeDetector.exe
CONFIG = Debug

all: build compile run

build:
	cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=$(CONFIG)

compile:
	cmake --build build

run:
	./$(EXE)

