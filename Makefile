.PHONY: all clean

all: data/raw/data-simplification.tar.bz2

clean:
	rm -rf data/raw/*
	touch data/raw/.gitkeep

data/raw/data-simplification.tar.bz2:
	python cleartext/utils/datasets.py data/ $@