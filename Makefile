.PHONY: all clean

all: data

clean:
	rm -rf data/raw/*
	touch data/raw/.gitkeep

data: wiki glove

wiki:
	wget -N -c -P data/raw/ https://raw.githubusercontent.com/louismartin/dress-data/master/data-simplification.tar.bz2
	tar -xvjf data/raw/data-simplification.tar.bz2 -C data/raw
	rm data/raw/data-simplification.tar.bz2

glove:
	wget -N -c -P data/raw/ http://nlp.stanford.edu/data/glove.6B.zip
	unzip data/raw/glove.6B.zip -d data/raw
	rm data/raw/glove.6B.zip
