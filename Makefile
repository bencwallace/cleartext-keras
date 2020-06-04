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
	wget -N -c -P models/glove/ http://nlp.stanford.edu/data/glove.6B.zip
	unzip models/glove/glove.6B.zip -d models/glove/
	rm models/glove/glove.6B.zip
