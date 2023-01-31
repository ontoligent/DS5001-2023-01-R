#! /bin/sh

#python -c "import nltk; nltk.help.upenn_tagset()" | egrep '^\S.*:' | sed 's/^\(.+\):/\1\t/' > upenn_tagset.txt
python -c "import nltk; nltk.help.upenn_tagset()" | egrep '^\S.*:' | sed 's/:/\t/' | sed 's/\t:/:\t/' > upenn_tagset.txt
