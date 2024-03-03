
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2 

mkdir data
wikiextractor enwiki-latest-pages-articles.xml.bz2 -o data --json -n 1000000 

