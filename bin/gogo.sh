
if [ $# -ne 1 ]; then
	echo "USAGE: weight file"
	exit 1
fi


files=`ls imgdata/lu*jpg; ls imgdata/ru*jpg`
weight=$1



for line in ${files}
do
	echo "file==>" $line
	./bin/edged.py ${line}
	cp ${line} ./target.jpg
   	python3 predict.py ${weight} ./edged.jpg
	echo "hit enter file==>" $line
	read -p "Hit enter: ==>"
done
