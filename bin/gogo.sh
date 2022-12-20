
if [ $# -ne 1 ]; then
	echo "USAGE: weight file"
	exit 1
fi


files=`ls imgdata/lu*jpg; ls imgdata/ru*jpg`
weight=$1



for line in ${files}
do
	echo "file==>" $line
      	python3 predict.py ${weight} ./${line}
	echo "hit enter file==>" $line
	read -p "Hit enter: ==>"
done
