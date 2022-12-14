files=`ls imgdata/lu*jpg; ls imgdata/ru*jpg`

for line in ${files}
do
	echo "file==>" $line
      	python3 predict.py ./${line}
	echo "hit enter file==>" $line
	read -p "Hit enter: ==>"
done
