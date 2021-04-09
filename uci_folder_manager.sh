# will pick the new uci file from main directory & take it to mount folder 
# will del uci1 
# rename the files
# from uci1 -> uci15

cd /home/ubuntu/recSysDB/rawS3data/
sudo cp uci_meta /mnt/prod_uci/check/
cd /mnt/prod_uci/check/
sudo rm uci1
sudo mv uci2 uci1
sudo mv uci3 uci2
sudo mv uci4 uci3
sudo mv uci5 uci4
sudo mv uci6 uci5
sudo mv uci7 uci6
sudo mv uci8 uci7
sudo mv uci9 uci8
sudo mv uci10 uci9
sudo mv uci11 uci10
sudo mv uci12 uci11
sudo mv uci13 uci12
sudo mv uci14 uci13
sudo mv uci15 uci14
sudo mv uci_meta uci15
