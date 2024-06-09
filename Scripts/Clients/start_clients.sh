#!/bin/bash
echo "starting client1.py"
python client1.py --partition= 1 &
echo "starting client2.py"
python client2.py --partition= 2 &
echo "starting client3.py"
python client3.py --partition= 3 &
echo "starting client4.py"
python client4.py --partition= 4 &
echo "starting client5.py"
python client5.py --partition= 5 &
echo "starting client6.py"
python client6.py --partition= 6 &
echo "starting client7.py"
python client7.py --partition= 7 &
echo "starting client8.py"
python client8.py --partition= 8 &
echo "starting client9.py"
python client9.py --partition= 9 &
echo "starting client10.py"
python client10.py --partition= 10 &

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
wait