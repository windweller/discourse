marker="after"
grep -i "^"$marker"[^a-z]" books_large_p1.txt >> has_marker.txt
grep -i "[^a-z]"$marker"[^a-z]" books_large_p1.txt >> has_marker.txt
grep -i "^"$marker"[^a-z]" books_large_p2.txt >> has_marker.txt
grep -i "[^a-z]"$marker"[^a-z]" books_large_p2.txt >> has_marker.txt
marker="also"
grep -i "^"$marker"[^a-z]" books_large_p1.txt >> has_marker.txt
grep -i "[^a-z]"$marker"[^a-z]" books_large_p1.txt >> has_marker.txt
grep -i "^"$marker"[^a-z]" books_large_p2.txt >> has_marker.txt
grep -i "[^a-z]"$marker"[^a-z]" books_large_p2.txt >> has_marker.txt
marker="although"
grep -i "^"$marker"[^a-z]" books_large_p1.txt >> has_marker.txt
grep -i "[^a-z]"$marker"[^a-z]" books_large_p1.txt >> has_marker.txt
grep -i "^"$marker"[^a-z]" books_large_p2.txt >> has_marker.txt
grep -i "[^a-z]"$marker"[^a-z]" books_large_p2.txt >> has_marker.txt
marker="and"
grep -i "^"$marker"[^a-z]" books_large_p1.txt >> has_marker.txt
grep -i "[^a-z]"$marker"[^a-z]" books_large_p1.txt >> has_marker.txt
grep -i "^"$marker"[^a-z]" books_large_p2.txt >> has_marker.txt
grep -i "[^a-z]"$marker"[^a-z]" books_large_p2.txt >> has_marker.txt
marker="as"
grep -i "^"$marker"[^a-z]" books_large_p1.txt >> has_marker.txt
grep -i "[^a-z]"$marker"[^a-z]" books_large_p1.txt >> has_marker.txt
grep -i "^"$marker"[^a-z]" books_large_p2.txt >> has_marker.txt
grep -i "[^a-z]"$marker"[^a-z]" books_large_p2.txt >> has_marker.txt
marker="because"
grep -i "^"$marker"[^a-z]" books_large_p1.txt >> has_marker.txt
grep -i "[^a-z]"$marker"[^a-z]" books_large_p1.txt >> has_marker.txt
grep -i "^"$marker"[^a-z]" books_large_p2.txt >> has_marker.txt
grep -i "[^a-z]"$marker"[^a-z]" books_large_p2.txt >> has_marker.txt
marker="before"
grep -i "^"$marker"[^a-z]" books_large_p1.txt >> has_marker.txt
grep -i "[^a-z]"$marker"[^a-z]" books_large_p1.txt >> has_marker.txt
grep -i "^"$marker"[^a-z]" books_large_p2.txt >> has_marker.txt
grep -i "[^a-z]"$marker"[^a-z]" books_large_p2.txt >> has_marker.txt
marker="but"
grep -i "^"$marker"[^a-z]" books_large_p1.txt >> has_marker.txt
grep -i "[^a-z]"$marker"[^a-z]" books_large_p1.txt >> has_marker.txt
grep -i "^"$marker"[^a-z]" books_large_p2.txt >> has_marker.txt
grep -i "[^a-z]"$marker"[^a-z]" books_large_p2.txt >> has_marker.txt
marker="for example"
grep -i "^"$marker"[^a-z]" books_large_p1.txt >> has_marker.txt
grep -i "[^a-z]"$marker"[^a-z]" books_large_p1.txt >> has_marker.txt
grep -i "^"$marker"[^a-z]" books_large_p2.txt >> has_marker.txt
grep -i "[^a-z]"$marker"[^a-z]" books_large_p2.txt >> has_marker.txt



marker="before"
grep -ci "^"$marker"[^a-z]" books_large_p1.txt # init1
grep -ci "[^a-z]"$marker"[^a-z]" books_large_p1.txt # total1
grep -ci "^"$marker"[^a-z]" books_large_p2.txt # init2
grep -ci "[^a-z]"$marker"[^a-z]" books_large_p2.txt # internal2
marker="but"
grep -ci "^"$marker"[^a-z]" books_large_p1.txt # init1
grep -ci "[^a-z]"$marker"[^a-z]" books_large_p1.txt # total1
grep -ci "^"$marker"[^a-z]" books_large_p2.txt # init2
grep -ci "[^a-z]"$marker"[^a-z]" books_large_p2.txt # internal2
marker="for example"
grep -ci "^for example[^a-z]" books_large_p1.txt # init1
grep -ci "[^a-z]for example[^a-z]" books_large_p1.txt # total1
grep -ci "^for example[^a-z]" books_large_p2.txt # init2
grep -ci "[^a-z]for example[^a-z]" books_large_p2.txt # internal2
marker="however"
grep -ci "^"$marker"[^a-z]" books_large_p1.txt # init1
grep -ci "[^a-z]"$marker"[^a-z]" books_large_p1.txt # total1
grep -ci "^"$marker"[^a-z]" books_large_p2.txt # init2
grep -ci "[^a-z]"$marker"[^a-z]" books_large_p2.txt # internal2
marker="if"
grep -ci "^"$marker"[^a-z]" books_large_p1.txt # init1
grep -ci "[^a-z]"$marker"[^a-z]" books_large_p1.txt # total1
grep -ci "^"$marker"[^a-z]" books_large_p2.txt # init2
grep -ci "[^a-z]"$marker"[^a-z]" books_large_p2.txt # internal2
marker="meanwhile"
grep -ci "^"$marker"[^a-z]" books_large_p1.txt # init1
grep -ci "[^a-z]"$marker"[^a-z]" books_large_p1.txt # total1
grep -ci "^"$marker"[^a-z]" books_large_p2.txt # init2
grep -ci "[^a-z]"$marker"[^a-z]" books_large_p2.txt # internal2
marker="so"
grep -ci "^"$marker"[^a-z]" books_large_p1.txt # init1
grep -ci "[^a-z]"$marker"[^a-z]" books_large_p1.txt # total1
grep -ci "^"$marker"[^a-z]" books_large_p2.txt # init2
grep -ci "[^a-z]"$marker"[^a-z]" books_large_p2.txt # internal2
marker="still"
grep -ci "^"$marker"[^a-z]" books_large_p1.txt # init1
grep -ci "[^a-z]"$marker"[^a-z]" books_large_p1.txt # total1
grep -ci "^"$marker"[^a-z]" books_large_p2.txt # init2
grep -ci "[^a-z]"$marker"[^a-z]" books_large_p2.txt # internal2
marker="then"
grep -ci "^"$marker"[^a-z]" books_large_p1.txt # init1
grep -ci "[^a-z]"$marker"[^a-z]" books_large_p1.txt # total1
grep -ci "^"$marker"[^a-z]" books_large_p2.txt # init2
grep -ci "[^a-z]"$marker"[^a-z]" books_large_p2.txt # internal2
marker="though"
grep -ci "^"$marker"[^a-z]" books_large_p1.txt # init1
grep -ci "[^a-z]"$marker"[^a-z]" books_large_p1.txt # total1
grep -ci "^"$marker"[^a-z]" books_large_p2.txt # init2
grep -ci "[^a-z]"$marker"[^a-z]" books_large_p2.txt # internal2
marker="when"
grep -ci "^"$marker"[^a-z]" books_large_p1.txt # init1
grep -ci "[^a-z]"$marker"[^a-z]" books_large_p1.txt # total1
grep -ci "^"$marker"[^a-z]" books_large_p2.txt # init2
grep -ci "[^a-z]"$marker"[^a-z]" books_large_p2.txt # internal2
marker="while"
grep -ci "^"$marker"[^a-z]" books_large_p1.txt # init1
grep -ci "[^a-z]"$marker"[^a-z]" books_large_p1.txt # total1
grep -ci "^"$marker"[^a-z]" books_large_p2.txt # init2
grep -ci "[^a-z]"$marker"[^a-z]" books_large_p2.txt # internal2