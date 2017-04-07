cat train.txt | awk -F, '{a[NF]=a[NF]+1;}END{for(i=1; i<=60;i++)print i,a[i]}'
#cat test.txt | awk -F, '{a[NF]=a[NF]+1;}END{for(i=1; i<=60;i++)print i,a[i]}'
