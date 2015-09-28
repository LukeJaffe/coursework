def bday(n):
    s = 1.0
    for i in range(1,n):
        s *= 1.0 - float(i)/365.0
    return 1.0 - s

for i in range(25):
    p = bday(i)
    if p > 0.5:
        print "# people:", i
        print "P[2 have same birthday] =", p
        break
